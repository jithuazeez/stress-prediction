"""
Fine-tuning and LOSO evaluation for Subject-Aware SSL.

Loads pre-trained encoder, adds classification head, and evaluates
using Leave-One-Subject-Out cross-validation.

Usage:
    python train.py --mode invariant --window_size 120 --horizon 3
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import json
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import get_all_subjects
from shared.logging_utils import setup_logger, log_experiment_start, log_experiment_end
from shared.evaluation import (
    evaluate_predictions, aggregate_fold_metrics, save_results,
    save_predictions, plot_results, find_optimal_threshold
)

from config import SSLConfig, SSLMode, DEFAULT_SSL_CONFIG, get_run_name
from dataset import load_windows_at_8hz, SSLDataset
from encoder import (
    SSLEncoder, ClassificationHead, SSLModel,
    create_ssl_encoder, count_parameters
)


def train_epoch(
    model: SSLModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Run one training epoch.
    
    Args:
        model: SSL model (encoder + classification head)
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, labels, _ in dataloader:
        x = x.to(device).float()  # Ensure float32 to avoid dtype mismatch
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate_model(
    model: SSLModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on data.
    
    Args:
        model: SSL model
        dataloader: Data loader
        device: Torch device
        threshold: Classification threshold
    
    Returns:
        Tuple of (y_true, y_pred, y_proba)
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, labels, _ in dataloader:
            x = x.to(device).float()  # Ensure float32 to avoid dtype mismatch
            
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Prob of positive class
            
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_proba = np.array(all_probs)
    y_pred = (y_proba >= threshold).astype(int)
    
    return y_true, y_pred, y_proba


def compute_training_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute key metrics for training monitoring.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
    
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # G-mean
    gmean = np.sqrt(sensitivity * specificity)
    
    # Balanced accuracy
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        "precision": precision,
        "recall": sensitivity,
        "gmean": gmean,
        "balanced_accuracy": balanced_acc,
        "specificity": specificity
    }


def finetune_fold(
    encoder: SSLEncoder,
    train_windows: List[Dict],
    test_windows: List[Dict],
    config: SSLConfig,
    subject_to_idx: Dict[str, int],
    device: torch.device,
    logger,
    fold_name: str
) -> Dict:
    """
    Fine-tune on one LOSO fold.
    
    Args:
        encoder: Pre-trained encoder (will be copied, not modified)
        train_windows: Training windows
        test_windows: Test windows
        config: SSL configuration
        subject_to_idx: Subject ID to index mapping
        device: Torch device
        logger: Logger instance
        fold_name: Name for this fold
    
    Returns:
        Dictionary of metrics for this fold
    """
    # Copy encoder to avoid modifying original
    encoder_copy = create_ssl_encoder(
        input_channels=config.n_channels,
        embedding_dim=config.embedding_dim
    ).to(device)
    encoder_copy.load_state_dict(encoder.state_dict())
    
    # Create classification head
    cls_head = ClassificationHead(
        input_dim=config.embedding_dim,
        hidden_dim=64,
        n_classes=2,
        dropout=0.3
    ).to(device)
    
    model = SSLModel(encoder_copy, cls_head)
    
    # Create datasets
    train_dataset = SSLDataset(train_windows, config, subject_to_idx)
    test_dataset = SSLDataset(test_windows, config, subject_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.finetune_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.finetune_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    # Class weights for imbalanced data
    n_pos = (train_dataset.labels == 1).sum()
    n_neg = (train_dataset.labels == 0).sum()
    
    if n_pos > 0 and n_neg > 0:
        pos_weight = np.sqrt(n_neg / n_pos)
    else:
        pos_weight = 1.0
    
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with discriminative learning rates
    param_groups = [
        {"params": encoder_copy.parameters(), "lr": config.finetune_lr * config.finetune_encoder_lr_factor},
        {"params": cls_head.parameters(), "lr": config.finetune_lr}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=config.finetune_weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Training loop with early stopping
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config.finetune_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            # Save best model state
            best_state = {
                "encoder": encoder_copy.state_dict(),
                "head": cls_head.state_dict()
            }
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            break
    
    # Load best model
    encoder_copy.load_state_dict(best_state["encoder"])
    cls_head.load_state_dict(best_state["head"])
    model = SSLModel(encoder_copy, cls_head)
    
    # Get predictions on training data
    y_train_true, _, y_train_proba = evaluate_model(model, train_loader, device)
    
    # Try different threshold methods
    threshold_methods = ["youden", "f1", "balanced", "geometric_mean"]
    threshold_results = {}
    
    logger.info(f"  Comparing threshold methods on training data:")
    
    for method in threshold_methods:
        # Find threshold with this method
        thresh, _ = find_optimal_threshold(y_train_true, y_train_proba, method=method)
        
        # Apply to training data to evaluate
        _, y_train_pred, _ = evaluate_model(model, train_loader, device, threshold=thresh)
        train_metrics = compute_training_metrics(y_train_true, y_train_pred, y_train_proba)
        
        threshold_results[method] = {
            "threshold": thresh,
            "gmean": train_metrics["gmean"],
            "recall": train_metrics["recall"],
            "precision": train_metrics["precision"],
            "balanced_accuracy": train_metrics["balanced_accuracy"],
            "specificity": train_metrics["specificity"]
        }
        
        logger.info(
            f"    {method:15s}: thresh={thresh:.3f}, "
            f"G-mean={train_metrics['gmean']:.3f}, "
            f"Recall={train_metrics['recall']:.3f}, "
            f"Prec={train_metrics['precision']:.3f}, "
            f"Spec={train_metrics['specificity']:.3f}"
        )
    
    # Choose best method based on G-mean (highest balance)
    best_method = max(threshold_results.keys(), key=lambda m: threshold_results[m]["gmean"])
    optimal_threshold = threshold_results[best_method]["threshold"]
    
    logger.info(f"  → Selected method: {best_method} with G-mean={threshold_results[best_method]['gmean']:.3f}")
    
    # Evaluate on test set with selected threshold
    y_true, y_pred, y_proba = evaluate_model(model, test_loader, device, threshold=optimal_threshold)
    
    # Compute test metrics
    metrics = evaluate_predictions(y_true, y_pred, y_proba, fold_name, threshold=optimal_threshold)
    
    # Add G-mean to metrics
    test_metrics = compute_training_metrics(y_true, y_pred, y_proba)
    metrics["gmean"] = test_metrics["gmean"]
    
    # Save threshold method info
    metrics["threshold_method"] = best_method
    metrics["optimal_threshold"] = optimal_threshold
    metrics["train_epochs"] = epoch + 1
    
    # Add all threshold comparison results
    for method, results in threshold_results.items():
        metrics[f"train_{method}_threshold"] = results["threshold"]
        metrics[f"train_{method}_gmean"] = results["gmean"]
    
    return metrics, y_true, y_pred, y_proba


def loso_evaluation(
    encoder: SSLEncoder,
    windows: List[Dict],
    subject_to_idx: Dict[str, int],
    config: SSLConfig,
    device: torch.device,
    logger
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run Leave-One-Subject-Out cross-validation.
    
    Args:
        encoder: Pre-trained encoder
        windows: All windows
        subject_to_idx: Subject ID to index mapping
        config: SSL configuration
        device: Torch device
        logger: Logger instance
    
    Returns:
        Tuple of (aggregated_metrics, fold_metrics_df)
    """
    logger.info(f"\n{'='*60}")
    logger.info("LOSO CROSS-VALIDATION")
    logger.info(f"{'='*60}")
    
    subjects = list(subject_to_idx.keys())
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    
    pbar = tqdm(subjects, desc="LOSO folds", unit="fold")
    
    for test_subject in pbar:
        pbar.set_postfix({"subject": test_subject[:8]})
        
        # Split data
        train_windows = [w for w in windows if w.get("subject_id") != test_subject]
        test_windows = [w for w in windows if w.get("subject_id") == test_subject]
        
        if len(train_windows) == 0 or len(test_windows) == 0:
            logger.warning(f"Skipping {test_subject}: empty split")
            continue
        
        # Fine-tune and evaluate
        metrics, y_true, y_pred, y_proba = finetune_fold(
            encoder, train_windows, test_windows,
            config, subject_to_idx, device, logger,
            fold_name=f"fold_{test_subject[:8]}"
        )
        
        metrics["test_subject"] = test_subject
        fold_metrics.append(metrics)
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_true))
        
        logger.info(
            f"  {test_subject[:8]}: "
            f"AUROC={metrics.get('auroc', 0):.3f}, "
            f"PR-AUC={metrics.get('pr_auc', 0):.3f}, "
            f"F1={metrics.get('f1', 0):.3f}, "
            f"G-mean={metrics.get('gmean', 0):.3f}, "
            f"Method={metrics.get('threshold_method', 'unknown')}"
        )
    
    pbar.close()
    
    # Aggregate metrics
    aggregated = aggregate_fold_metrics(fold_metrics)
    
    # Create fold metrics DataFrame
    fold_df = pd.DataFrame(fold_metrics)
    
    # Overall predictions
    overall_metrics = evaluate_predictions(
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_proba),
        "ssl_overall"
    )
    aggregated.update({f"overall_{k}": v for k, v in overall_metrics.items()})
    
    return aggregated, fold_df, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba), np.array(all_subjects)


def load_pretrained_encoder(
    config: SSLConfig,
    device: torch.device,
    logger
) -> SSLEncoder:
    """
    Load pre-trained encoder from checkpoint.
    
    Args:
        config: SSL configuration
        device: Torch device
        logger: Logger instance
    
    Returns:
        Pre-trained encoder
    """
    run_name = get_run_name(config)
    encoder_path = config.results_path / "checkpoints" / f"pretrain_encoder_{run_name}.pt"
    
    if not encoder_path.exists():
        logger.error(f"Pre-trained encoder not found at {encoder_path}")
        raise FileNotFoundError(f"Pre-trained encoder not found at {encoder_path}")
        
    else:
        logger.info(f"Loading pre-trained encoder from {encoder_path}")
        encoder = create_ssl_encoder(
            input_channels=config.n_channels,
            embedding_dim=config.embedding_dim
        ).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    
    return encoder


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="SSL Fine-tuning and Evaluation")
    parser.add_argument(
        "--mode", type=str, default="invariant",
        choices=["base", "invariant", "specific"],
        help="SSL mode (must match pre-trained encoder)"
    )
    parser.add_argument(
        "--window_size", type=int, default=120,
        help="Window size in seconds (60 or 120)"
    )
    parser.add_argument(
        "--horizon", type=int, default=3,
        help="Prediction horizon in minutes (3 or 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = DEFAULT_SSL_CONFIG
    config.ssl_mode = SSLMode(args.mode)
    config.window_size_sec = args.window_size
    config.target_label = f"label_{args.horizon}min"
    config.finetune_epochs = args.epochs
    config.finetune_batch_size = args.batch_size
    config.finetune_lr = args.lr
    
    # Setup directories
    config.results_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    run_name = get_run_name(config)
    log_file = config.results_path / f"train_{run_name}.log"
    logger = setup_logger("ssl_train", log_file=log_file)
    
    log_experiment_start(logger, "SUBJECT-AWARE SSL FINE-TUNING")
    
    logger.info(f"Configuration:")
    logger.info(f"  Mode: {config.ssl_mode.value}")
    logger.info(f"  Window: {config.window_size_sec}s")
    logger.info(f"  Horizon: {args.horizon} min")
    logger.info(f"  Target: {config.target_label}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Load data
    logger.info(f"\n{'-'*50}")
    logger.info("Loading data at 8Hz...")
    logger.info(f"{'-'*50}")
    
    windows, subject_to_idx = load_windows_at_8hz(config)
    
    if not windows:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(windows)} windows from {len(subject_to_idx)} subjects")
    
    # Load pre-trained encoder
    encoder = load_pretrained_encoder(config, device, logger)
    logger.info(f"Encoder parameters: {count_parameters(encoder):,}")
    
    # LOSO evaluation
    aggregated, fold_df, y_true, y_pred, y_proba, subjects = loso_evaluation(
        encoder, windows, subject_to_idx, config, device, logger
    )
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"AUROC: {aggregated.get('auroc_mean', 0):.3f} ± {aggregated.get('auroc_std', 0):.3f}")
    logger.info(f"PR-AUC: {aggregated.get('pr_auc_mean', 0):.3f} ± {aggregated.get('pr_auc_std', 0):.3f}")
    logger.info(f"F1: {aggregated.get('f1_mean', 0):.3f} ± {aggregated.get('f1_std', 0):.3f}")
    logger.info(f"Recall: {aggregated.get('recall_mean', 0):.3f} ± {aggregated.get('recall_std', 0):.3f}")
    logger.info(f"Precision: {aggregated.get('precision_mean', 0):.3f} ± {aggregated.get('precision_std', 0):.3f}")
    
    # Save results
    save_results(aggregated, config.results_path, f"ssl_{run_name}")
    
    fold_df.to_csv(config.results_path / f"ssl_{run_name}_fold_metrics.csv", index=False)
    logger.info(f"Saved fold metrics to {config.results_path / f'ssl_{run_name}_fold_metrics.csv'}")
    
    save_predictions(y_true, y_pred, y_proba, subjects, config.results_path, f"ssl_{run_name}")
    
    plot_results(y_true, y_proba, f"ssl_{run_name}", config.results_path, y_pred)
    
    log_experiment_end(logger, "SUBJECT-AWARE SSL FINE-TUNING")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SSL {config.ssl_mode.value.upper()} - LOSO Results")
    print(f"{'='*60}")
    print(f"Window: {config.window_size_sec}s | Horizon: {args.horizon}min")
    print(f"AUROC: {aggregated.get('auroc_mean', 0):.3f} ± {aggregated.get('auroc_std', 0):.3f}")
    print(f"PR-AUC: {aggregated.get('pr_auc_mean', 0):.3f} ± {aggregated.get('pr_auc_std', 0):.3f}")
    print(f"F1: {aggregated.get('f1_mean', 0):.3f} ± {aggregated.get('f1_std', 0):.3f}")
    print(f"Results saved to: {config.results_path}")


if __name__ == "__main__":
    main()
