"""
Training script for Multi-Rate Late Fusion.

Uses Leave-One-Subject-Out cross-validation to evaluate the model.

Usage:
    python train.py --window_size 120 --horizon 3
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
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
from tqdm import tqdm

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import get_all_subjects
from shared.logging_utils import setup_logger, log_experiment_start, log_experiment_end
from shared.evaluation import (
    evaluate_predictions, aggregate_fold_metrics, save_results,
    save_predictions, plot_results, find_optimal_threshold
)

from config import MultiRateConfig, DEFAULT_MULTIRATE_CONFIG, get_run_name
from dataset import load_windows_multirate, MultiRateDataset, collate_multirate
from model import MultiRateFusionModel, create_model
from encoders import count_parameters


def train_epoch(
    model: MultiRateFusionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Run one training epoch.
    
    Args:
        model: Multi-rate fusion model
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
    
    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate_model(
    model: MultiRateFusionModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on data.
    
    Args:
        model: Multi-rate fusion model
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
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            probs = model.predict_proba(batch)[:, 1]  # Prob of positive class
            
            all_labels.extend(batch["label"].cpu().numpy())
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


def train_fold(
    train_windows: List[Dict],
    test_windows: List[Dict],
    config: MultiRateConfig,
    subject_to_idx: Dict[str, int],
    device: torch.device,
    logger,
    fold_name: str
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train and evaluate on one LOSO fold.
    
    Args:
        train_windows: Training windows
        test_windows: Test windows
        config: Multi-rate configuration
        subject_to_idx: Subject ID to index mapping
        device: Torch device
        logger: Logger instance
        fold_name: Name for this fold
    
    Returns:
        Tuple of (metrics, y_true, y_pred, y_proba)
    """
    # Create datasets
    train_dataset = MultiRateDataset(train_windows, config, subject_to_idx)
    test_dataset = MultiRateDataset(test_windows, config, subject_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_multirate,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_multirate,
        num_workers=0
    )
    
    # Create model
    model = create_model(config).to(device)
    
    # Class weights for imbalanced data
    n_pos = (train_dataset.labels == 1).sum()
    n_neg = (train_dataset.labels == 0).sum()
    
    if n_pos > 0 and n_neg > 0:
        pos_weight = n_neg / n_pos
    else:
        pos_weight = 1.0
    
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Training loop with early stopping
    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    
    logger.info(f"  Training {fold_name}...")
    
    for epoch in range(config.n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)
        
        # Log metrics every 10 epochs or at the first epoch
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # Evaluate on training set to monitor progress
            y_train_true, _, y_train_proba = evaluate_model(model, train_loader, device, threshold=0.5)
            train_threshold, _ = find_optimal_threshold(y_train_true, y_train_proba, method="geometric_mean")
            _, y_train_pred, _ = evaluate_model(model, train_loader, device, threshold=train_threshold)
            
            train_metrics = compute_training_metrics(y_train_true, y_train_pred, y_train_proba)
            
            logger.info(
                f"    Epoch {epoch+1:3d}/{config.n_epochs}: "
                f"Loss={train_loss:.4f}, "
                f"Recall={train_metrics['recall']:.3f}, "
                f"Precision={train_metrics['precision']:.3f}, "
                f"G-mean={train_metrics['gmean']:.3f}, "
                f"BalAcc={train_metrics['balanced_accuracy']:.3f}"
            )
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Get predictions on training data
    y_train_true, _, y_train_proba = evaluate_model(model, train_loader, device)
    
    # Use constrained gmean as default (same as MOMENT)
    # This ensures minimum recall (0.85) while maintaining low false positive rate (max 0.20)
    optimal_threshold, _ = find_optimal_threshold(
        y_train_true, y_train_proba,
        method="constrained_gmean",
        min_recall=0.85,
        max_fpr=0.20
    )
    
    # Evaluate with optimal threshold
    _, y_train_pred, _ = evaluate_model(model, train_loader, device, threshold=optimal_threshold)
    train_metrics = compute_training_metrics(y_train_true, y_train_pred, y_train_proba)
    
    logger.info(
        f"  Constrained G-mean threshold: {optimal_threshold:.4f} "
        f"(min_recall=0.85, max_fpr=0.20)"
    )
    logger.info(
        f"  Training metrics: "
        f"G-mean={train_metrics['gmean']:.3f}, "
        f"Recall={train_metrics['recall']:.3f}, "
        f"Precision={train_metrics['precision']:.3f}, "
        f"Specificity={train_metrics['specificity']:.3f}"
    )
    
    # Evaluate on test set with selected threshold
    y_true, y_pred, y_proba = evaluate_model(model, test_loader, device, threshold=optimal_threshold)
    
    # Compute test metrics
    metrics = evaluate_predictions(y_true, y_pred, y_proba, fold_name, threshold=optimal_threshold)
    
    # Add G-mean to metrics
    test_metrics = compute_training_metrics(y_true, y_pred, y_proba)
    metrics["gmean"] = test_metrics["gmean"]
    
    # Save threshold method info
    metrics["threshold_method"] = "constrained_gmean"
    metrics["optimal_threshold"] = optimal_threshold
    metrics["train_epochs"] = epoch + 1
    metrics["min_recall"] = 0.85
    metrics["max_fpr"] = 0.20
    
    return metrics, y_true, y_pred, y_proba


def loso_evaluation(
    windows: List[Dict],
    subject_to_idx: Dict[str, int],
    config: MultiRateConfig,
    device: torch.device,
    logger
) -> Tuple[Dict, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Leave-One-Subject-Out cross-validation.
    
    Args:
        windows: All windows
        subject_to_idx: Subject ID to index mapping
        config: Multi-rate configuration
        device: Torch device
        logger: Logger instance
    
    Returns:
        Tuple of (aggregated_metrics, fold_df, y_true, y_pred, y_proba, subjects)
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
        
        # Train and evaluate
        metrics, y_true, y_pred, y_proba = train_fold(
            train_windows, test_windows,
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
            f"Recall={metrics.get('recall', 0):.3f}, "
            f"Method={metrics.get('threshold_method', 'unknown')}"
        )
    
    pbar.close()
    
    # Aggregate metrics
    aggregated = aggregate_fold_metrics(fold_metrics)
    
    # Create fold metrics DataFrame
    fold_df = pd.DataFrame(fold_metrics)
    
    # Overall predictions using fold-level predictions (no re-thresholding)
    # Ensures: overall_metrics ≈ average(fold_metrics)
    overall_metrics = evaluate_predictions(
        np.array(all_y_true),
        np.array(all_y_pred),  # Predictions made with fold-specific thresholds
        np.array(all_y_proba),
        "multirate_overall"
    )
    
    # Add threshold statistics to aggregated metrics
    fold_thresholds = [f.get("threshold", 0.5) for f in fold_metrics]
    overall_metrics["threshold_mean"] = float(np.mean(fold_thresholds))
    overall_metrics["threshold_std"] = float(np.std(fold_thresholds))
    overall_metrics["threshold_min"] = float(np.min(fold_thresholds))
    overall_metrics["threshold_max"] = float(np.max(fold_thresholds))
    
    aggregated.update({f"overall_{k}": v for k, v in overall_metrics.items()})
    
    return (
        aggregated, fold_df,
        np.array(all_y_true), np.array(all_y_pred),
        np.array(all_y_proba), np.array(all_subjects)
    )


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Multi-Rate Fusion Training")
    parser.add_argument(
        "--window_size", type=int, default=120,
        help="Window size in seconds (60 or 120)"
    )
    parser.add_argument(
        "--horizon", type=int, default=3,
        help="Prediction horizon in minutes (3 or 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = DEFAULT_MULTIRATE_CONFIG.get_config_for_run(
        args.window_size, args.horizon
    )
    config.n_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Setup directories
    config.results_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    run_name = get_run_name(config)
    log_file = config.results_path / f"train_{run_name}.log"
    logger = setup_logger("multirate_train", log_file=log_file)
    
    log_experiment_start(logger, "MULTI-RATE LATE FUSION TRAINING")
    
    logger.info(f"Configuration:")
    logger.info(f"  Window: {config.window_size_sec}s")
    logger.info(f"  Horizon: {args.horizon} min")
    logger.info(f"  Target: {config.target_label}")
    logger.info(f"  ACC samples: {config.acc_samples_per_window} (3 channels at 32Hz)")
    logger.info(f"  Physio samples: {config.physio_samples_per_window} (5 channels at 1Hz)")
    logger.info(f"  Threshold method: constrained_gmean (min_recall=0.85, max_fpr=0.20)")
    logger.info(f"  Channels: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Load data
    logger.info(f"\n{'-'*50}")
    logger.info("Loading data at native rates...")
    logger.info(f"{'-'*50}")
    
    windows, subject_to_idx = load_windows_multirate(config)
    
    if not windows:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(windows)} windows from {len(subject_to_idx)} subjects")
    
    # Create model for parameter count
    test_model = create_model(config)
    logger.info(f"\nModel parameters: {count_parameters(test_model):,}")
    del test_model
    
    # LOSO evaluation
    aggregated, fold_df, y_true, y_pred, y_proba, subjects = loso_evaluation(
        windows, subject_to_idx, config, device, logger
    )
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"AUROC: {aggregated.get('auroc_mean', 0):.3f} ± {aggregated.get('auroc_std', 0):.3f}")
    logger.info(f"PR-AUC: {aggregated.get('pr_auc_mean', 0):.3f} ± {aggregated.get('pr_auc_std', 0):.3f}")
    logger.info(f"F1: {aggregated.get('f1_mean', 0):.3f} ± {aggregated.get('f1_std', 0):.3f}")
    logger.info(f"G-mean: {aggregated.get('gmean_mean', 0):.3f} ± {aggregated.get('gmean_std', 0):.3f}")
    logger.info(f"Recall: {aggregated.get('recall_mean', 0):.3f} ± {aggregated.get('recall_std', 0):.3f}")
    logger.info(f"Precision: {aggregated.get('precision_mean', 0):.3f} ± {aggregated.get('precision_std', 0):.3f}")
    logger.info(f"Balanced Accuracy: {aggregated.get('balanced_accuracy_mean', 0):.3f} ± {aggregated.get('balanced_accuracy_std', 0):.3f}")
    
    # Save results
    save_results(aggregated, config.results_path, f"multirate_{run_name}")
    
    fold_df.to_csv(config.results_path / f"multirate_{run_name}_fold_metrics.csv", index=False)
    logger.info(f"Saved fold metrics to {config.results_path / f'multirate_{run_name}_fold_metrics.csv'}")
    
    save_predictions(y_true, y_pred, y_proba, subjects, config.results_path, f"multirate_{run_name}")
    
    plot_results(y_true, y_proba, f"multirate_{run_name}", config.results_path, y_pred)
    
    log_experiment_end(logger, "MULTI-RATE LATE FUSION TRAINING")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MULTI-RATE FUSION - LOSO Results")
    print(f"{'='*60}")
    print(f"Window: {config.window_size_sec}s | Horizon: {args.horizon}min")
    print(f"Channels: 8 (same as MOMENT, at native rates)")
    print(f"Threshold Method: constrained_gmean (min_recall=0.85, max_fpr=0.20)")
    print(f"AUROC: {aggregated.get('auroc_mean', 0):.3f} ± {aggregated.get('auroc_std', 0):.3f}")
    print(f"PR-AUC: {aggregated.get('pr_auc_mean', 0):.3f} ± {aggregated.get('pr_auc_std', 0):.3f}")
    print(f"F1: {aggregated.get('f1_mean', 0):.3f} ± {aggregated.get('f1_std', 0):.3f}")
    print(f"G-mean: {aggregated.get('gmean_mean', 0):.3f} ± {aggregated.get('gmean_std', 0):.3f}")
    print(f"Recall: {aggregated.get('recall_mean', 0):.3f} ± {aggregated.get('recall_std', 0):.3f}")
    print(f"Balanced Accuracy: {aggregated.get('balanced_accuracy_mean', 0):.3f} ± {aggregated.get('balanced_accuracy_std', 0):.3f}")
    print(f"Results saved to: {config.results_path}")


if __name__ == "__main__":
    main()


