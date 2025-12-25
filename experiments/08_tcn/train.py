"""
Training script for TCN model with LOSO cross-validation.

Uses Temporal Convolutional Network for stress classification with features
extracted on-the-fly (similar to classical ML, excluding HR/HRV).

References:
- https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
- https://arxiv.org/pdf/1803.01271.pdf
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import time
import json
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_to_1hz
from shared.windowing import create_labeled_windows, parse_stress_events, compute_subject_stats
from shared.evaluation import (
    evaluate_predictions, aggregate_fold_metrics,
    save_results, save_predictions, plot_results,
    find_optimal_threshold
)
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import (
    setup_logger, log_experiment_start, log_experiment_end,
    log_data_summary, log_model_results
)

from dataset import VitaStressTCNDataset
from model import create_tcn_model


def load_all_windows(config: Config, logger) -> Dict[str, List[Dict]]:
    """Load and process all subjects, returning windows by subject."""
    subjects = get_all_subjects(config.data_path)
    windows_by_subject = {}
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Processing subjects", unit="subject",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    successful = 0
    total_windows = 0
    total_potential_windows = 0
    total_rejected_windows = 0
    subjects_failed = 0
    
    for subject_folder in pbar:
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            subjects_failed += 1
            continue
        
        aligned = align_to_1hz(signals, start, end)
        if aligned is None or len(aligned) == 0:
            subjects_failed += 1
            continue
        
        # Compute subject-level statistics for subject-wise normalization
        subject_stats = compute_subject_stats(aligned)
        
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        # Calculate potential windows before filtering
        duration_sec = (end - start).total_seconds()
        skip_sec = config.skip_first_minutes * 60
        usable_sec = duration_sec - skip_sec
        step_size = int(config.window_size_sec * (1 - config.overlap_ratio))
        if step_size < 1:
            step_size = 1
        potential_windows = max(0, int((usable_sec - config.window_size_sec) / step_size))
        
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=config.window_size_sec,
            overlap_ratio=config.overlap_ratio,
            horizons_minutes=config.horizons_minutes,
            skip_first_minutes=config.skip_first_minutes,
            subject_stats=subject_stats
        )
        
        if windows:
            windows_by_subject[subject_id] = windows
            successful += 1
            total_windows += len(windows)
            total_potential_windows += potential_windows
            total_rejected_windows += (potential_windows - len(windows))
            pbar.set_postfix({"OK": successful, "Windows": total_windows, "Rejected": total_rejected_windows})
        else:
            subjects_failed += 1
    
    pbar.close()
    
    # Detailed windowing summary
    logger.info(f"Loaded {successful} subjects with {total_windows} total windows")
    if total_potential_windows > 0:
        acceptance_rate = 100 * total_windows / total_potential_windows
        rejection_rate = 100 * total_rejected_windows / total_potential_windows
        logger.info(f"  Potential windows: {total_potential_windows}")
        logger.info(f"  Accepted windows:  {total_windows} ({acceptance_rate:.1f}%)")
        logger.info(f"  Rejected windows:  {total_rejected_windows} ({rejection_rate:.1f}%)")
        if total_rejected_windows > 0:
            logger.info(f"  Rejection reason:  Insufficient data (<50% samples in window)")
    if subjects_failed > 0:
        logger.info(f"  Subjects failed:   {subjects_failed} (no data/alignment issues)")
    
    return windows_by_subject


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                n_epochs: int,
                verbose: bool = False) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        n_epochs: Total epochs
        verbose: If True, show per-batch progress bar
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    loader = train_loader
    if verbose:
        loader = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{n_epochs}", 
                      leave=False, unit="batch")
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if verbose and hasattr(loader, 'set_postfix'):
            loader.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    if verbose and hasattr(loader, 'close'):
        loader.close()
    
    return total_loss / max(n_batches, 1)


def evaluate_epoch(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on test data.
    
    Returns only y_true and y_proba. The actual predictions (y_pred) are
    computed later using an optimal threshold.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Tuple of (y_true, y_proba) arrays
    """
    model.eval()
    
    all_y_true = []
    all_y_proba = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            proba = torch.softmax(logits, dim=-1)[:, 1]
            
            all_y_true.extend(y.cpu().numpy())
            all_y_proba.extend(proba.cpu().numpy())
    
    return np.array(all_y_true), np.array(all_y_proba)


def save_model_checkpoint(model: nn.Module,
                          fold_idx: int,
                          subject_id: str,
                          metrics: Dict,
                          hyperparameters: Dict,
                          output_dir: Path,
                          is_best: bool = False):
    """
    Save model checkpoint with weights and hyperparameters.
    
    Args:
        model: Trained model
        fold_idx: Current fold index
        subject_id: Test subject ID
        metrics: Performance metrics
        hyperparameters: Training hyperparameters
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'fold_idx': fold_idx,
        'subject_id': subject_id,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'hyperparameters': hyperparameters,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save fold-specific checkpoint
    fold_path = checkpoint_dir / f"fold_{fold_idx+1}_subject_{subject_id[:8]}.pt"
    torch.save(checkpoint, fold_path)
    
    # Save as best model if applicable
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        
    return fold_path


def loso_cross_validation(windows_by_subject: Dict[str, List[Dict]],
                          config: Config,
                          device: torch.device,
                          logger,
                          n_epochs: int = 50,
                          batch_size: int = 32,
                          learning_rate: float = 1e-3,
                          threshold_method: str = "geometric_mean",
                          tcn_channels: List[int] = None,
                          kernel_size: int = 3,
                          dilation_base: int = 2,
                          dropout: float = 0.3,
                          fc_hidden_dim: int = 128) -> Dict:
    """
    Perform LOSO (Leave-One-Subject-Out) cross-validation with TCN.
    
    Args:
        windows_by_subject: Dict mapping subject_id to list of window dicts
        config: Experiment configuration
        device: PyTorch device
        logger: Logger instance
        n_epochs: Number of training epochs per fold
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        threshold_method: Method to find optimal threshold
        tcn_channels: List of channel sizes for TCN blocks
        kernel_size: Kernel size for convolutions
        dilation_base: Base for exponential dilation
        dropout: Dropout probability
        fc_hidden_dim: Hidden dimension for FC layers
    
    Returns:
        Dict with metrics, predictions, and fold results
    """
    if tcn_channels is None:
        tcn_channels = [64, 64, 64]
    
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO CV: {n_subjects} subjects | {n_epochs} epochs | batch={batch_size} | lr={learning_rate}")
    logger.info(f"TCN: channels={tcn_channels} | kernel={kernel_size} | dilation={dilation_base}")
    logger.info(f"Threshold: {threshold_method} (computed on training data)")
    logger.info(f"{'='*60}")
    
    all_y_true = []
    all_y_proba = []
    all_y_pred = []
    all_subjects = []
    fold_metrics = []
    
    best_gmean = 0.0
    best_fold_idx = -1
    
    # Store hyperparameters
    hyperparameters = {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'tcn_channels': tcn_channels,
        'kernel_size': kernel_size,
        'dilation_base': dilation_base,
        'dropout': dropout,
        'fc_hidden_dim': fc_hidden_dim,
        'window_size_sec': config.window_size_sec,
        'overlap_ratio': config.overlap_ratio,
        'prediction_horizons': config.horizons_minutes,
        'target_label': config.target_label,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss_weighted',
        'device': str(device),
        'threshold_method': threshold_method,
    }
    
    start_time = time.time()
    
    pbar = tqdm(enumerate(subjects), total=n_subjects, desc="LOSO folds", unit="fold")
    
    for fold_idx, test_subject in pbar:
        pbar.set_description(f"Fold {fold_idx+1}/{n_subjects} ({test_subject[:8]}...)")
        
        # Create datasets
        train_windows = []
        test_windows = []
        
        for subject_id, windows in windows_by_subject.items():
            for w in windows:
                w["subject_id"] = subject_id
            
            if subject_id == test_subject:
                test_windows.extend(windows)
            else:
                train_windows.extend(windows)
        
        if len(train_windows) == 0 or len(test_windows) == 0:
            continue
        
        train_dataset = VitaStressTCNDataset(train_windows, config.target_label, normalize=True)
        test_dataset = VitaStressTCNDataset(test_windows, config.target_label, normalize=True)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        num_features = train_dataset.get_n_features()
        model = create_tcn_model(
            num_inputs=num_features,
            num_classes=2,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            fc_hidden_dim=fc_hidden_dim
        ).to(device)
        
        # Log model info on first fold
        if fold_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            receptive_field = model.get_receptive_field(kernel_size, dilation_base)
            logger.info(f"Model: {total_params:,} params | Trainable: {trainable_params:,} | RF: {receptive_field}")
            logger.info(f"Features: {num_features}")
        
        # Class weights
        n_pos = sum(1 for w in train_windows if w.get(config.target_label, 0) == 1)
        n_neg = len(train_windows) - n_pos
        weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop with early stopping
        best_train_loss = float('inf')
        patience = 15
        patience_counter = 0
        epoch_log_interval = 15
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                    device, epoch, n_epochs)
            
            # Early stopping check
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress every N epochs
            if (epoch + 1) % epoch_log_interval == 0 or epoch == 0 or epoch == n_epochs - 1:
                logger.info(f"  Fold {fold_idx+1} Epoch {epoch+1:3d}/{n_epochs} | Loss: {train_loss:.4f} | Best: {best_train_loss:.4f} | Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"  ⚠️  Early stopping at epoch {epoch+1}/{n_epochs} (no improvement for {patience} epochs)")
                logger.info(f"     Best loss: {best_train_loss:.4f}")
                break
        
        # Get predictions on TRAINING data to find optimal threshold
        train_y_true, train_y_proba = evaluate_epoch(model, train_loader, device)
        
        # Find optimal threshold on TRAINING data
        fold_threshold, train_thresh_metrics = find_optimal_threshold(
            train_y_true, train_y_proba, 
            method=threshold_method
        )
        
        # Evaluate on TEST data using threshold from training
        y_true, y_proba = evaluate_epoch(model, test_loader, device)
        
        # Apply fold-specific threshold to get predictions
        y_pred = (y_proba >= fold_threshold).astype(int)
        
        # Store results
        all_y_true.extend(y_true)
        all_y_proba.extend(y_proba)
        all_y_pred.extend(y_pred)
        all_subjects.extend([test_subject] * len(y_true))
        
        # Fold metrics
        fold_metric = evaluate_predictions(
            y_true, 
            y_pred=y_pred,
            y_proba=y_proba, 
            model_name="tcn",
            threshold=fold_threshold
        )
        fold_metric["subject"] = test_subject
        fold_metric["train_loss"] = best_train_loss
        fold_metric["epochs_trained"] = epoch + 1
        fold_metric["early_stopped"] = (patience_counter >= patience)
        fold_metric["train_sensitivity"] = train_thresh_metrics.get("sensitivity", float("nan"))
        fold_metric["train_specificity"] = train_thresh_metrics.get("specificity", float("nan"))
        fold_metric["train_gmean"] = train_thresh_metrics.get("gmean", float("nan"))
        fold_metrics.append(fold_metric)
        
        # Log fold results
        fold_auroc = fold_metric.get("auroc", float("nan"))
        fold_pr_auc = fold_metric.get("pr_auc", float("nan"))
        fold_sensitivity = fold_metric.get("sensitivity", float("nan"))
        fold_specificity = fold_metric.get("specificity", float("nan"))
        fold_precision = fold_metric.get("precision", float("nan"))
        fold_recall = fold_metric.get("recall", float("nan"))
        fold_f1 = fold_metric.get("f1", float("nan"))
        fold_gmean = fold_metric.get("gmean", float("nan"))
        fold_balanced_acc = fold_metric.get("balanced_accuracy", float("nan"))
        fold_threshold_val = fold_metric.get("threshold", float("nan"))
        
        logger.info(
            f"Fold {fold_idx+1:2d}/{n_subjects} | "
            f"Subj: {test_subject[:8]} | "
            f"Thr: {fold_threshold_val:.3f} | "
            f"Recall: {fold_recall:.3f} | "
            f"Precision: {fold_precision:.3f} | "
            f"Gmean: {fold_gmean:.3f} | "
            f"BalAcc: {fold_balanced_acc:.3f} | "
            f"F1: {fold_f1:.3f} | "
            f"PR-AUC: {fold_pr_auc:.3f}"
        )
        
        # Save model checkpoint
        is_best = fold_gmean > best_gmean
        if is_best:
            best_gmean = fold_gmean
            best_fold_idx = fold_idx
        
        if (fold_idx + 1) % 5 == 0 or is_best or fold_idx == n_subjects - 1:
            results_dir = Path(__file__).parent / "results"
            save_model_checkpoint(
                model, fold_idx, test_subject, fold_metric,
                hyperparameters, results_dir, is_best=is_best
            )
            if is_best:
                logger.info(f"  -> New best! Gmean={fold_gmean:.4f} (Sens={fold_sensitivity:.3f}, Spec={fold_specificity:.3f})")
        
        pbar.set_postfix({
            "Loss": f"{best_train_loss:.3f}",
            "Gmean": f"{fold_gmean:.3f}",
            "F1": f"{fold_f1:.3f}",
            "AUROC": f"{fold_auroc:.3f}",
            "PR-AUC": f"{fold_pr_auc:.3f}",
        })
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO CROSS-VALIDATION COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed/n_subjects:.1f}s per fold)")
    logger.info(f"Folds completed: {len(fold_metrics)}/{n_subjects}")
    
    # Report early stopping statistics
    early_stopped_folds = sum(1 for f in fold_metrics if f.get("early_stopped", False))
    if early_stopped_folds > 0:
        avg_epochs_trained = np.mean([f.get("epochs_trained", n_epochs) for f in fold_metrics])
        logger.info(f"\nEarly Stopping Summary:")
        logger.info(f"  Folds stopped early: {early_stopped_folds}/{len(fold_metrics)} ({100*early_stopped_folds/len(fold_metrics):.1f}%)")
        logger.info(f"  Average epochs trained: {avg_epochs_trained:.1f}/{n_epochs}")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # Aggregate metrics
    aggregate_metrics = evaluate_predictions(
        all_y_true, 
        y_pred=all_y_pred,
        y_proba=all_y_proba, 
        model_name="tcn"
    )
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    # Report threshold statistics
    fold_thresholds = [f.get("threshold", 0.5) for f in fold_metrics]
    aggregate_metrics["threshold_mean"] = float(np.mean(fold_thresholds))
    aggregate_metrics["threshold_std"] = float(np.std(fold_thresholds))
    aggregate_metrics["threshold_min"] = float(np.min(fold_thresholds))
    aggregate_metrics["threshold_max"] = float(np.max(fold_thresholds))
    
    # Log overall performance
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL PERFORMANCE ACROSS ALL FOLDS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples:     {len(all_y_true)}")
    logger.info(f"Positive samples:  {sum(all_y_true)} ({100*sum(all_y_true)/len(all_y_true):.1f}%)")
    logger.info(f"Negative samples:  {len(all_y_true)-sum(all_y_true)} ({100*(len(all_y_true)-sum(all_y_true))/len(all_y_true):.1f}%)")
    logger.info(f"\nThreshold Selection (computed on training data per fold):")
    logger.info(f"  Method:          {threshold_method}")
    logger.info(f"  Threshold Range: {aggregate_metrics['threshold_min']:.4f} - {aggregate_metrics['threshold_max']:.4f}")
    logger.info(f"  Threshold Mean:  {aggregate_metrics['threshold_mean']:.4f} ± {aggregate_metrics['threshold_std']:.4f}")
    logger.info(f"\n--- Threshold-Independent Metrics ---")
    logger.info(f"  AUROC:           {aggregate_metrics.get('auroc', float('nan')):.4f} ± {aggregate_metrics.get('auroc_std', 0):.4f}")
    logger.info(f"  PR-AUC:          {aggregate_metrics.get('pr_auc', float('nan')):.4f} ± {aggregate_metrics.get('pr_auc_std', 0):.4f}")
    logger.info(f"\n--- Test Set Performance ---")
    logger.info(f"  Gmean:           {aggregate_metrics.get('gmean', float('nan')):.4f} ± {aggregate_metrics.get('gmean_std', 0):.4f}")
    logger.info(f"  F1-Score:        {aggregate_metrics.get('f1', float('nan')):.4f} ± {aggregate_metrics.get('f1_std', 0):.4f}")
    logger.info(f"  Sensitivity:     {aggregate_metrics.get('sensitivity', float('nan')):.4f} ± {aggregate_metrics.get('sensitivity_std', 0):.4f}")
    logger.info(f"  Specificity:     {aggregate_metrics.get('specificity', float('nan')):.4f} ± {aggregate_metrics.get('specificity_std', 0):.4f}")
    logger.info(f"  Precision:       {aggregate_metrics.get('precision', float('nan')):.4f} ± {aggregate_metrics.get('precision_std', 0):.4f}")
    logger.info(f"  Recall:          {aggregate_metrics.get('recall', float('nan')):.4f} ± {aggregate_metrics.get('recall_std', 0):.4f}")
    logger.info(f"  Accuracy:        {aggregate_metrics.get('accuracy', float('nan')):.4f} ± {aggregate_metrics.get('accuracy_std', 0):.4f}")
    logger.info(f"  Balanced Acc:    {aggregate_metrics.get('balanced_accuracy', float('nan')):.4f} ± {aggregate_metrics.get('balanced_accuracy_std', 0):.4f}")
    logger.info(f"  False Alarm:     {aggregate_metrics.get('false_alarm_rate', float('nan')):.4f} ± {aggregate_metrics.get('false_alarm_rate_std', 0):.4f}")
    logger.info(f"  Avg Train Loss:  {np.mean([f.get('train_loss', float('nan')) for f in fold_metrics]):.4f}")
    logger.info(f"\nBest Model: Fold {best_fold_idx+1} with Gmean={best_gmean:.4f}")
    logger.info(f"{'='*60}\n")
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": all_subjects,
        "fold_metrics": fold_metrics,
        "hyperparameters": hyperparameters,
        "best_fold_idx": best_fold_idx,
        "best_gmean": best_gmean,
        "threshold_mean": aggregate_metrics.get("threshold_mean"),
        "threshold_std": aggregate_metrics.get("threshold_std"),
        "threshold_min": aggregate_metrics.get("threshold_min"),
        "threshold_max": aggregate_metrics.get("threshold_max"),
        "threshold_method": threshold_method
    }


def main():
    """Main training pipeline."""
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("tcn", log_file=results_dir / "training.log")
    
    log_experiment_start(logger, "TCN MODEL TRAINING")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load all windows
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Loading and preprocessing data")
    logger.info("-"*50)
    
    windows_by_subject = load_all_windows(config, logger)
    
    if not windows_by_subject:
        logger.error("No data loaded!")
        return
    
    total_windows = sum(len(w) for w in windows_by_subject.values())
    n_pos = sum(sum(1 for w in windows if w.get(config.target_label, 0) == 1) 
                for windows in windows_by_subject.values())
    
    log_data_summary(
        logger,
        n_subjects=len(windows_by_subject),
        n_windows=total_windows,
        n_features=0,  # Will be computed dynamically
        n_positive=n_pos,
        n_negative=total_windows - n_pos
    )
    
    # Run LOSO CV
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Model Training (LOSO CV)")
    logger.info("-"*50)
    
    results = loso_cross_validation(
        windows_by_subject,
        config,
        device,
        logger,
        n_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        threshold_method="geometric_mean",
        tcn_channels=[64, 64, 64],
        kernel_size=3,
        dilation_base=2,
        dropout=0.3,
        fc_hidden_dim=128
    )
    
    # Log results
    log_model_results(logger, "TCN", results["metrics"])
    
    # Save results
    save_results(results["metrics"], results_dir, "tcn")
    save_predictions(
        results["y_true"], results["y_pred"], results["y_proba"],
        results["subjects"], results_dir, "tcn"
    )
    
    if len(np.unique(results["y_true"])) > 1:
        plot_results(results["y_true"], results["y_proba"], "tcn", results_dir,
                    y_pred=results["y_pred"])
    
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / "tcn_fold_metrics.csv", index=False)
    
    # Save configuration summary
    config_dir = results_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(results["hyperparameters"], f, indent=2, default=str)
    
    logger.info(f"\n✅ Configuration saved to: {config_path}")
    
    log_experiment_end(logger, "TCN MODEL TRAINING")
    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()

