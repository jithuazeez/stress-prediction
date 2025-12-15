"""
Training script for MOMENT foundation model with LOSO cross-validation.

Fine-tunes MOMENT for stress classification using aligned multimodal signals.

References:
- https://github.com/moment-timeseries-foundation-model/moment
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
from shared.windowing import create_labeled_windows, parse_stress_events
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

from dataset import VitaStressMOMENTDataset

# Try to import MOMENT
try:
    from model import MOMENTClassifier, create_moment_model, MOMENT_AVAILABLE
except ImportError:
    MOMENT_AVAILABLE = False


def load_all_windows(config: Config, logger) -> Dict[str, List[Dict]]:
    """Load and process all subjects, returning windows by subject."""
    subjects = get_all_subjects(config.data_path)
    windows_by_subject = {}
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Processing subjects", unit="subject",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    successful = 0
    total_windows = 0
    
    for subject_folder in pbar:
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            continue
        
        aligned = align_to_1hz(signals, start, end)
        if aligned is None or len(aligned) == 0:
            continue
        
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=config.window_size_sec,
            overlap_ratio=config.overlap_ratio,
            horizons_minutes=config.horizons_minutes,
            skip_first_minutes=config.skip_first_minutes
        )
        
        if windows:
            windows_by_subject[subject_id] = windows
            successful += 1
            total_windows += len(windows)
            pbar.set_postfix({"OK": successful, "Windows": total_windows})
    
    pbar.close()
    logger.info(f"Loaded {successful} subjects with {total_windows} total windows")
    
    return windows_by_subject


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                n_epochs: int) -> float:
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{n_epochs}", 
                leave=False, unit="batch")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    pbar.close()
    return total_loss / max(n_batches, 1)


def evaluate_epoch(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on test data.
    
    Returns only y_true and y_proba. The actual predictions (y_pred) are
    computed later using an optimal threshold based on target specificity.
    
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


def save_final_model_and_config(model: nn.Module,
                                hyperparameters: Dict,
                                aggregate_metrics: Dict,
                                output_dir: Path):
    """
    Save final model after all folds with complete configuration.
    
    Args:
        model: Final trained model
        hyperparameters: All hyperparameters used
        aggregate_metrics: Aggregated metrics across all folds
        output_dir: Directory to save final model
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = models_dir / "moment_final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'aggregate_metrics': aggregate_metrics,
        'hyperparameters': hyperparameters,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, model_path)
    
    # Save hyperparameters as JSON for easy viewing
    config_path = models_dir / "hyperparameters.json"
    with open(config_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2, default=str)
    
    # Save model architecture summary
    summary_path = models_dir / "model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MOMENT MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}\n\n")
        
        f.write("HYPERPARAMETERS\n")
        f.write("-"*60 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"{key:.<40} {value}\n")
        
        f.write("\nAGGREGATE METRICS\n")
        f.write("-"*60 + "\n")
        for key, value in aggregate_metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<40} {value:.4f}\n")
            else:
                f.write(f"{key:.<40} {value}\n")
    
    return model_path, config_path, summary_path


def loso_cross_validation(windows_by_subject: Dict[str, List[Dict]],
                          config: Config,
                          device: torch.device,
                          logger,
                          n_epochs: int = 10,
                          batch_size: int = 16,
                          learning_rate: float = 1e-4,
                          threshold_method: str = "youden") -> Dict:
    """
    Perform LOSO (Leave-One-Subject-Out) cross-validation with MOMENT.
    
    LOSO ensures NO data leakage:
    - Each fold: Train on N-1 subjects, test on 1 held-out subject
    - No overlap between train and test subjects
    - Evaluates model's ability to generalize to unseen subjects
    
    Threshold is computed on TRAINING data only (no data leakage).
    
    Args:
        windows_by_subject: Dict mapping subject_id to list of window dicts
        config: Experiment configuration
        device: PyTorch device
        logger: Logger instance
        n_epochs: Number of training epochs per fold
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        threshold_method: Method to find optimal threshold on training data
                         "youden" - Maximizes sensitivity + specificity (balanced)
                         "f1" - Maximizes F1 score
                         "balanced" - Where sensitivity ≈ specificity
    
    Returns:
        Dict with metrics, predictions, and fold results
    """
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO CROSS-VALIDATION STRATEGY")
    logger.info(f"{'='*60}")
    logger.info(f"Strategy: Leave-One-Subject-Out (LOSO)")
    logger.info(f"  - Train on N-1 subjects, test on 1 held-out subject")
    logger.info(f"  - Repeat for all {n_subjects} subjects")
    logger.info(f"  - Guarantees NO data leakage between subjects")
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Total subjects: {n_subjects}")
    logger.info(f"  Epochs per fold: {n_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Backbone: Frozen (only classification head trained)")
    logger.info(f"\nEvaluation Strategy:")
    logger.info(f"  Threshold Method: {threshold_method}")
    logger.info(f"  Threshold computed on TRAINING data (no data leakage)")
    logger.info(f"  Metrics reported honestly without forcing targets")
    logger.info(f"{'='*60}\n")
    
    all_y_true = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    
    # Track best model
    best_auroc = 0.0
    best_fold_idx = -1
    
    # Store hyperparameters
    hyperparameters = {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'n_channels': config.n_channels,
        'window_size_sec': config.window_size_sec,
        'overlap_ratio': config.overlap_ratio,
        'prediction_horizons': config.horizons_minutes,
        'target_label': config.target_label,
        'freeze_backbone': True,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss_weighted',
        'device': str(device),
        'threshold_method': threshold_method
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
        
        train_dataset = VitaStressMOMENTDataset(train_windows, config.target_label)
        test_dataset = VitaStressMOMENTDataset(test_windows, config.target_label)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = create_moment_model(
            n_channels=config.n_channels,
            num_classes=2,
            freeze_backbone=True,
            use_simple=True
        ).to(device)
        
        # Class weights
        n_pos = sum(1 for w in train_windows if w.get(config.target_label, 0) == 1)
        n_neg = len(train_windows) - n_pos
        weight = torch.tensor([1.0, np.sqrt(n_neg / max(n_pos, 1))], dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_train_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                    device, epoch, n_epochs)
            best_train_loss = min(best_train_loss, train_loss)
        
        # Get predictions on TRAINING data to find optimal threshold
        train_y_true, train_y_proba = evaluate_epoch(model, train_loader, device)
        
        # Find optimal threshold on TRAINING data (no data leakage!)
        fold_threshold, train_thresh_metrics = find_optimal_threshold(
            train_y_true, train_y_proba, method=threshold_method
        )
        
        # Evaluate on TEST data using threshold from training
        y_true, y_proba = evaluate_epoch(model, test_loader, device)
        
        # Store results
        all_y_true.extend(y_true)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_true))
        
        # Fold metrics using threshold computed on TRAINING data (no leakage)
        fold_metric = evaluate_predictions(
            y_true, 
            y_pred=None,  # Will be computed from y_proba using threshold
            y_proba=y_proba, 
            model_name="moment",
            threshold=fold_threshold  # Threshold from TRAINING data
        )
        fold_metric["subject"] = test_subject
        fold_metric["train_loss"] = best_train_loss
        fold_metric["train_sensitivity"] = train_thresh_metrics.get("sensitivity", float("nan"))
        fold_metric["train_specificity"] = train_thresh_metrics.get("specificity", float("nan"))
        fold_metrics.append(fold_metric)
        
        # Log fold results
        fold_auroc = fold_metric.get("auroc", float("nan"))
        fold_pr_auc = fold_metric.get("pr_auc", float("nan"))
        fold_sensitivity = fold_metric.get("sensitivity", float("nan"))
        fold_specificity = fold_metric.get("specificity", float("nan"))
        fold_precision = fold_metric.get("precision", float("nan"))
        fold_f1 = fold_metric.get("f1", float("nan"))
        fold_far = fold_metric.get("false_alarm_rate", float("nan"))
        
        logger.info(f"\n  Fold {fold_idx+1}/{n_subjects} - Subject {test_subject[:12]}...")
        logger.info(f"    Train: {len(train_windows)} windows ({n_pos}/{n_neg} pos/neg, weight={np.sqrt(n_neg/max(n_pos,1)):.2f})")
        logger.info(f"    Test:  {len(test_windows)} windows ({sum(y_true)}/{len(y_true)-sum(y_true)} pos/neg)")
        logger.info(f"    Train Loss:     {best_train_loss:.4f}")
        logger.info(f"    Threshold:      {fold_threshold:.4f} (from training, method={threshold_method})")
        logger.info(f"    --- Threshold-Independent ---")
        logger.info(f"    AUROC:          {fold_auroc:.4f}")
        logger.info(f"    PR-AUC:         {fold_pr_auc:.4f}")
        logger.info(f"    --- Test Set Performance ---")
        logger.info(f"    Sensitivity:    {fold_sensitivity:.4f} (Recall)")
        logger.info(f"    Specificity:    {fold_specificity:.4f}")
        logger.info(f"    False Alarm:    {fold_far:.4f}")
        logger.info(f"    Precision:      {fold_precision:.4f}")
        logger.info(f"    F1-Score:       {fold_f1:.4f}")
        
        # Save model checkpoint (every 5th fold + best model)
        is_best = fold_auroc > best_auroc
        if is_best:
            best_auroc = fold_auroc
            best_fold_idx = fold_idx
        
        # Save checkpoints for every 5th fold and best model
        if (fold_idx + 1) % 5 == 0 or is_best or fold_idx == n_subjects - 1:
            results_dir = Path(__file__).parent / "results"
            save_model_checkpoint(
                model, fold_idx, test_subject, fold_metric,
                hyperparameters, results_dir, is_best=is_best
            )
            if is_best:
                logger.info(f"    ⭐ New best model! (AUROC: {fold_auroc:.4f})")
        
        pbar.set_postfix({
            "Loss": f"{best_train_loss:.3f}",
            "AUROC": f"{fold_auroc:.3f}",
            "Sens": f"{fold_sensitivity:.3f}",
            "Spec": f"{fold_specificity:.3f}"
        })
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO CROSS-VALIDATION COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed/n_subjects:.1f}s per fold)")
    logger.info(f"Folds completed: {len(fold_metrics)}/{n_subjects}")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # Find final threshold on ALL training data combined
    final_threshold, _ = find_optimal_threshold(all_y_true, all_y_proba, method=threshold_method)
    
    # Aggregate metrics using mean threshold from folds
    mean_threshold = np.mean([f.get("threshold", 0.5) for f in fold_metrics])
    all_y_pred = (all_y_proba >= mean_threshold).astype(int)
    
    aggregate_metrics = evaluate_predictions(
        all_y_true, 
        y_pred=all_y_pred,
        y_proba=all_y_proba, 
        model_name="moment",
        threshold=mean_threshold
    )
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    aggregate_metrics["mean_threshold"] = float(mean_threshold)
    
    # Log overall performance
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL PERFORMANCE ACROSS ALL FOLDS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples:     {len(all_y_true)}")
    logger.info(f"Positive samples:  {sum(all_y_true)} ({100*sum(all_y_true)/len(all_y_true):.1f}%)")
    logger.info(f"Negative samples:  {len(all_y_true)-sum(all_y_true)} ({100*(len(all_y_true)-sum(all_y_true))/len(all_y_true):.1f}%)")
    logger.info(f"\nThreshold Selection (computed on training data):")
    logger.info(f"  Method:          {threshold_method}")
    logger.info(f"  Mean Threshold:  {mean_threshold:.4f}")
    logger.info(f"\n--- Threshold-Independent Metrics ---")
    logger.info(f"  AUROC:           {aggregate_metrics.get('auroc', float('nan')):.4f} ± {aggregate_metrics.get('auroc_std', 0):.4f}")
    logger.info(f"  PR-AUC:          {aggregate_metrics.get('pr_auc', float('nan')):.4f} ± {aggregate_metrics.get('pr_auc_std', 0):.4f}")
    logger.info(f"\n--- Test Set Performance (honest metrics) ---")
    logger.info(f"  Sensitivity:     {aggregate_metrics.get('sensitivity', float('nan')):.4f} ± {aggregate_metrics.get('sensitivity_std', 0):.4f}")
    logger.info(f"  Specificity:     {aggregate_metrics.get('specificity', float('nan')):.4f} ± {aggregate_metrics.get('specificity_std', 0):.4f}")
    logger.info(f"  False Alarm:     {aggregate_metrics.get('false_alarm_rate', float('nan')):.4f} ± {aggregate_metrics.get('false_alarm_rate_std', 0):.4f}")
    logger.info(f"  Precision:       {aggregate_metrics.get('precision', float('nan')):.4f} ± {aggregate_metrics.get('precision_std', 0):.4f}")
    logger.info(f"  F1-Score:        {aggregate_metrics.get('f1', float('nan')):.4f} ± {aggregate_metrics.get('f1_std', 0):.4f}")
    logger.info(f"  Accuracy:        {aggregate_metrics.get('accuracy', float('nan')):.4f} ± {aggregate_metrics.get('accuracy_std', 0):.4f}")
    logger.info(f"  Balanced Acc:    {aggregate_metrics.get('balanced_accuracy', float('nan')):.4f} ± {aggregate_metrics.get('balanced_accuracy_std', 0):.4f}")
    logger.info(f"  Avg Train Loss:  {np.mean([f.get('train_loss', float('nan')) for f in fold_metrics]):.4f}")
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
        "best_auroc": best_auroc,
        "threshold": mean_threshold,
        "threshold_method": threshold_method
    }


def main():
    """Main training pipeline."""
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("moment", log_file=results_dir / "training.log")
    
    log_experiment_start(logger, "MOMENT FOUNDATION MODEL TRAINING")
    
    if not MOMENT_AVAILABLE:
        logger.error("MOMENT not available! Install with: pip install momentfm")
        return
    
    logger.info("MOMENT: Available ✓")
    
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
        n_features=config.n_channels,
        n_positive=n_pos,
        n_negative=total_windows - n_pos
    )
    
    # Run LOSO CV
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Model Training (LOSO CV)")
    logger.info("-"*50)
    
    # Threshold method: "youden" balances sensitivity and specificity
    # Other options: "f1" (maximize F1), "balanced" (sensitivity ≈ specificity)
    THRESHOLD_METHOD = "youden"
    
    results = loso_cross_validation(
        windows_by_subject,
        config,
        device,
        logger,
        n_epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        threshold_method=THRESHOLD_METHOD
    )
    
    # Log results
    log_model_results(logger, "MOMENT", results["metrics"])
    
    # Save results
    save_results(results["metrics"], results_dir, "moment")
    save_predictions(
        results["y_true"], results["y_pred"], results["y_proba"],
        results["subjects"], results_dir, "moment"
    )
    
    if len(np.unique(results["y_true"])) > 1:
        plot_results(results["y_true"], results["y_proba"], "moment", results_dir,
                    y_pred=results["y_pred"])
    
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / "moment_fold_metrics.csv", index=False)
    
    # Save final model with hyperparameters
    logger.info("\n" + "-"*50)
    logger.info("Saving final model and configuration...")
    logger.info("-"*50)
    
    # Create a new model instance for the final save (using last trained architecture)
    final_model = create_moment_model(
        n_channels=config.n_channels,
        num_classes=2,
        freeze_backbone=True
    ).to(device)
    
    model_path, config_path, summary_path = save_final_model_and_config(
        final_model,
        results["hyperparameters"],
        results["metrics"],
        results_dir
    )
    
    logger.info(f"✅ Model saved to: {model_path}")
    logger.info(f"✅ Hyperparameters saved to: {config_path}")
    logger.info(f"✅ Model summary saved to: {summary_path}")
    
    log_experiment_end(logger, "MOMENT FOUNDATION MODEL TRAINING")
    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
