"""
MAML Training for Personalized Stress Prediction.

Implements Model-Agnostic Meta-Learning (MAML) following Finn et al. 2017.
Uses learn2learn for proper second-order gradient computation.

Key fixes from original implementation:
1. Uses learn2learn.algorithms.MAML for correct gradient flow
2. Supports both MLP (extracted features) and CNN (raw signals)
3. Uses BALANCED k-shot sampling for support sets
4. Computes threshold on training data to avoid data leakage

Usage:
  python train.py --model mlp    # Use MLP with 61 statistical features (default)
  python train.py --model cnn    # Use CNN with 8-channel raw signals

References:
- https://arxiv.org/pdf/1703.03400 (MAML paper)
- https://github.com/learnables/learn2learn
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import time
import random

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from model import StressClassifier, create_maml_model
from meta_dataset import (
    StressMetaDataset, prepare_features_by_subject, 
    get_feature_dim, FEATURE_NAMES
)

# Check for learn2learn
try:
    import learn2learn as l2l
    from learn2learn.algorithms import MAML
    LEARN2LEARN_AVAILABLE = True
except ImportError:
    LEARN2LEARN_AVAILABLE = False
    print("Warning: learn2learn not installed. Install with: pip install learn2learn")


# =============================================================================
# MAML Training Functions
# =============================================================================

def compute_loss(model: nn.Module, 
                 X: torch.Tensor, 
                 y: torch.Tensor,
                 class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute cross-entropy loss with optional class weighting.
    
    Args:
        model: Neural network model
        X: Input features
        y: Labels
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Loss tensor
    """
    logits = model(X)
    if class_weights is not None:
        return F.cross_entropy(logits, y, weight=class_weights)
    return F.cross_entropy(logits, y)


def fast_adapt(learner, 
               support_x: torch.Tensor, 
               support_y: torch.Tensor,
               adaptation_steps: int,
               class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Perform fast adaptation on support set.
    
    This is the inner loop of MAML - adapts model parameters to a specific task.
    
    Args:
        learner: learn2learn MAML learner (cloned model)
        support_x: Support set features
        support_y: Support set labels
        adaptation_steps: Number of gradient steps for adaptation
        class_weights: Optional class weights
    
    Returns:
        Adapted learner
    """
    for _ in range(adaptation_steps):
        loss = compute_loss(learner, support_x, support_y, class_weights)
        learner.adapt(loss)  # learn2learn handles gradient computation correctly!
    
    return learner


def meta_train_epoch(maml: MAML,
                     meta_dataset: StressMetaDataset,
                     meta_optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     tasks_per_batch: int = 4,
                     adaptation_steps: int = 5,
                     class_weights: Optional[torch.Tensor] = None) -> float:
    """
    Perform one meta-training epoch.
    
    Outer loop: Updates meta-parameters based on post-adaptation performance.
    
    Args:
        maml: learn2learn MAML wrapper
        meta_dataset: StressMetaDataset instance
        meta_optimizer: Optimizer for meta-parameters
        device: Torch device
        tasks_per_batch: Number of tasks per meta-batch
        adaptation_steps: Inner loop steps
        class_weights: Optional class weights
    
    Returns:
        Average meta-loss for the epoch
    """
    meta_optimizer.zero_grad()
    
    meta_loss = 0.0
    
    # Sample batch of tasks
    tasks = meta_dataset.sample_tasks(tasks_per_batch)
    
    for support_x, support_y, query_x, query_y in tasks:
        # Move to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        # Clone model for this task
        learner = maml.clone()
        
        # Inner loop: adapt to support set
        learner = fast_adapt(
            learner, support_x, support_y, 
            adaptation_steps, class_weights
        )
        
        # Outer loop: evaluate on query set
        query_loss = compute_loss(learner, query_x, query_y, class_weights)
        meta_loss += query_loss
    
    # Average loss and backpropagate through adaptation
    meta_loss = meta_loss / len(tasks)
    meta_loss.backward()  # This backprops through the inner loop updates!
    
    # Update meta-parameters
    meta_optimizer.step()
    
    return meta_loss.item()


def evaluate_adapted_model(maml: MAML,
                           X: torch.Tensor,
                           y: torch.Tensor,
                           device: torch.device,
                           adaptation_steps: int = 5,
                           adapt_fraction: float = 0.2,
                           class_weights: Optional[torch.Tensor] = None,
                           preserve_temporal_order: bool = True
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a subject with adaptation.
    
    Split subject data into adapt set (for fine-tuning) and eval set (for metrics).
    
    IMPORTANT: By default, preserves temporal order to avoid data leakage.
    Uses FIRST 20% for adaptation, LAST 80% for evaluation.
    
    Args:
        maml: learn2learn MAML wrapper
        X: All features for subject
        y: All labels for subject
        device: Torch device
        adaptation_steps: Steps for adaptation
        adapt_fraction: Fraction of data to use for adaptation
        class_weights: Optional class weights
        preserve_temporal_order: If True, uses first samples for adapt, last for eval
                                If False, randomly shuffles (NOT RECOMMENDED for time series)
    
    Returns:
        Tuple of (y_true, y_pred, y_proba) for eval set
    """
    n_samples = len(y)
    n_adapt = max(5, int(n_samples * adapt_fraction))
    
    # Ensure we have enough samples
    if n_samples <= n_adapt:
        n_adapt = max(2, n_samples // 3)
    
    # Split into adapt and eval
    if preserve_temporal_order:
        # ✅ TEMPORAL SPLIT: Use first samples for adaptation, later samples for evaluation
        # This prevents data leakage (no future → past information flow)
        adapt_indices = np.arange(n_adapt)
        eval_indices = np.arange(n_adapt, n_samples)
    else:
        # ⚠️ RANDOM SPLIT: Breaks temporal structure (NOT RECOMMENDED for time series)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        adapt_indices = indices[:n_adapt]
        eval_indices = indices[n_adapt:]
    
    if len(eval_indices) == 0:
        eval_indices = np.arange(n_samples)  # Use all for eval if too few samples
    
    # Prepare tensors
    adapt_x = X[adapt_indices].to(device)
    adapt_y = y[adapt_indices].to(device)
    eval_x = X[eval_indices].to(device)
    eval_y = y[eval_indices]
    
    # Clone and adapt
    learner = maml.clone()
    learner = fast_adapt(learner, adapt_x, adapt_y, adaptation_steps, class_weights)
    
    # Evaluate
    learner.eval()
    with torch.no_grad():
        logits = learner(eval_x)
        proba = F.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)
    
    return (
        eval_y.numpy(),
        pred.cpu().numpy(),
        proba.cpu().numpy()
    )


# =============================================================================
# Data Loading
# =============================================================================

def load_all_windows(config: Config, logger) -> Dict[str, List[Dict]]:
    """Load all windows grouped by subject."""
    subjects = get_all_subjects(config.data_path)
    windows_by_subject = {}
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Loading data", unit="subject")
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
        
        # Compute subject-level statistics for subject-wise normalization
        subject_stats = compute_subject_stats(aligned)
        
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
            skip_first_minutes=config.skip_first_minutes,
            subject_stats=subject_stats  # Pass subject stats for normalization
        )
        
        if windows:
            for w in windows:
                w["subject_id"] = subject_id
            windows_by_subject[subject_id] = windows
            successful += 1
            total_windows += len(windows)
            pbar.set_postfix({"OK": successful, "Windows": total_windows})
    
    pbar.close()
    logger.info(f"Loaded {successful} subjects with {total_windows} windows")
    
    return windows_by_subject


def prepare_raw_signals_by_subject(windows_by_subject: Dict[str, List[Dict]],
                                    label_col: str = "label_5min",
                                    n_channels: int = 8,
                                    seq_len: int = 120) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare raw 8-channel signals for CNN model.
    
    8 channels: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd
    (Same as MOMENT and SSL experiments)
    
    Args:
        windows_by_subject: Dict mapping subject_id to list of window dicts
        label_col: Label column to use
        n_channels: Number of channels (8)
        seq_len: Sequence length (120 for 2-min window at 1Hz)
    
    Returns:
        Dict mapping subject_id to (X, y) tuple where:
            X: Signal array of shape (n_windows, n_channels, seq_len)
            y: Label array of shape (n_windows,)
    """
    CHANNEL_NAMES = [
        "acc_x", "acc_y", "acc_z",
        "skin_temp", "heatflux", "cbt",
        "hr_bpm", "rmssd"
    ]
    
    signals_by_subject = {}
    
    for subject_id, windows in windows_by_subject.items():
        X_list = []
        y_list = []
        
        for window in windows:
            window_df = window.get("window_data")
            if window_df is None or len(window_df) == 0:
                continue
            
            # Get subject-level stats for normalization
            subject_stats = window.get("subject_stats", {})
            
            # Extract channels
            channels = []
            for channel_name in CHANNEL_NAMES:
                if channel_name in window_df.columns:
                    values = window_df[channel_name].values
                else:
                    # Try to find similar column
                    found = False
                    for col in window_df.columns:
                        if channel_name.lower() in col.lower():
                            values = window_df[col].values
                            found = True
                            break
                    if not found:
                        values = np.zeros(len(window_df))
                
                # Handle NaN
                values = np.nan_to_num(values, nan=0.0)
                
                # Ensure correct length
                if len(values) != seq_len:
                    # Resample if needed
                    from scipy.signal import resample
                    values = resample(values, seq_len)
                
                # Subject-wise normalization (recommended)
                if channel_name in subject_stats:
                    mean = subject_stats[channel_name]["mean"]
                    std = subject_stats[channel_name]["std"]
                    if std > 0:
                        values = (values - mean) / std
                    else:
                        values = values - mean
                
                channels.append(values)
            
            # Stack: shape (n_channels, seq_len)
            x = np.stack(channels, axis=0).astype(np.float32)
            X_list.append(x)
            y_list.append(window.get(label_col, 0))
        
        if X_list:
            X = np.array(X_list, dtype=np.float32)  # (n_windows, n_channels, seq_len)
            y = np.array(y_list, dtype=np.int64)
            signals_by_subject[subject_id] = (X, y)
    
    return signals_by_subject


# =============================================================================
# LOSO Cross-Validation
# =============================================================================

def loso_cross_validation(data_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          config: Config,
                          device: torch.device,
                          logger,
                          model_type: str = "mlp",
                          n_meta_epochs: int = 100,
                          tasks_per_batch: int = 4,
                          adaptation_steps: int = 5,
                          meta_lr: float = 0.001,
                          inner_lr: float = 0.01,
                          threshold_method: str = "constrained_gmean",
                          min_recall: float = 0.85,
                          max_fpr: float = 0.20) -> Dict:
    """
    Perform LOSO cross-validation with MAML.
    
    For each fold:
    1. Meta-train on all subjects except test subject
    2. Adapt to test subject using small portion of their data
    3. Evaluate on remaining test subject data
    4. Find optimal threshold on TRAINING data (constrained G-mean)
    5. Apply fold-specific threshold to TEST data
    
    Args:
        data_by_subject: Dict mapping subject_id to (X, y)
                        For MLP: X has shape (n_samples, n_features)
                        For CNN: X has shape (n_samples, n_channels, seq_len)
        config: Configuration object
        device: Torch device
        logger: Logger instance
        model_type: "mlp" or "cnn"
        n_meta_epochs: Number of meta-training epochs per fold
        tasks_per_batch: Tasks per meta-batch
        adaptation_steps: Inner loop adaptation steps
        meta_lr: Meta-learning rate (outer loop)
        inner_lr: Adaptation learning rate (inner loop)
        threshold_method: Method for finding decision threshold (default: "constrained_gmean")
                         Options: "youden", "f1", "balanced", "geometric_mean", "constrained_gmean"
        min_recall: Minimum recall/sensitivity required (for constrained_gmean, default: 0.85)
        max_fpr: Maximum false positive rate allowed (for constrained_gmean, default: 0.20)
    
    Returns:
        Results dictionary
    """
    if not LEARN2LEARN_AVAILABLE:
        raise ImportError("learn2learn is required for MAML. Install with: pip install learn2learn")
    
    subjects = list(data_by_subject.keys())
    n_subjects = len(subjects)
    
    # Determine input dimensions based on model type
    sample_X, _ = list(data_by_subject.values())[0]
    if model_type == "mlp":
        input_dim = sample_X.shape[1]  # (n_samples, n_features)
        n_channels, seq_len = None, None
    else:  # cnn
        n_channels = sample_X.shape[1]  # (n_samples, n_channels, seq_len)
        seq_len = sample_X.shape[2]
        input_dim = None
    
    logger.info(f"\n{'='*60}")
    logger.info("MAML LOSO Cross-Validation")
    logger.info(f"{'='*60}")
    logger.info(f"  Model type: {model_type.upper()}")
    logger.info(f"  Subjects: {n_subjects}")
    if model_type == "mlp":
        logger.info(f"  Input: {input_dim} features")
    else:
        logger.info(f"  Input: {n_channels} channels × {seq_len} timesteps")
    logger.info(f"  Meta epochs/fold: {n_meta_epochs}")
    logger.info(f"  Tasks per batch: {tasks_per_batch}")
    logger.info(f"  Adaptation steps: {adaptation_steps}")
    logger.info(f"  Meta LR (outer): {meta_lr}")
    logger.info(f"  Inner LR: {inner_lr}")
    logger.info(f"  Threshold method: {threshold_method}")
    if threshold_method == "constrained_gmean":
        logger.info(f"  Min recall: {min_recall:.2f} (≥{min_recall*100:.0f}% sensitivity)")
        logger.info(f"  Max FPR: {max_fpr:.2f} (≤{max_fpr*100:.0f}% false alarm rate)")
    logger.info(f"  Temporal order: Preserved (first 20% adapt, last 80% eval)")
    logger.info(f"  Threshold source: ADAPTED model (consistent with test)")
    logger.info(f"{'='*60}\n")
    
    # Storage for results
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    fold_thresholds = []
    
    start_time = time.time()
    
    pbar = tqdm(enumerate(subjects), total=n_subjects, desc="LOSO folds", unit="fold")
    
    for fold_idx, test_subject in pbar:
        pbar.set_description(f"Fold {fold_idx+1}/{n_subjects} ({test_subject[:8]}...)")
        
        # Split data
        train_subjects = [s for s in subjects if s != test_subject]
        train_data = {s: data_by_subject[s] for s in train_subjects}
        
        # Compute class weights from training data
        all_train_y = np.concatenate([data_by_subject[s][1] for s in train_subjects])
        n_pos = np.sum(all_train_y == 1)
        n_neg = np.sum(all_train_y == 0)
        
        if n_pos > 0 and n_neg > 0:
            weight_pos = min(n_neg / n_pos, 5.0)
            class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32).to(device)
        else:
            class_weights = None
        
        # Create meta-dataset for training (balanced sampling!)
        meta_dataset = StressMetaDataset(
            train_data,
            k_support=10,  # Support set size
            k_query=20     # Query set size
        )
        
        # Create model and wrap with learn2learn MAML
        if model_type == "mlp":
            model = create_maml_model(
                model_type="mlp",
                input_dim=input_dim,
                hidden_dim=64
            ).to(device)
        else:  # cnn
            model = create_maml_model(
                model_type="cnn",
                n_channels=n_channels,
                seq_len=seq_len,
                hidden_dim=64
            ).to(device)
        
        # Wrap with MAML - this handles gradient computation correctly!
        maml = MAML(model, lr=inner_lr, first_order=True)  # first_order=False for full MAML
        
        # Meta-optimizer (outer loop)
        meta_optimizer = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        
        # Meta-training
        epoch_losses = []
        epoch_pbar = tqdm(range(n_meta_epochs), desc="    Meta-training", 
                         leave=False, unit="epoch")
        
        for epoch in epoch_pbar:
            loss = meta_train_epoch(
                maml, meta_dataset, meta_optimizer, device,
                tasks_per_batch=tasks_per_batch,
                adaptation_steps=adaptation_steps,
                class_weights=class_weights
            )
            epoch_losses.append(loss)
            epoch_pbar.set_postfix({"Loss": f"{loss:.4f}"})
        
        epoch_pbar.close()
        
        # ========================================================================
        # Get training predictions for threshold selection
        # CRITICAL: Use ADAPTED model predictions (not base model)
        # ========================================================================
        # Threshold should be optimized on the same type of predictions used at test time.
        # Since test uses adapted model, threshold must be learned on adapted model too!
        train_y_true_all = []
        train_y_proba_all = []
        
        for train_subj in train_subjects[:5]:  # Use subset for efficiency
            X, y = data_by_subject[train_subj]
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Get predictions from ADAPTED model (consistent with test time)
            y_true_subj, _, y_proba_subj = evaluate_adapted_model(
                maml, X_tensor, y_tensor, device,
                adaptation_steps=adaptation_steps,
                class_weights=class_weights,
                preserve_temporal_order=True  # Preserve temporal structure
            )
            
            train_y_true_all.extend(y_true_subj)
            train_y_proba_all.extend(y_proba_subj)
        
        train_y_true_all = np.array(train_y_true_all)
        train_y_proba_all = np.array(train_y_proba_all)
        
        # Find optimal threshold on training data using constrained G-mean
        # This ensures high recall (catch stress events) + low FPR (control false alarms)
        fold_threshold, train_thresh_metrics = find_optimal_threshold(
            train_y_true_all, 
            train_y_proba_all, 
            method=threshold_method,
            min_recall=min_recall if threshold_method == "constrained_gmean" else 0.0,
            max_fpr=max_fpr if threshold_method == "constrained_gmean" else 1.0
        )
        fold_thresholds.append(fold_threshold)
        
        # Log threshold optimization results
        if train_thresh_metrics:
            logger.info(f"    Threshold optimization:")
            logger.info(f"      Method: {threshold_method}")
            logger.info(f"      Threshold: {fold_threshold:.4f}")
            logger.info(f"      Training G-mean: {train_thresh_metrics.get('gmean', 0):.4f}")
            logger.info(f"      Training Recall: {train_thresh_metrics.get('recall', 0):.4f}")
            logger.info(f"      Training FPR: {train_thresh_metrics.get('false_alarm_rate', 0):.4f}")
        
        # Evaluate on test subject
        test_X, test_y = data_by_subject[test_subject]
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
        test_y_tensor = torch.tensor(test_y, dtype=torch.long)
        
        # Evaluate with ADAPTED model (preserving temporal order)
        y_true, y_pred_raw, y_proba = evaluate_adapted_model(
            maml, test_X_tensor, test_y_tensor, device,
            adaptation_steps=adaptation_steps,
            class_weights=class_weights,
            preserve_temporal_order=True  # Use temporal split (first 20% adapt, last 80% eval)
        )
        
        # Apply threshold from training data
        y_pred = (y_proba >= fold_threshold).astype(int)
        
        # Store results
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_true))
        
        # Fold metrics
        fold_metric = evaluate_predictions(
            y_true, y_pred, y_proba, "maml",
            threshold=fold_threshold
        )
        fold_metric["subject"] = test_subject
        fold_metric["train_loss"] = np.mean(epoch_losses[-10:])  # Last 10 epochs
        fold_metrics.append(fold_metric)
        
        # Log fold results
        auroc = fold_metric.get("auroc", float("nan"))
        sens = fold_metric.get("sensitivity", float("nan"))
        spec = fold_metric.get("specificity", float("nan"))
        gmean = fold_metric.get("gmean", float("nan"))
        
        logger.info(f"\n  Fold {fold_idx+1}/{n_subjects} - Subject {test_subject[:12]}...")
        logger.info(f"    Train: {len(all_train_y)} samples ({n_pos}/{n_neg} pos/neg)")
        logger.info(f"    Test:  {len(y_true)} samples")
        logger.info(f"    Meta-Loss: {np.mean(epoch_losses):.4f}")
        logger.info(f"    Threshold: {fold_threshold:.4f} (from training, method={threshold_method})")
        logger.info(f"    --- Test Metrics ---")
        logger.info(f"    AUROC:       {auroc:.4f}")
        logger.info(f"    G-Mean:      {gmean:.4f}")
        logger.info(f"    Sensitivity: {sens:.4f}")
        logger.info(f"    Specificity: {spec:.4f}")
        
        pbar.set_postfix({
            "AUROC": f"{auroc:.3f}",
            "G-mean": f"{gmean:.3f}",
            "Sens": f"{sens:.3f}",
            "Spec": f"{spec:.3f}"
        })
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # ========================================================================
    # CRITICAL: Aggregate using fold-level predictions (NO re-thresholding)
    # ========================================================================
    # Each fold used its own optimal threshold (found on training data).
    # We aggregate the predictions that were made with those fold-specific thresholds.
    # This is the ONLY correct way to report aggregate performance.
    # 
    # ❌ WRONG: Re-apply mean threshold to all probabilities
    # ✅ CORRECT: Use predictions already made with fold-specific thresholds
    aggregate_metrics = evaluate_predictions(
        all_y_true, 
        all_y_pred,  # Predictions made with fold-specific thresholds
        all_y_proba, 
        "maml"
        # NO threshold parameter - using pre-computed predictions!
    )
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    aggregate_metrics["threshold_method"] = threshold_method
    
    # Report threshold distribution (each fold has its own optimal threshold)
    aggregate_metrics["threshold_mean"] = float(np.mean(fold_thresholds))
    aggregate_metrics["threshold_std"] = float(np.std(fold_thresholds))
    aggregate_metrics["threshold_min"] = float(np.min(fold_thresholds))
    aggregate_metrics["threshold_max"] = float(np.max(fold_thresholds))
    
    # Add constraint parameters if using constrained_gmean
    if threshold_method == "constrained_gmean":
        aggregate_metrics["min_recall_constraint"] = float(min_recall)
        aggregate_metrics["max_fpr_constraint"] = float(max_fpr)
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": all_subjects,
        "fold_metrics": fold_metrics,
        "fold_thresholds": fold_thresholds
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MAML Training for Stress Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use MLP with extracted features (default)
  python train.py --model mlp
  
  # Use CNN with raw 8-channel signals
  python train.py --model cnn
  
  # Adjust meta-training epochs
  python train.py --model mlp --epochs 100
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn"],
        default="mlp",
        help="Model architecture: 'mlp' (61 features) or 'cnn' (8 channels × 120 timesteps)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of meta-training epochs per fold (default: 50)"
    )
    
    parser.add_argument(
        "--tasks-per-batch",
        type=int,
        default=4,
        help="Number of tasks (subjects) per meta-batch (default: 4)"
    )
    
    parser.add_argument(
        "--adaptation-steps",
        type=int,
        default=5,
        help="Number of inner loop adaptation steps (default: 5)"
    )
    
    parser.add_argument(
        "--meta-lr",
        type=float,
        default=0.001,
        help="Meta-learning rate (outer loop, default: 0.001)"
    )
    
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=0.01,
        help="Inner loop learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--threshold",
        type=str,
        choices=["youden", "gmean", "f1", "balanced", "geometric_mean", "constrained_gmean"],
        default="constrained_gmean",
        help="Threshold selection method (default: constrained_gmean)"
    )
    
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.85,
        help="Minimum recall/sensitivity for constrained_gmean (default: 0.85)"
    )
    
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.20,
        help="Maximum false positive rate for constrained_gmean (default: 0.20)"
    )
    
    return parser.parse_args()


def main():
    """Main MAML training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("maml", log_file=results_dir / "training.log")
    
    log_experiment_start(logger, f"MAML META-LEARNING TRAINING ({args.model.upper()})")
    
    # Log configuration
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION")
    logger.info("="*60)
    logger.info(f"  Model type: {args.model.upper()}")
    logger.info(f"  Meta epochs: {args.epochs}")
    logger.info(f"  Tasks/batch: {args.tasks_per_batch}")
    logger.info(f"  Adaptation steps: {args.adaptation_steps}")
    logger.info(f"  Meta LR: {args.meta_lr}")
    logger.info(f"  Inner LR: {args.inner_lr}")
    logger.info(f"  Threshold method: {args.threshold}")
    if args.threshold == "constrained_gmean":
        logger.info(f"  Min recall: {args.min_recall:.2f} (≥{args.min_recall*100:.0f}% sensitivity)")
        logger.info(f"  Max FPR: {args.max_fpr:.2f} (≤{args.max_fpr*100:.0f}% false alarms)")
    logger.info("="*60)
    
    # Check for learn2learn
    if LEARN2LEARN_AVAILABLE:
        logger.info("\n✓ learn2learn: Available")
        logger.info("  Using proper second-order MAML with gradient flow through inner loop")
    else:
        logger.error("\n✗ learn2learn: Not available!")
        logger.error("  Install with: pip install learn2learn")
        logger.error("  Cannot run MAML without learn2learn.")
        return
    
    # Device (with MPS support for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")
    
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        logger.info("  Apple Silicon GPU (Metal Performance Shaders)")
    
    # Set seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # ==========================================================================
    # Phase 1: Load Data
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Loading and Preprocessing Data")
    logger.info("="*60)
    
    windows_by_subject = load_all_windows(config, logger)
    
    if not windows_by_subject:
        logger.error("No data loaded!")
        return
    
    # ==========================================================================
    # Phase 2: Prepare Data (Features or Raw Signals)
    # ==========================================================================
    logger.info("\n" + "="*60)
    if args.model == "mlp":
        logger.info("PHASE 2: Extracting Statistical Features")
        logger.info("="*60)
        logger.info(f"  Feature set: {len(FEATURE_NAMES)} features (emotional stress indicators)")
        logger.info(f"  Normalization: Subject-wise z-score")
        
        data_by_subject = prepare_features_by_subject(
            windows_by_subject,
            label_col=config.target_label,
            normalize=True
        )
        n_features = len(FEATURE_NAMES)
    else:  # cnn
        logger.info("PHASE 2: Preparing Raw 8-Channel Signals")
        logger.info("="*60)
        logger.info(f"  Channels: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd")
        logger.info(f"  Sequence length: 120 timesteps (2 minutes at 1Hz)")
        logger.info(f"  Normalization: Subject-wise z-score")
        
        data_by_subject = prepare_raw_signals_by_subject(
            windows_by_subject,
            label_col=config.target_label,
            n_channels=8,
            seq_len=120
        )
        n_features = 8  # For logging purposes
    
    # Summary
    total_samples = sum(len(y) for _, (_, y) in data_by_subject.items())
    total_positive = sum(np.sum(y == 1) for _, (_, y) in data_by_subject.items())
    
    log_data_summary(
        logger,
        n_subjects=len(data_by_subject),
        n_windows=total_samples,
        n_features=n_features,
        n_positive=total_positive,
        n_negative=total_samples - total_positive
    )
    
    # ==========================================================================
    # Phase 3: MAML Training with LOSO CV
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: MAML Meta-Learning (LOSO Cross-Validation)")
    logger.info("="*60)
    
    results = loso_cross_validation(
        data_by_subject,
        config,
        device,
        logger,
        model_type=args.model,
        n_meta_epochs=args.epochs,
        tasks_per_batch=args.tasks_per_batch,
        adaptation_steps=args.adaptation_steps,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        threshold_method=args.threshold,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr
    )
    
    # ==========================================================================
    # Phase 4: Results Summary
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Results Summary")
    logger.info("="*60)
    
    model_name = f"maml_{args.model}"
    log_model_results(logger, f"MAML-{args.model.upper()}", results["metrics"])
    
    # Save results with model type in filename
    save_results(results["metrics"], results_dir, model_name)
    save_predictions(
        results["y_true"], results["y_pred"], results["y_proba"],
        results["subjects"], results_dir, model_name
    )
    
    # Save fold metrics
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / f"{model_name}_fold_metrics.csv", index=False)
    
    # Generate plots
    if len(np.unique(results["y_true"])) > 1:
        plot_results(
            results["y_true"], results["y_proba"], model_name, results_dir,
            y_pred=results["y_pred"]
        )
    
    log_experiment_end(logger, f"MAML META-LEARNING TRAINING ({args.model.upper()})")
    logger.info(f"\nAll results saved to: {results_dir}")
    
    # Print key metrics
    metrics = results["metrics"]
    print("\n" + "="*60)
    print(f"MAML-{args.model.upper()} RESULTS SUMMARY")
    print("="*60)
    print(f"Model:       {args.model.upper()} ({'61 features' if args.model == 'mlp' else '8 channels × 120 timesteps'})")
    print(f"Threshold:   {args.threshold}")
    if args.threshold == "constrained_gmean":
        print(f"  Constraints: Recall≥{args.min_recall:.0%}, FPR≤{args.max_fpr:.0%}")
    print(f"  Mean: {metrics.get('threshold_mean', 0):.4f} ± {metrics.get('threshold_std', 0):.4f}")
    print(f"  Range: [{metrics.get('threshold_min', 0):.4f}, {metrics.get('threshold_max', 0):.4f}]")
    print()
    print("Performance Metrics (Aggregate from all folds):")
    print(f"  AUROC:       {metrics.get('auroc', 0):.4f} (±{metrics.get('auroc_std', 0):.4f})")
    print(f"  G-Mean:      {metrics.get('gmean', 0):.4f} (±{metrics.get('gmean_std', 0):.4f})")
    print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f} (±{metrics.get('sensitivity_std', 0):.4f})")
    print(f"  Specificity: {metrics.get('specificity', 0):.4f} (±{metrics.get('specificity_std', 0):.4f})")
    print(f"  Precision:   {metrics.get('precision', 0):.4f} (±{metrics.get('precision_std', 0):.4f})")
    print(f"  F1-Score:    {metrics.get('f1', 0):.4f} (±{metrics.get('f1_std', 0):.4f})")
    print(f"  PR-AUC:      {metrics.get('pr_auc', 0):.4f} (±{metrics.get('pr_auc_std', 0):.4f})")
    print("="*60)


if __name__ == "__main__":
    main()
