"""
MAML Training for Personalized Stress Prediction.

Implements Model-Agnostic Meta-Learning (MAML) following Finn et al. 2017.
Uses learn2learn for proper second-order gradient computation.

Key fixes from original implementation:
1. Uses learn2learn.algorithms.MAML for correct gradient flow
2. Uses extracted statistical features (not flattened raw signals)
3. Uses BALANCED k-shot sampling for support sets
4. Computes threshold on training data to avoid data leakage

References:
- https://arxiv.org/pdf/1703.03400 (MAML paper)
- https://github.com/learnables/learn2learn
"""

import sys
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
                           class_weights: Optional[torch.Tensor] = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a subject with adaptation.
    
    Split subject data into adapt set (for fine-tuning) and eval set (for metrics).
    
    Args:
        maml: learn2learn MAML wrapper
        X: All features for subject
        y: All labels for subject
        device: Torch device
        adaptation_steps: Steps for adaptation
        adapt_fraction: Fraction of data to use for adaptation
        class_weights: Optional class weights
    
    Returns:
        Tuple of (y_true, y_pred, y_proba) for eval set
    """
    n_samples = len(y)
    n_adapt = max(5, int(n_samples * adapt_fraction))
    
    # Ensure we have enough samples
    if n_samples <= n_adapt:
        n_adapt = max(2, n_samples // 3)
    
    # Split into adapt and eval
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    adapt_indices = indices[:n_adapt]
    eval_indices = indices[n_adapt:]
    
    if len(eval_indices) == 0:
        eval_indices = indices  # Use all for eval if too few samples
    
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
            for w in windows:
                w["subject_id"] = subject_id
            windows_by_subject[subject_id] = windows
            successful += 1
            total_windows += len(windows)
            pbar.set_postfix({"OK": successful, "Windows": total_windows})
    
    pbar.close()
    logger.info(f"Loaded {successful} subjects with {total_windows} windows")
    
    return windows_by_subject


# =============================================================================
# LOSO Cross-Validation
# =============================================================================

def loso_cross_validation(features_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          config: Config,
                          device: torch.device,
                          logger,
                          n_meta_epochs: int = 100,
                          tasks_per_batch: int = 4,
                          adaptation_steps: int = 5,
                          meta_lr: float = 0.001,
                          inner_lr: float = 0.01,
                          threshold_method: str = "youden") -> Dict:
    """
    Perform LOSO cross-validation with MAML.
    
    For each fold:
    1. Meta-train on all subjects except test subject
    2. Adapt to test subject using small portion of their data
    3. Evaluate on remaining test subject data
    
    Args:
        features_by_subject: Dict mapping subject_id to (X, y)
        config: Configuration object
        device: Torch device
        logger: Logger instance
        n_meta_epochs: Number of meta-training epochs per fold
        tasks_per_batch: Tasks per meta-batch
        adaptation_steps: Inner loop adaptation steps
        meta_lr: Meta-learning rate (outer loop)
        inner_lr: Adaptation learning rate (inner loop)
        threshold_method: Method for finding decision threshold
    
    Returns:
        Results dictionary
    """
    if not LEARN2LEARN_AVAILABLE:
        raise ImportError("learn2learn is required for MAML. Install with: pip install learn2learn")
    
    subjects = list(features_by_subject.keys())
    n_subjects = len(subjects)
    
    # Determine feature dimension
    sample_X, _ = list(features_by_subject.values())[0]
    input_dim = sample_X.shape[1]
    
    logger.info(f"\n{'='*60}")
    logger.info("MAML LOSO Cross-Validation")
    logger.info(f"{'='*60}")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Feature dim: {input_dim}")
    logger.info(f"  Meta epochs/fold: {n_meta_epochs}")
    logger.info(f"  Tasks per batch: {tasks_per_batch}")
    logger.info(f"  Adaptation steps: {adaptation_steps}")
    logger.info(f"  Meta LR (outer): {meta_lr}")
    logger.info(f"  Inner LR: {inner_lr}")
    logger.info(f"  Threshold method: {threshold_method}")
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
        train_features = {s: features_by_subject[s] for s in train_subjects}
        
        # Compute class weights from training data
        all_train_y = np.concatenate([features_by_subject[s][1] for s in train_subjects])
        n_pos = np.sum(all_train_y == 1)
        n_neg = np.sum(all_train_y == 0)
        
        if n_pos > 0 and n_neg > 0:
            weight_pos = n_neg / n_pos
            class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32).to(device)
        else:
            class_weights = None
        
        # Create meta-dataset for training (balanced sampling!)
        meta_dataset = StressMetaDataset(
            train_features,
            k_support=10,  # Support set size
            k_query=20     # Query set size
        )
        
        # Create model and wrap with learn2learn MAML
        model = create_maml_model(
            model_type="mlp",
            input_dim=input_dim,
            hidden_dim=64
        ).to(device)
        
        # Wrap with MAML - this handles gradient computation correctly!
        maml = MAML(model, lr=inner_lr, first_order=False)  # first_order=False for full MAML
        
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
        
        # Get training predictions for threshold selection
        train_y_true_all = []
        train_y_proba_all = []
        
        for train_subj in train_subjects[:5]:  # Use subset for efficiency
            X, y = features_by_subject[train_subj]
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            _, _, proba = evaluate_adapted_model(
                maml, X_tensor, y_tensor, device,
                adaptation_steps=adaptation_steps,
                class_weights=class_weights
            )
            # Note: This uses eval portion, but for threshold we need more coverage
            # Use raw model prediction on all data
            maml.module.eval()
            with torch.no_grad():
                all_proba = F.softmax(maml.module(X_tensor.to(device)), dim=-1)[:, 1]
            train_y_true_all.extend(y)
            train_y_proba_all.extend(all_proba.cpu().numpy())
        
        train_y_true_all = np.array(train_y_true_all)
        train_y_proba_all = np.array(train_y_proba_all)
        
        # Find optimal threshold on training data
        fold_threshold, _ = find_optimal_threshold(
            train_y_true_all, train_y_proba_all, method=threshold_method
        )
        fold_thresholds.append(fold_threshold)
        
        # Evaluate on test subject
        test_X, test_y = features_by_subject[test_subject]
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
        test_y_tensor = torch.tensor(test_y, dtype=torch.long)
        
        y_true, y_pred_raw, y_proba = evaluate_adapted_model(
            maml, test_X_tensor, test_y_tensor, device,
            adaptation_steps=adaptation_steps,
            class_weights=class_weights
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
        
        logger.info(f"\n  Fold {fold_idx+1}/{n_subjects} - Subject {test_subject[:12]}...")
        logger.info(f"    Train: {len(all_train_y)} samples ({n_pos}/{n_neg} pos/neg)")
        logger.info(f"    Test:  {len(y_true)} samples")
        logger.info(f"    Meta-Loss: {np.mean(epoch_losses):.4f}")
        logger.info(f"    Threshold: {fold_threshold:.4f} (from training, method={threshold_method})")
        logger.info(f"    --- Metrics ---")
        logger.info(f"    AUROC:       {auroc:.4f}")
        logger.info(f"    Sensitivity: {sens:.4f}")
        logger.info(f"    Specificity: {spec:.4f}")
        
        pbar.set_postfix({
            "AUROC": f"{auroc:.3f}", 
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
    
    # Final threshold (mean of fold thresholds)
    final_threshold = np.mean(fold_thresholds)
    logger.info(f"\nFinal threshold (mean of folds): {final_threshold:.4f}")
    
    # Aggregate metrics
    aggregate_metrics = evaluate_predictions(
        all_y_true, all_y_pred, all_y_proba, "maml",
        threshold=final_threshold
    )
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    aggregate_metrics["threshold_method"] = threshold_method
    aggregate_metrics["final_threshold"] = final_threshold
    
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

def main():
    """Main MAML training pipeline."""
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("maml", log_file=results_dir / "training.log")
    
    log_experiment_start(logger, "MAML META-LEARNING TRAINING (learn2learn)")
    
    # Check for learn2learn
    if LEARN2LEARN_AVAILABLE:
        logger.info("✓ learn2learn: Available")
        logger.info("  Using proper second-order MAML with gradient flow through inner loop")
    else:
        logger.error("✗ learn2learn: Not available!")
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
    # Phase 2: Extract Features
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Extracting Statistical Features")
    logger.info("="*60)
    logger.info(f"  Feature set: {len(FEATURE_NAMES)} features (emotional stress indicators)")
    logger.info(f"  Normalization: Subject-wise z-score")
    
    features_by_subject = prepare_features_by_subject(
        windows_by_subject,
        label_col=config.target_label,
        normalize=True
    )
    
    # Summary
    total_samples = sum(len(y) for _, (_, y) in features_by_subject.items())
    total_positive = sum(np.sum(y == 1) for _, (_, y) in features_by_subject.items())
    
    log_data_summary(
        logger,
        n_subjects=len(features_by_subject),
        n_windows=total_samples,
        n_features=len(FEATURE_NAMES),
        n_positive=total_positive,
        n_negative=total_samples - total_positive
    )
    
    # ==========================================================================
    # Phase 3: MAML Training with LOSO CV
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: MAML Meta-Learning (LOSO Cross-Validation)")
    logger.info("="*60)
    
    # Hyperparameters
    N_META_EPOCHS = 50          # Meta-training epochs per fold
    TASKS_PER_BATCH = 4         # Tasks (subjects) per meta-batch
    ADAPTATION_STEPS = 5        # Inner loop gradient steps
    META_LR = 0.001             # Outer loop learning rate
    INNER_LR = 0.01             # Inner loop learning rate
    THRESHOLD_METHOD = "youden" # Threshold selection method
    
    logger.info(f"\nHyperparameters:")
    logger.info(f"  Meta epochs: {N_META_EPOCHS}")
    logger.info(f"  Tasks/batch: {TASKS_PER_BATCH}")
    logger.info(f"  Adapt steps: {ADAPTATION_STEPS}")
    logger.info(f"  Meta LR: {META_LR}")
    logger.info(f"  Inner LR: {INNER_LR}")
    logger.info(f"  Threshold: {THRESHOLD_METHOD}")
    
    results = loso_cross_validation(
        features_by_subject,
        config,
        device,
        logger,
        n_meta_epochs=N_META_EPOCHS,
        tasks_per_batch=TASKS_PER_BATCH,
        adaptation_steps=ADAPTATION_STEPS,
        meta_lr=META_LR,
        inner_lr=INNER_LR,
        threshold_method=THRESHOLD_METHOD
    )
    
    # ==========================================================================
    # Phase 4: Results Summary
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Results Summary")
    logger.info("="*60)
    
    log_model_results(logger, "MAML", results["metrics"])
    
    # Save results
    save_results(results["metrics"], results_dir, "maml")
    save_predictions(
        results["y_true"], results["y_pred"], results["y_proba"],
        results["subjects"], results_dir, "maml"
    )
    
    # Save fold metrics
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / "maml_fold_metrics.csv", index=False)
    
    # Generate plots
    if len(np.unique(results["y_true"])) > 1:
        plot_results(
            results["y_true"], results["y_proba"], "maml", results_dir,
            y_pred=results["y_pred"]
        )
    
    log_experiment_end(logger, "MAML META-LEARNING TRAINING")
    logger.info(f"\nAll results saved to: {results_dir}")
    
    # Print key metrics
    metrics = results["metrics"]
    print("\n" + "="*60)
    print("MAML RESULTS SUMMARY")
    print("="*60)
    print(f"AUROC:       {metrics.get('auroc', 0):.4f} (±{metrics.get('auroc_std', 0):.4f})")
    print(f"PR-AUC:      {metrics.get('pr_auc', 0):.4f} (±{metrics.get('pr_auc_std', 0):.4f})")
    print(f"Sensitivity: {metrics.get('sensitivity', 0):.4f} (±{metrics.get('sensitivity_std', 0):.4f})")
    print(f"Specificity: {metrics.get('specificity', 0):.4f} (±{metrics.get('specificity_std', 0):.4f})")
    print(f"F1-Score:    {metrics.get('f1', 0):.4f} (±{metrics.get('f1_std', 0):.4f})")
    print(f"Threshold:   {metrics.get('final_threshold', 0.5):.4f} (method={THRESHOLD_METHOD})")
    print("="*60)


if __name__ == "__main__":
    main()
