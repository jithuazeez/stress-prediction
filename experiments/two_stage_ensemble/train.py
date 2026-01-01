"""
Two-Stage Ensemble Training with LOSO Cross-Validation

Implements two decision strategies:

A) LR-Only (Maximize Recall):
   1. Logistic Regression: Makes all decisions (high recall threshold)
   2. TCN: Provides confidence scores only (NEVER vetoes LR)
   
   Decision Logic:
   prediction = (LR_proba >= LR_threshold)
   confidence = TCN_proba if prediction==1 else (1-TCN_proba)

B) Hard AND Cascade (Minimize False Alarms):
   1. Logistic Regression: Screening stage
   2. TCN: Confirmation gate (must agree for positive prediction)
   
   Decision Logic:
   if LR_proba < LR_threshold:
       prediction = 0  # NO STRESS (skip TCN)
   else:
       prediction = 1 if TCN_proba >= TCN_threshold else 0

Architecture:
- LR trains on extracted features from aligned 1Hz data + HRV
- TCN trains on raw multichannel time series (8x120)
- Each model uses subject-wise normalization
- Asymmetric threshold optimization on training data
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
import joblib

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Shared modules
from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_signals  # Using new 4 Hz alignment
from shared.windowing import create_labeled_windows, parse_stress_events, compute_subject_stats
from shared.evaluation import evaluate_predictions, aggregate_fold_metrics, save_results, save_predictions, plot_results
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import setup_logger, log_experiment_start, log_experiment_end, log_data_summary

# Classical ML components
try:
    # sys.path.insert(0, str(Path(__file__).parent.parent / "01_classical_ml"))
    from classical_ml.feature_extraction import BasicFeatureExtractor
except ImportError:
    from experiments.classical_ml.feature_extraction import BasicFeatureExtractor

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# TCN components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "08_tcn"))
    from tcn.dataset import VitaStressTCNDataset
    from tcn.model import create_tcn_model, FocalLoss
except ImportError:
    from experiments.tcn.dataset import VitaStressTCNDataset
    from experiments.tcn.model import create_tcn_model, FocalLoss

# Two-stage threshold selection
from threshold_selection import (
    find_recall_first_threshold,
    find_far_first_threshold,
    apply_two_stage_decision,
    apply_lr_decision_with_tcn_confidence
)


def load_and_prepare_windows(config: Config, logger) -> Dict[str, List[Dict]]:
    """
    Load and process all subjects, returning windows by subject.
    
    Returns:
        Dictionary mapping subject_id to list of window dictionaries
    """
    subjects = get_all_subjects(config.data_path)
    windows_by_subject = {}
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Processing subjects", unit="subject")
    successful = 0
    failed = 0
    
    for subject_folder in pbar:
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            failed += 1
            continue
        
        # Align to 4Hz
        aligned = align_signals(signals, start, end, target_hz=4.0)
        if aligned is None or len(aligned) == 0:
            failed += 1
            continue
        
        # Compute subject stats
        subject_stats = compute_subject_stats(aligned)
        
        # Parse events
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        # Create windows
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
            pbar.set_postfix({"OK": successful, "Fail": failed})
        else:
            failed += 1
    
    pbar.close()
    
    logger.info(f"Loaded {successful} subjects with windows (failed: {failed})")
    return windows_by_subject


def extract_features_from_windows(
    windows: List[Dict],
    extractor: BasicFeatureExtractor,
    logger
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract classical ML features from windows.
    
    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
    """
    rows = []
    
    for window in windows:
        features = extractor.extract_from_window(window["window_data"])
        row = {**features, "label": window.get("label_3min", 0)}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col != "label"]
    X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = df["label"].values
    
    # Impute missing values
    # imputer = SimpleImputer(strategy='median')
    # X = imputer.fit_transform(X_df)
    X = np.nan_to_num(X_df, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, feature_cols


def train_lr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    logger
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train Logistic Regression model.
    
    Returns:
        Trained model and fitted scaler
    """
    # Scale features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    
    # Train LR with class weights
    lr_model = LogisticRegression(
        max_iter=1000,
        C=5.0,
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    return lr_model


def train_tcn_model(
    train_windows: List[Dict],
    config: Config,
    device: torch.device,
    logger,
    n_epochs: int = 100
) -> nn.Module:
    """
    Train TCN model on raw time series.
    
    Returns:
        Trained TCN model
    """
    # Create dataset
    train_dataset = VitaStressTCNDataset(
        train_windows,
        config.target_label,
        seq_len=480,  # 480 for 4 Hz
        normalize=True,
        normalization_mode="subject"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = create_tcn_model(
        num_inputs=8,
        num_classes=2,
        num_channels=[16, 16, 16, 16, 16, 16],
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128],  # Extended for 480 timesteps
        dropout=0.3,
        fc_hidden_dim=128,
        use_last_timestep=True
    ).to(device)
    
    # Loss and optimizer
    n_pos = sum(1 for w in train_windows if w.get(config.target_label, 0) == 1)
    n_neg = len(train_windows) - n_pos
    n_total = len(train_windows)
    
    # sklearn balanced weights
    weight_class_0 = n_total / (2 * n_neg)
    weight_class_1 = n_total / (2 * n_pos)
    class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"    TCN early stopping at epoch {epoch+1}/{n_epochs}")
            break
    
    return model


def get_tcn_probabilities(
    model: nn.Module,
    windows: List[Dict],
    config: Config,
    device: torch.device
) -> np.ndarray:
    """
    Get TCN predicted probabilities.
    
    Returns:
        Array of probabilities for positive class
    """
    dataset = VitaStressTCNDataset(
        windows,
        config.target_label,
        seq_len=480,  # 480 for 4 Hz
        normalize=True,
        normalization_mode="subject"
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_proba = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            proba = torch.softmax(logits, dim=-1)[:, 1]
            all_proba.extend(proba.cpu().numpy())
    
    return np.array(all_proba)


def stacked_loso_cross_validation(
    windows_by_subject: Dict[str, List[Dict]],
    config: Config,
    device: torch.device,
    logger,
    n_epochs_tcn: int = 100,
    meta_model_type: str = "logistic_regression"
) -> Dict:
    """
    Stacked ensemble with nested LOSO cross-validation.
    
    Algorithm:
    For each outer fold (test subject):
        1. PHASE 1: Nested LOSO on training subjects
           - For each training subject, predict it using models trained on other training subjects
           - This generates out-of-fold predictions for meta-training
        2. PHASE 2: Train meta-model on out-of-fold predictions
        3. PHASE 3: Train final base models (LR, TCN) on all training subjects
        4. PHASE 4: Get predictions from final base models on test subject
        5. PHASE 5: Meta-model makes final prediction
    
    Args:
        windows_by_subject: Dictionary of windows per subject
        config: Experiment configuration
        device: PyTorch device
        logger: Logger instance
        n_epochs_tcn: Number of epochs for TCN training
        meta_model_type: Type of meta-model ("logistic_regression", "xgboost", "random_forest")
    
    Returns:
        Dictionary with metrics, predictions, and fold results
    """
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STACKED ENSEMBLE LOSO CV (Nested)")
    logger.info(f"{'='*60}")
    logger.info(f"Total Subjects: {n_subjects}")
    logger.info(f"Meta-Model: {meta_model_type}")
    logger.info(f"Expected trainings: {n_subjects} outer × ({n_subjects-1} inner + 3 final) = {n_subjects * (n_subjects - 1 + 3)}")
    logger.info(f"{'='*60}\n")
    
    # Initialize feature extractor
    extractor = BasicFeatureExtractor()
    
    # Storage for final results
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    
    # ========== OUTER LOOP: Main LOSO ==========
    outer_pbar = tqdm(enumerate(subjects), total=n_subjects, desc="Outer folds", unit="fold")
    
    for outer_idx, outer_test_subject in outer_pbar:
        outer_pbar.set_description(f"Outer {outer_idx+1}/{n_subjects} (Test={outer_test_subject[:8]}...)")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OUTER FOLD {outer_idx+1}/{n_subjects}: Test={outer_test_subject}")
        logger.info(f"{'='*60}")
        
        # Outer split
        outer_train_subjects = [s for s in subjects if s != outer_test_subject]
        outer_test_windows = windows_by_subject[outer_test_subject]
        
        if len(outer_test_windows) == 0:
            logger.warning(f"  No windows for test subject {outer_test_subject}, skipping...")
            continue
        
        # ========== PHASE 1: Generate Meta-Features via Nested LOSO ==========
        logger.info(f"\nPhase 1: Nested LOSO for meta-features ({len(outer_train_subjects)} inner folds)...")
        
        meta_X_train = []
        meta_y_train = []
        
        inner_pbar = tqdm(enumerate(outer_train_subjects), total=len(outer_train_subjects), 
                         desc="  Inner folds", leave=False, unit="fold")
        
        for inner_idx, inner_val_subject in inner_pbar:
            inner_pbar.set_description(f"  Inner {inner_idx+1}/{len(outer_train_subjects)} (Val={inner_val_subject[:8]}...)")
            
            # Inner split
            inner_train_subjects = [s for s in outer_train_subjects if s != inner_val_subject]
            
            # Collect windows
            inner_train_windows = []
            for s in inner_train_subjects:
                inner_train_windows.extend(windows_by_subject[s])
            
            inner_val_windows = windows_by_subject[inner_val_subject]
            
            if len(inner_train_windows) == 0 or len(inner_val_windows) == 0:
                continue
            
            # --- Train LR on inner_train ---
            X_train_inner, y_train_inner, _ = extract_features_from_windows(
                inner_train_windows, extractor, logger
            )
            X_val_inner, y_val_inner, _ = extract_features_from_windows(
                inner_val_windows, extractor, logger
            )
            
            # Impute and scale
            imputer_inner = SimpleImputer(strategy='median')
            X_train_inner = imputer_inner.fit_transform(X_train_inner)
            X_val_inner = imputer_inner.transform(X_val_inner)
            
            scaler_inner = StandardScaler()
            X_train_inner_scaled = scaler_inner.fit_transform(X_train_inner)
            X_val_inner_scaled = scaler_inner.transform(X_val_inner)
            
            lr_inner = train_lr_model(X_train_inner_scaled, y_train_inner, logger)
            
            # --- Train TCN on inner_train ---
            tcn_inner = train_tcn_model(
                inner_train_windows, config, device, logger, n_epochs=n_epochs_tcn
            )
            
            # --- Get out-of-fold predictions on inner_val ---
            lr_val_proba = lr_inner.predict_proba(X_val_inner_scaled)[:, 1]
            tcn_val_proba = get_tcn_probabilities(
                tcn_inner, inner_val_windows, config, device
            )
            
            # Store as meta-features (out-of-fold!)
            for lr_p, tcn_p, label in zip(lr_val_proba, tcn_val_proba, y_val_inner):
                meta_X_train.append([lr_p, tcn_p])
                meta_y_train.append(label)
        
        inner_pbar.close()
        
        meta_X_train = np.array(meta_X_train)
        meta_y_train = np.array(meta_y_train)
        
        logger.info(f"  Generated {len(meta_X_train)} meta-training samples")
        logger.info(f"  Meta class distribution: {np.sum(meta_y_train)} positive, "
                   f"{len(meta_y_train) - np.sum(meta_y_train)} negative")
        
        # ========== PHASE 2: Train Meta-Model ==========
        logger.info(f"\nPhase 2: Training meta-model ({meta_model_type})...")
        
        if meta_model_type == "logistic_regression":
            meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif meta_model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                n_pos = np.sum(meta_y_train == 1)
                n_neg = np.sum(meta_y_train == 0)
                scale_pos_weight = n_neg / max(n_pos, 1)
                meta_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42
                )
            except ImportError:
                logger.warning("  XGBoost not available, falling back to Logistic Regression")
                meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif meta_model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            meta_model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            )
        else:
            logger.warning(f"  Unknown meta-model type: {meta_model_type}, using Logistic Regression")
            meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        
        meta_model.fit(meta_X_train, meta_y_train)
        
        # Log meta-model weights if LR
        if meta_model_type == "logistic_regression":
            logger.info(f"  Meta-model weights: LR={meta_model.coef_[0][0]:.4f}, TCN={meta_model.coef_[0][1]:.4f}")
            logger.info(f"  Meta-model intercept: {meta_model.intercept_[0]:.4f}")
        
        # ========== PHASE 3: Train Final Base Models on ALL Training Subjects ==========
        logger.info(f"\nPhase 3: Training final base models on {len(outer_train_subjects)} subjects...")
        
        outer_train_windows = []
        for s in outer_train_subjects:
            outer_train_windows.extend(windows_by_subject[s])
        
        # Train LR
        X_train_outer, y_train_outer, _ = extract_features_from_windows(
            outer_train_windows, extractor, logger
        )
        
        imputer_outer = SimpleImputer(strategy='median')
        X_train_outer = imputer_outer.fit_transform(X_train_outer)
        
        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        
        lr_final = train_lr_model(X_train_outer_scaled, y_train_outer, logger)
        
        # Train TCN
        tcn_final = train_tcn_model(
            outer_train_windows, config, device, logger, n_epochs=n_epochs_tcn
        )
        
        # ========== PHASE 4: Get Test Predictions from Base Models ==========
        logger.info(f"\nPhase 4: Getting test predictions for {outer_test_subject}...")
        
        X_test_outer, y_test_outer, _ = extract_features_from_windows(
            outer_test_windows, extractor, logger
        )
        
        X_test_outer = imputer_outer.transform(X_test_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)
        
        lr_test_proba = lr_final.predict_proba(X_test_outer_scaled)[:, 1]
        tcn_test_proba = get_tcn_probabilities(
            tcn_final, outer_test_windows, config, device
        )
        
        # Create meta-features for test
        meta_X_test = np.column_stack([lr_test_proba, tcn_test_proba])
        
        # ========== PHASE 5: Meta-Model Final Prediction ==========
        logger.info(f"\nPhase 5: Meta-model making final predictions...")
        
        y_pred = meta_model.predict(meta_X_test)
        y_proba = meta_model.predict_proba(meta_X_test)[:, 1]
        
        # Store results
        all_y_true.extend(y_test_outer)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([outer_test_subject] * len(y_test_outer))
        
        # Evaluate fold
        fold_metric = evaluate_predictions(
            y_test_outer,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name="stacked_ensemble"
        )
        fold_metric["subject"] = outer_test_subject
        fold_metrics.append(fold_metric)
        
        # Log fold metrics
        logger.info(f"\n  Fold {outer_idx+1} Test Metrics:")
        logger.info(f"    Recall:     {fold_metric['recall']:.4f}")
        logger.info(f"    Precision:  {fold_metric['precision']:.4f}")
        logger.info(f"    Bal. Acc:   {fold_metric.get('balanced_accuracy', float('nan')):.4f}")
        logger.info(f"    F1:         {fold_metric['f1']:.4f}")
        logger.info(f"    G-mean:     {fold_metric['gmean']:.4f}")
        logger.info(f"    Specificity:{fold_metric['specificity']:.4f}")
        logger.info(f"    FAR:        {fold_metric.get('false_alarm_rate', float('nan')):.4f}")
        logger.info(f"    AUROC:      {fold_metric.get('auroc', float('nan')):.4f}")
        logger.info(f"    PR-AUC:     {fold_metric.get('pr_auc', float('nan')):.4f}")
        
        # Update progress bar
        outer_pbar.set_postfix({
            "Recall": f"{fold_metric['recall']:.3f}",
            "Prec": f"{fold_metric['precision']:.3f}",
            "Gmean": f"{fold_metric['gmean']:.3f}"
        })
    
    outer_pbar.close()
    
    # ========== Aggregate Results ==========
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    aggregate_metrics = evaluate_predictions(
        all_y_true,
        y_pred=all_y_pred,
        y_proba=all_y_proba,
        model_name="stacked_ensemble"
    )
    
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    # Log overall results
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL STACKED ENSEMBLE PERFORMANCE")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(all_y_true)}")
    logger.info(f"Positive: {np.sum(all_y_true)} ({100*np.sum(all_y_true)/len(all_y_true):.1f}%)")
    logger.info(f"Negative: {len(all_y_true)-np.sum(all_y_true)} ({100*(len(all_y_true)-np.sum(all_y_true))/len(all_y_true):.1f}%)")
    logger.info(f"\nMetrics:")
    logger.info(f"  AUROC:      {aggregate_metrics.get('auroc', float('nan')):.4f}")
    logger.info(f"  PR-AUC:     {aggregate_metrics.get('pr_auc', float('nan')):.4f}")
    logger.info(f"  Recall:     {aggregate_metrics.get('recall', float('nan')):.4f}")
    logger.info(f"  Precision:  {aggregate_metrics.get('precision', float('nan')):.4f}")
    logger.info(f"  Specificity:{aggregate_metrics.get('specificity', float('nan')):.4f}")
    logger.info(f"  F1:         {aggregate_metrics.get('f1', float('nan')):.4f}")
    logger.info(f"  G-mean:     {aggregate_metrics.get('gmean', float('nan')):.4f}")
    logger.info(f"  FAR:        {aggregate_metrics.get('false_alarm_rate', float('nan')):.4f}")
    logger.info(f"{'='*60}\n")
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": np.array(all_subjects),
        "fold_metrics": fold_metrics
    }


def loso_cross_validation(
    windows_by_subject: Dict[str, List[Dict]],
    config: Config,
    device: torch.device,
    logger,
    min_recall_lr: float = 0.75,
    max_far_tcn: float = 0.25,
    n_epochs_tcn: int = 100,
    decision_strategy: str = "lr_only"
) -> Dict:
    """
    Perform LOSO cross-validation with two-stage ensemble.
    
    Algorithm:
    1. For each test subject:
       2. Split data: test vs. train
       3. Train LR on extracted features
       4. Train TCN on raw time series
       5. Get training probabilities from both models
       6. Select asymmetric thresholds on TRAINING data:
          - LR: Recall-first (min recall constraint)
          - TCN: FAR-first (max FAR constraint) OR just for confidence
       7. Apply decision strategy on TEST data
       8. Evaluate and store results
    9. Aggregate across all folds
    
    Args:
        windows_by_subject: Dictionary of windows per subject
        config: Experiment configuration
        device: PyTorch device
        logger: Logger instance
        min_recall_lr: Minimum recall for LR (default: 0.75)
        max_far_tcn: Maximum FAR for TCN (default: 0.25)
        n_epochs_tcn: Number of epochs for TCN training
        decision_strategy: Decision logic (default: "lr_only")
            - "lr_only": LR makes decision, TCN provides confidence
            - "and_cascade": LR screens, TCN must confirm (hard AND)
    
    Returns:
        Dictionary with metrics, predictions, and fold results
    """
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TWO-STAGE ENSEMBLE LOSO CV")
    logger.info(f"{'='*60}")
    logger.info(f"Subjects: {n_subjects}")
    logger.info(f"Decision Strategy: {decision_strategy}")
    if decision_strategy == "lr_only":
        logger.info(f"  LR: Makes all decisions (min recall={min_recall_lr*100:.0f}%)")
        logger.info(f"  TCN: Provides confidence scores only")
    else:  # and_cascade
        logger.info(f"  LR: Screening (min recall={min_recall_lr*100:.0f}%)")
        logger.info(f"  TCN: Confirmation gate (max FAR={max_far_tcn*100:.0f}%)")
    logger.info(f"{'='*60}\n")
    
    # Initialize feature extractor
    extractor = BasicFeatureExtractor()
    
    # Storage
    all_y_true = []
    all_y_pred = []
    all_y_proba_lr = []
    all_y_proba_tcn = []
    all_confidence = []  # TCN confidence scores (for lr_only mode)
    all_subjects = []
    fold_metrics = []
    
    # Progress bar
    pbar = tqdm(enumerate(subjects), total=n_subjects, desc="LOSO folds", unit="fold")
    
    for fold_idx, test_subject in pbar:
        pbar.set_description(f"Fold {fold_idx+1}/{n_subjects} ({test_subject[:8]}...)")
        
        # ========== STEP 1: SPLIT DATA ==========
        train_windows = []
        test_windows = []
        
        for subject_id, windows in windows_by_subject.items():
            # Add subject_id to each window
            for w in windows:
                w["subject_id"] = subject_id
            
            if subject_id == test_subject:
                test_windows.extend(windows)
            else:
                train_windows.extend(windows)
        
        if len(train_windows) == 0 or len(test_windows) == 0:
            continue
        
        # ========== STEP 2: TRAIN LR MODEL ==========
        logger.info(f"Fold {fold_idx+1}: Training LR...")
        X_train, y_train, feature_names = extract_features_from_windows(
            train_windows, extractor, logger
        )
        X_test, y_test, _ = extract_features_from_windows(
            test_windows, extractor, logger
        )

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        lr_scaler = StandardScaler()
        X_train_scaled = lr_scaler.fit_transform(X_train)
        X_test_scaled = lr_scaler.transform(X_test)
        lr_model = train_lr_model(X_train_scaled, y_train, logger)
        
        # ========== STEP 3: TRAIN TCN MODEL ==========
        logger.info(f"Fold {fold_idx+1}: Training TCN...")
        tcn_model = train_tcn_model(
            train_windows, config, device, logger, n_epochs=n_epochs_tcn
        )
        
        # ========== STEP 4: GET TRAINING PROBABILITIES ==========
        # LR training 
        lr_train_proba = lr_model.predict_proba(X_train_scaled)[:, 1]
        
        # TCN training probabilities
        tcn_train_proba = get_tcn_probabilities(
            tcn_model, train_windows, config, device
        )
        
        # ========== STEP 5: SELECT THRESHOLDS (TRAINING DATA) ==========
        # LR: Recall-first threshold
        lr_threshold, lr_train_metrics = find_recall_first_threshold(
            y_train, lr_train_proba, min_recall=min_recall_lr
        )
        
        # TCN: FAR-first threshold
        tcn_threshold, tcn_train_metrics = find_far_first_threshold(
            y_train, tcn_train_proba, max_far=max_far_tcn
        )
        
        logger.info(f"  LR threshold: {lr_threshold:.4f} (Recall={lr_train_metrics['recall']:.3f})")
        logger.info(f"  TCN threshold: {tcn_threshold:.4f} (FAR={tcn_train_metrics['false_alarm_rate']:.3f})")
        
        # ========== STEP 6: TEST ON HELD-OUT SUBJECT ==========
        # Get test probabilities
      
        lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        tcn_test_proba = get_tcn_probabilities(
            tcn_model, test_windows, config, device
        )
        
        # Apply decision strategy
        if decision_strategy == "lr_only":
            # LR makes decision, TCN provides confidence
            y_pred, confidence_scores = apply_lr_decision_with_tcn_confidence(
                lr_test_proba, tcn_test_proba, lr_threshold
            )
        else:  # and_cascade
            # Hard AND cascade: LR screens, TCN confirms
            y_pred = apply_two_stage_decision(
                lr_test_proba, tcn_test_proba, lr_threshold, tcn_threshold
            )
            confidence_scores = None
        
        # ========== STEP 7: EVALUATE ==========
        # Store results
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba_lr.extend(lr_test_proba)
        all_y_proba_tcn.extend(tcn_test_proba)
        if confidence_scores is not None:
            all_confidence.extend(confidence_scores)
        all_subjects.extend([test_subject] * len(y_test))
        
        # Compute fold metrics
        # Use average of LR and TCN probabilities for AUROC calculation
        ensemble_proba = (lr_test_proba + tcn_test_proba) / 2
        
        fold_metric = evaluate_predictions(
            y_test,
            y_pred=y_pred,
            y_proba=ensemble_proba,
            model_name="two_stage_ensemble"
        )
        fold_metric["subject"] = test_subject
        fold_metric["lr_threshold"] = lr_threshold
        fold_metric["tcn_threshold"] = tcn_threshold
        fold_metric["lr_train_recall"] = lr_train_metrics["recall"]
        fold_metric["tcn_train_far"] = tcn_train_metrics["false_alarm_rate"]
        fold_metrics.append(fold_metric)
        
        # Log detailed fold metrics
        logger.info(f"  Test metrics:")
        logger.info(f"    Recall:     {fold_metric['recall']:.4f}")
        logger.info(f"    Precision:  {fold_metric['precision']:.4f}")
        logger.info(f"    Bal. Acc:   {fold_metric.get('balanced_accuracy', float('nan')):.4f}")
        logger.info(f"    F1:         {fold_metric['f1']:.4f}")
        logger.info(f"    G-mean:     {fold_metric['gmean']:.4f}")
        logger.info(f"    Specificity:{fold_metric['specificity']:.4f}")
        logger.info(f"    FAR:        {fold_metric.get('false_alarm_rate', float('nan')):.4f}")
        logger.info(f"    AUROC:      {fold_metric.get('auroc', float('nan')):.4f}")
        logger.info(f"    PR-AUC:     {fold_metric.get('pr_auc', float('nan')):.4f}")
        
        # Log confidence statistics if available
        if confidence_scores is not None:
            avg_confidence = np.mean(confidence_scores)
            avg_confidence_stress = np.mean(confidence_scores[y_pred == 1]) if np.sum(y_pred) > 0 else 0.0
            avg_confidence_nostress = np.mean(confidence_scores[y_pred == 0]) if np.sum(y_pred == 0) > 0 else 0.0
            logger.info(f"  TCN Confidence:")
            logger.info(f"    Overall:    {avg_confidence:.4f}")
            logger.info(f"    Stress:     {avg_confidence_stress:.4f} (n={np.sum(y_pred)})")
            logger.info(f"    No-stress:  {avg_confidence_nostress:.4f} (n={np.sum(y_pred == 0)})")
        
        logger.info("")
        
        # Update progress
        pbar.set_postfix({
            "Recall": f"{fold_metric['recall']:.3f}",
            "Prec": f"{fold_metric['precision']:.3f}",
            "Gmean": f"{fold_metric['gmean']:.3f}"
        })
    
    pbar.close()
    
    # ========== STEP 8: AGGREGATE RESULTS ==========
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba_lr = np.array(all_y_proba_lr)
    all_y_proba_tcn = np.array(all_y_proba_tcn)
    ensemble_proba = (all_y_proba_lr + all_y_proba_tcn) / 2
    
    aggregate_metrics = evaluate_predictions(
        all_y_true,
        y_pred=all_y_pred,
        y_proba=ensemble_proba,
        model_name="two_stage_ensemble"
    )
    
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL PERFORMANCE")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(all_y_true)}")
    logger.info(f"Positive: {np.sum(all_y_true)} ({100*np.sum(all_y_true)/len(all_y_true):.1f}%)")
    logger.info(f"Negative: {len(all_y_true)-np.sum(all_y_true)} ({100*(len(all_y_true)-np.sum(all_y_true))/len(all_y_true):.1f}%)")
    logger.info(f"\nMetrics:")
    logger.info(f"  AUROC:      {aggregate_metrics.get('auroc', float('nan')):.4f}")
    logger.info(f"  PR-AUC:     {aggregate_metrics.get('pr_auc', float('nan')):.4f}")
    logger.info(f"  Recall:     {aggregate_metrics.get('recall', float('nan')):.4f}")
    logger.info(f"  Precision:  {aggregate_metrics.get('precision', float('nan')):.4f}")
    logger.info(f"  Specificity:{aggregate_metrics.get('specificity', float('nan')):.4f}")
    logger.info(f"  F1:         {aggregate_metrics.get('f1', float('nan')):.4f}")
    logger.info(f"  G-mean:     {aggregate_metrics.get('gmean', float('nan')):.4f}")
    logger.info(f"  FAR:        {aggregate_metrics.get('false_alarm_rate', float('nan')):.4f}")
    logger.info(f"{'='*60}\n")
    
    results_dict = {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba_lr": all_y_proba_lr,
        "y_proba_tcn": all_y_proba_tcn,
        "y_proba_ensemble": ensemble_proba,
        "subjects": np.array(all_subjects),
        "fold_metrics": fold_metrics
    }
    
    # Add confidence scores if available
    if len(all_confidence) > 0:
        results_dict["confidence_scores"] = np.array(all_confidence)
    
    return results_dict


def main(use_stacking: bool = False, meta_model_type: str = "logistic_regression"):
    """
    Main training pipeline.
    
    Args:
        use_stacking: If True, use stacked ensemble with nested LOSO.
                     If False, use simple two-stage ensemble (default).
        meta_model_type: Type of meta-model for stacking (default: "logistic_regression")
    """
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    experiment_name = "STACKED ENSEMBLE" if use_stacking else "TWO-STAGE ENSEMBLE"
    logger = setup_logger(
        "two_stage_ensemble",
        log_file=results_dir / "training.log"
    )
    
    log_experiment_start(logger, f"{experiment_name} (LR + TCN)")
    
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")
    
    # Load windows
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Loading data")
    logger.info("-"*50)
    
    windows_by_subject = load_and_prepare_windows(config, logger)
    
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
        n_features=0,  # Dynamic
        n_positive=n_pos,
        n_negative=total_windows - n_pos
    )
    
    # Train ensemble
    logger.info("\n" + "-"*50)
    logger.info(f"PHASE 2: {experiment_name} Training")
    logger.info("-"*50)
    
    if use_stacking:
        # ========== STACKED ENSEMBLE (Nested LOSO) ==========
        logger.info("Using STACKED ENSEMBLE with nested LOSO")
        logger.info("⚠️  WARNING: This will take ~16x longer than regular ensemble!")
        logger.info("   Estimated time: ~24 hours for 21 subjects with TCN\n")
        
        results = stacked_loso_cross_validation(
            windows_by_subject,
            config,
            device,
            logger,
            n_epochs_tcn=100,
            meta_model_type=meta_model_type  # Options: "logistic_regression", "xgboost", "random_forest"
        )
        
        model_name = "stacked_ensemble"
        y_proba_key = "y_proba"
        
    else:
        # ========== SIMPLE TWO-STAGE ENSEMBLE ==========
        logger.info("Using SIMPLE TWO-STAGE ENSEMBLE")
        
        results = loso_cross_validation(
            windows_by_subject,
            config,
            device,
            logger,
            min_recall_lr=0.75,      # LR must catch 75% of stress (high recall)
            max_far_tcn=0.30,        # TCN threshold (not used in lr_only mode)
            n_epochs_tcn=100,
            decision_strategy="lr_only"  # Options: "lr_only" or "and_cascade"
        )
        
        model_name = "two_stage_ensemble"
        y_proba_key = "y_proba_ensemble"
    
    # Save results
    save_results(results["metrics"], results_dir, model_name)
    save_predictions(
        results["y_true"],
        results["y_pred"],
        results[y_proba_key],
        results["subjects"],
        results_dir,
        model_name
    )
    
    # Plot results
    if len(np.unique(results["y_true"])) > 1:
        plot_results(
            results["y_true"],
            results[y_proba_key],
            model_name,
            results_dir,
            y_pred=results["y_pred"]
        )
    
    # Save fold metrics
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / f"{model_name}_fold_metrics.csv", index=False)
    
    logger.info(f"\n✅ All results saved to: {results_dir}")
    log_experiment_end(logger, f"{experiment_name} (LR + TCN)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-Stage Ensemble Training")
    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Use stacked ensemble with nested LOSO (much slower but potentially better)"
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "xgboost", "random_forest"],
        help="Meta-model type for stacking (only used with --stacking)"
    )
    
    args = parser.parse_args()
    
    main(use_stacking=args.stacking, meta_model_type=args.meta_model)

