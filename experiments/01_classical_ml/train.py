"""
Training script for classical ML models with LOSO cross-validation.

Features:
- Dynamic sampling rate (never hardcoded)
- Missing data analysis per modality
- Class ratio reporting
- Full dataset export to CSV

Trains XGBoost, Random Forest, and Logistic Regression on extracted features.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import time
import json
import joblib
import logging

warnings.filterwarnings("ignore")

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_to_1hz
from shared.windowing import create_labeled_windows, parse_stress_events, compute_subject_stats
from shared.evaluation import (
    evaluate_predictions, aggregate_fold_metrics, 
    save_results, save_predictions, plot_results,
    plot_all_confusion_matrices
)
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import (
    setup_logger, log_experiment_start, log_experiment_end,
    log_data_summary, log_model_results
)
from feature_extraction import (
    BasicFeatureExtractor, FEATURE_NAMES,
    analyze_missing_data, calculate_class_ratio
)

# Import HRV extractor
try:
    from features.hrv_extractor import extract_hrv_from_window, HRV_FEATURE_NAMES
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False
    HRV_FEATURE_NAMES = []
    import logging
    logging.warning("HeartPy/HRV extractor not available - will skip PPG features")

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def process_subject(subject_folder: Path, 
                    config: Config,
                    extractor: BasicFeatureExtractor,
                    logger) -> Tuple[pd.DataFrame, str]:
    """
    Process a single subject with two-stage feature extraction.
    
    Stage 1: Align to 1Hz, create windows
    Stage 2: Extract HR/HRV from raw PPG at 64Hz (before alignment)
    Stage 3: Extract features from 1Hz aligned windows, attach HR/HRV
    """
    signals = load_raw_signals(subject_folder)
    subject_id = signals["subject_id"]
    
    try:
        start, end = get_experiment_time_range(signals)
    except ValueError as e:
        logger.warning(f"Subject {subject_id[:8]}...: Could not determine time range - {e}")
        return None, subject_id
    
    # Stage 1: Align to 1Hz for windowing
    aligned = align_to_1hz(signals, start, end)
    
    if aligned is None or len(aligned) == 0:
        logger.warning(f"Subject {subject_id[:8]}...: No aligned data")
        return None, subject_id
    
    # Compute subject-level statistics for potential subject-wise normalization
    subject_stats = compute_subject_stats(aligned)
    
    # Parse events
    event_info = parse_stress_events(
        signals.get("annotation"),
        config.stress_start_events,
        config.stress_stop_events,
        config.baseline_events
    )
    
    # Create windows with subject stats
    windows = create_labeled_windows(
        aligned,
        event_info,
        window_size_sec=config.window_size_sec,
        overlap_ratio=config.overlap_ratio,
        horizons_minutes=config.horizons_minutes,
        skip_first_minutes=config.skip_first_minutes,
        subject_stats=subject_stats  # Pass subject stats for normalization
    )
    
    if not windows:
        logger.warning(f"Subject {subject_id[:8]}...: No windows created")
        return None, subject_id
    
    # Stage 2: Extract HR/HRV from raw PPG at 64Hz (if available)
    hr_hrv_cache = {}
    hrv_stats = {
        "total_windows": len(windows),
        "hrv_attempted": 0,
        "hrv_successful": 0,
        "hrv_failed": 0,
        "hrv_partial": 0,  # Some features extracted
        "no_ppg": 0
    }
    
    if HEARTPY_AVAILABLE and signals.get("ppg") is not None:
        try:
            ppg_df = signals["ppg"]
            hrv_stats["hrv_attempted"] = len(windows)
            
            for window in windows:
                window_key = (window["window_start"], window["window_end"])
                # Use existing hrv_extractor function
                hr_hrv_features = extract_hrv_from_window(
                    ppg_df=ppg_df,
                    window_start=window["window_start"],
                    window_end=window["window_end"],
                    sample_rate=64.0
                )
                hr_hrv_cache[window_key] = hr_hrv_features
                
                # Analyze extraction quality
                nan_count = sum(1 for v in hr_hrv_features.values() 
                               if isinstance(v, float) and np.isnan(v))
                total_features = len(hr_hrv_features)
                
                if nan_count == 0:
                    hrv_stats["hrv_successful"] += 1
                elif nan_count == total_features:
                    hrv_stats["hrv_failed"] += 1
                else:
                    hrv_stats["hrv_partial"] += 1
            
            # Log detailed statistics
            success_rate = 100 * hrv_stats["hrv_successful"] / hrv_stats["total_windows"]
            failure_rate = 100 * hrv_stats["hrv_failed"] / hrv_stats["total_windows"]
            partial_rate = 100 * hrv_stats["hrv_partial"] / hrv_stats["total_windows"]
            
            logger.info(f"Subject {subject_id[:8]}... HRV Extraction:")
            logger.info(f"  ✓ Successful: {hrv_stats['hrv_successful']}/{hrv_stats['total_windows']} ({success_rate:.1f}%)")
            logger.info(f"  ✗ Failed:     {hrv_stats['hrv_failed']}/{hrv_stats['total_windows']} ({failure_rate:.1f}%)")
            if hrv_stats["hrv_partial"] > 0:
                logger.info(f"  ⚠ Partial:    {hrv_stats['hrv_partial']}/{hrv_stats['total_windows']} ({partial_rate:.1f}%)")
                
        except Exception as e:
            logger.warning(f"Subject {subject_id[:8]}...: HR/HRV extraction failed - {e}")
            hrv_stats["hrv_failed"] = len(windows)
    else:
        hrv_stats["no_ppg"] = len(windows)
        if not HEARTPY_AVAILABLE:
            logger.warning(f"Subject {subject_id[:8]}...: HeartPy not available")
        else:
            logger.warning(f"Subject {subject_id[:8]}...: No PPG data available")
    
    # Stage 3: Extract features from 1Hz aligned windows + attach HR/HRV
    rows = []
    hrv_rows = []  # Separate collection for HRV features only
    
    for window in windows:
        window_key = (window["window_start"], window["window_end"])
        hrv_features = hr_hrv_cache.get(window_key)
        
        # Extract from 1Hz aligned data
        features = extractor.extract_from_window(
            window["window_data"],
            hr_hrv_features=hrv_features
        )
        
        row = {
            "subject_id": subject_id,
            "window_id": window["window_id"],
            "window_start": window["window_start"],
            "window_end": window["window_end"],
            "context": window["context"],
            **features,
        }
        
        for horizon in config.horizons_minutes:
            row[f"label_{horizon}min"] = window.get(f"label_{horizon}min", 0)
        
        rows.append(row)
        
        # Collect HRV features separately with metadata
        hrv_row = {
            "subject_id": subject_id,
            "window_id": window["window_id"],
            "window_start": window["window_start"],
            "window_end": window["window_end"],
            "context": window["context"],
        }
        
        # Add labels
        for horizon in config.horizons_minutes:
            hrv_row[f"label_{horizon}min"] = window.get(f"label_{horizon}min", 0)
        
        # Add HRV features (or NaN if extraction failed)
        if hrv_features is not None:
            hrv_row.update(hrv_features)
        else:
            # Add NaN placeholders
            if HEARTPY_AVAILABLE:
                for key in HRV_FEATURE_NAMES:
                    hrv_row[key] = np.nan
        
        hrv_rows.append(hrv_row)
    
    duration = (end - start).total_seconds() / 60
    logger.debug(f"Subject {subject_id[:8]}...: {len(windows)} windows from {duration:.1f} min recording")
    
    # Return with HRV statistics and HRV-only dataset
    return pd.DataFrame(rows), pd.DataFrame(hrv_rows), subject_id, hrv_stats


def prepare_data(df: pd.DataFrame, 
                 label_col: str = "label_5min",
                 logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and labels from DataFrame with 2-stage quality filtering.
    
    Stage 1: Quality Filtering - Keep only windows where HRV extraction succeeded
    Stage 2: Feature-Level Handling - Impute remaining missing values with median
    
    Args:
        df: Combined dataframe with all features
        label_col: Target label column name
        logger: Logger instance (optional)
        
    Returns:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        subjects: Subject IDs (numpy array)
        feature_cols: List of feature column names
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    metadata_cols = ["subject_id", "window_id", "window_start", "window_end", 
                    "context", "label_3min", "label_5min", "label_10min"]
    
    # ===== STAGE 1: QUALITY FILTERING =====
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: Quality Filtering")
    logger.info("="*60)
    
    initial_windows = len(df)
    initial_subjects = df['subject_id'].nunique()
    
    # Keep only windows where HRV extraction succeeded (hr_bpm is present)
    if 'hr_bpm' in df.columns:
        valid_hrv_mask = df['hr_bpm'].notna()
        df_filtered = df[valid_hrv_mask].copy()
        
        filtered_windows = len(df_filtered)
        filtered_subjects = df_filtered['subject_id'].nunique()
        retention_pct = 100 * filtered_windows / initial_windows
        
        logger.info(f"  Initial windows: {initial_windows}")
        logger.info(f"  Windows with valid HRV: {filtered_windows} ({retention_pct:.1f}%)")
        logger.info(f"  Removed: {initial_windows - filtered_windows} windows ({100 - retention_pct:.1f}%)")
        logger.info(f"  Subjects retained: {filtered_subjects}/{initial_subjects}")
        
        # Check if any subject has too few windows
        min_windows_threshold = 20
        windows_per_subject = df_filtered.groupby('subject_id').size()
        insufficient = windows_per_subject[windows_per_subject < min_windows_threshold]
        
        if len(insufficient) > 0:
            logger.warning(f"  ⚠️  {len(insufficient)} subject(s) have <{min_windows_threshold} windows after filtering:")
            for subj, count in insufficient.items():
                logger.warning(f"      {subj[:12]}...: {count} windows")
        else:
            logger.info(f"  ✓ All subjects have ≥{min_windows_threshold} windows")
    else:
        logger.warning("  ⚠️  No 'hr_bpm' column found - skipping HRV quality filtering")
        df_filtered = df.copy()
    
    # ===== STAGE 2: FEATURE-LEVEL HANDLING =====
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: Feature-Level Handling")
    logger.info("="*60)
    
    # Get feature columns
    feature_cols = [col for col in df_filtered.columns if col not in metadata_cols]
    
    # Convert all feature columns to numeric
    X_df = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that are entirely NaN after conversion (can't be imputed)
    cols_to_drop = X_df.columns[X_df.isna().all()].tolist()
    if cols_to_drop:
        logger.warning(f"  ⚠️  Dropping {len(cols_to_drop)} columns that are entirely non-numeric: {cols_to_drop}")
        X_df = X_df.drop(columns=cols_to_drop)
        feature_cols = [col for col in feature_cols if col not in cols_to_drop]
    
    # Check missingness before imputation
    missing_before = {}
    hrv_features = [col for col in feature_cols if col.startswith(('hr_', 'hrv_'))]
    
    if hrv_features:
        logger.info(f"  HRV features missingness (after Stage 1 filtering):")
        for feat in hrv_features:
            n_missing = X_df[feat].isna().sum()
            pct_missing = 100 * n_missing / len(X_df)
            missing_before[feat] = pct_missing
            if pct_missing > 0:
                logger.info(f"    {feat:25s}: {n_missing:4d}/{len(X_df)} ({pct_missing:5.1f}% missing)")
    
    # Impute missing values with MEDIAN (not zero!)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)
    
    # Get the actual feature names after imputation (SimpleImputer preserves column order)
    final_feature_cols = X_df.columns.tolist()
    
    # Convert back to DataFrame for verification
    X_df_imputed = pd.DataFrame(X_imputed, columns=final_feature_cols, index=X_df.index)
    
    # Verify no missing values remain
    remaining_missing = X_df_imputed.isna().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"  ⚠️  {remaining_missing} missing values remain after imputation!")
    else:
        logger.info(f"  ✓ All missing values imputed successfully")
    
    # Final conversion
    X = X_df_imputed.values.astype(float)
    y = df_filtered[label_col].values
    subjects = df_filtered["subject_id"].values
    
    # Final safety check - replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"\n  Final dataset shape: {X.shape}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Windows: {X.shape[0]}")
    logger.info(f"  Subjects: {len(np.unique(subjects))}")
    logger.info("="*60 + "\n")
    
    return X, y, subjects, final_feature_cols


def get_hyperparameter_grid(model_name: str) -> Dict:
    """
    Get hyperparameter grid for each model based on best practices.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of hyperparameter lists to try
    """
    if model_name == "logistic_regression":
        return {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
            'penalty': ['l1', 'l2'],                      # Regularization type
            'solver': ['liblinear', 'saga'],              # Optimizers that support both L1/L2
            'max_iter': [1000]                            # Fixed for convergence
        }
    
    elif model_name == "random_forest":
        return {
            'n_estimators': [50, 100, 200],               # Number of trees
            'max_depth': [5, 10, 15, 20, None],           # Tree depth
            'min_samples_split': [2, 5, 10],              # Min samples to split node
            'min_samples_leaf': [1, 2, 4],                # Min samples at leaf
            'max_features': ['sqrt', 'log2', None],       # Features per split
            'bootstrap': [True],                          # Always use bootstrap
        }
    
    elif model_name == "svm":
        return {
            'C': [0.1, 1.0, 10.0, 100.0],                 # Regularization
            'kernel': ['rbf', 'poly', 'sigmoid'],         # Kernel types
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], # Kernel coefficient
            'degree': [2, 3, 4],                          # For poly kernel only
            'probability': [True]                         # Always need probabilities
        }
    
    elif model_name == "xgboost":
        return {
            'n_estimators': [50, 100, 200],               # Number of boosting rounds
            'max_depth': [3, 5, 7, 9],                    # Tree depth
            'learning_rate': [0.01, 0.05, 0.1, 0.3],      # Step size shrinkage
            'subsample': [0.7, 0.8, 1.0],                 # Sample ratio of training
            'colsample_bytree': [0.7, 0.8, 1.0],          # Feature sampling
            'min_child_weight': [1, 3, 5],                # Min sum of weights in child
            'gamma': [0, 0.1, 0.2],                       # Min loss reduction for split
        }
    
    else:
        return {}


def inner_cv_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    model_class,
    model_name: str,
    base_model_kwargs: Dict,
    param_grid: Dict,
    n_inner_folds: int = 5,
    metric: str = 'gmean',
    logger = None
) -> Dict:
    """
    Perform inner cross-validation for hyperparameter tuning.
    
    Uses subject-wise K-fold CV within the training data (nested CV).
    
    Args:
        X_train: Training features
        y_train: Training labels
        subjects_train: Training subject IDs
        model_class: Model class to instantiate
        model_name: Name of the model
        base_model_kwargs: Base hyperparameters (random_seed, etc.)
        param_grid: Grid of hyperparameters to search
        n_inner_folds: Number of inner CV folds (subject-wise)
        metric: Metric to optimize ('gmean', 'recall', 'f1')
        logger: Logger instance
        
    Returns:
        Best hyperparameters dictionary
    """
    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import StandardScaler
    
    if logger:
        logger.info(f"    Inner CV: Tuning {len(list(ParameterGrid(param_grid)))} configurations...")
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    
    # Get unique subjects for inner fold splitting
    unique_subjects = np.unique(subjects_train)
    n_subjects = len(unique_subjects)
    
    # If too few subjects, reduce inner folds
    if n_subjects < n_inner_folds:
        n_inner_folds = max(2, n_subjects // 2)
        if logger:
            logger.warning(f"    Reduced inner folds to {n_inner_folds} (only {n_subjects} subjects)")
    
    # Shuffle subjects for K-fold
    np.random.seed(42)
    shuffled_subjects = np.random.permutation(unique_subjects)
    fold_size = len(shuffled_subjects) // n_inner_folds
    
    best_score = -np.inf
    best_params = None
    
    # Try each hyperparameter combination
    for params in param_combinations:
        fold_scores = []
        
        # Inner K-fold CV (subject-wise)
        for fold_idx in range(n_inner_folds):
            # Define validation subjects for this inner fold
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < n_inner_folds - 1 else len(shuffled_subjects)
            val_subjects = shuffled_subjects[val_start:val_end]
            
            # Split into inner train and validation
            inner_val_mask = np.isin(subjects_train, val_subjects)
            inner_train_mask = ~inner_val_mask
            
            X_inner_train = X_train[inner_train_mask]
            y_inner_train = y_train[inner_train_mask]
            X_inner_val = X_train[inner_val_mask]
            y_inner_val = y_train[inner_val_mask]
            
            if len(X_inner_val) == 0 or len(np.unique(y_inner_val)) < 2:
                continue
            
            # Combine base kwargs with current params
            model_kwargs = {**base_model_kwargs, **params}
            
            # Add class weights
            n_neg = np.sum(y_inner_train == 0)
            n_pos = np.sum(y_inner_train == 1)
            
            if "XGB" in str(model_class):
                model_kwargs["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
            elif "class_weight" in str(model_class.__init__.__code__.co_varnames):
                model_kwargs["class_weight"] = "balanced"
            
            # Handle NaN
            X_inner_train = np.nan_to_num(X_inner_train, nan=0.0)
            X_inner_val = np.nan_to_num(X_inner_val, nan=0.0)
            
            # Scale
            scaler = StandardScaler()
            X_inner_train_scaled = scaler.fit_transform(X_inner_train)
            X_inner_val_scaled = scaler.transform(X_inner_val)
            
            # Train
            try:
                model = model_class(**model_kwargs)
                model.fit(X_inner_train_scaled, y_inner_train)
                
                # Predict probabilities
                if hasattr(model, 'predict_proba'):
                    y_val_proba = model.predict_proba(X_inner_val_scaled)[:, 1]
                else:
                    y_val_proba = model.decision_function(X_inner_val_scaled)
                
                # Use fixed threshold for hyperparameter selection
                y_val_pred = (y_val_proba >= 0.5).astype(int)
                
                # Calculate metric manually (simple G-mean)
                from sklearn.metrics import recall_score
                tn = np.sum((y_inner_val == 0) & (y_val_pred == 0))
                fp = np.sum((y_inner_val == 0) & (y_val_pred == 1))
                fn = np.sum((y_inner_val == 1) & (y_val_pred == 0))
                tp = np.sum((y_inner_val == 1) & (y_val_pred == 1))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                if metric == 'gmean':
                    score = np.sqrt(sensitivity * specificity)
                elif metric == 'recall':
                    score = sensitivity
                elif metric == 'f1':
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
                else:
                    score = np.sqrt(sensitivity * specificity)
                
                fold_scores.append(score)
                
            except Exception as e:
                # Skip this param combination if it fails
                continue
        
        # Average score across inner folds
        if len(fold_scores) > 0:
            avg_score = np.mean(fold_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
    
    if best_params is None:
        # Fallback to first combination if all failed
        best_params = param_combinations[0]
        if logger:
            logger.warning(f"    All hyperparameter combinations failed, using default: {best_params}")
    else:
        if logger:
            logger.info(f"    Best params: {best_params} (score: {best_score:.4f})")
    
    return best_params


def loso_cross_validation(X: np.ndarray,
                          y: np.ndarray,
                          subjects: np.ndarray,
                          feature_names: List[str],
                          model_class,
                          model_name: str,
                          model_kwargs: Dict,
                          logger,
                          enable_tuning: bool = True,
                          param_grid: Dict = None,
                          threshold_method: str = "geometric_mean",
                          min_recall: float = 0.0,
                          max_fpr: float = 1.0) -> Dict:
    """
    Perform Leave-One-Subject-Out cross-validation with optional hyperparameter tuning.
    
    Algorithm:
    1. For each test subject (OUTER LOOP):
        2. Split data: test subject vs. all other subjects
        3. If tuning enabled: Perform inner CV to find best hyperparameters
        4. Train final model with best hyperparameters on all training subjects
        5. Select optimal threshold on training data
        6. Evaluate on test subject
    7. Aggregate results across all folds
    
    Args:
        enable_tuning: If True, perform nested CV for hyperparameter tuning
        param_grid: Hyperparameter grid (if None, use get_hyperparameter_grid)
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    if model_kwargs is None:
        model_kwargs = {}
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    logger.info(f"LOSO CV with {n_subjects} folds (subjects)")
    
    # Get hyperparameter grid if tuning is enabled
    if enable_tuning:
        if param_grid is None:
            param_grid = get_hyperparameter_grid(model_name)
        
        if param_grid:
            logger.info(f"Hyperparameter tuning: ENABLED")
            logger.info(f"  Param grid size: {len(list(ParameterGrid(param_grid)))} combinations")
        else:
            enable_tuning = False
            logger.info(f"Hyperparameter tuning: DISABLED (no grid available)")
    else:
        logger.info(f"Hyperparameter tuning: DISABLED")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    fold_best_params = []  # Track best params per fold
    
    start_time = time.time()
    
    # Progress bar for LOSO
    pbar = tqdm(enumerate(unique_subjects), total=n_subjects, 
                desc=f"  {model_name}", unit="fold",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for fold_idx, test_subject in pbar:
        # ========== STEP 1: SUBJECT-WISE SPLIT ==========
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        subjects_train = subjects[train_mask]
        
        if len(X_test) == 0:
            continue
        
        # ========== STEP 2: HYPERPARAMETER TUNING (INNER CV) ==========
        if enable_tuning and param_grid:
            best_params = inner_cv_hyperparameter_tuning(
                X_train, y_train, subjects_train,
                model_class, model_name, model_kwargs, param_grid,
                n_inner_folds=5, metric='gmean', logger=logger
            )
            fold_best_params.append(best_params)
            
            # Merge best params with base kwargs
            model_kwargs_fold = {**model_kwargs, **best_params}
        else:
            model_kwargs_fold = model_kwargs.copy()
            fold_best_params.append({})
        
        # ========== STEP 3: TRAIN FINAL MODEL ==========
        # Add class weights
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        
        if "XGB" in str(model_class):
            model_kwargs_fold["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
        elif "class_weight" in str(model_class.__init__.__code__.co_varnames):
            model_kwargs_fold["class_weight"] = "balanced"
        
        # Handle any remaining NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train with best hyperparameters
        try:
            model = model_class(**model_kwargs_fold)
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            logger.warning(f"Fold {fold_idx+1}: Training failed for {test_subject[:8]}... - {e}")
            continue
        
        # ========== STEP 4: THRESHOLD SELECTION (TRAIN DATA ONLY) ==========
        # Get probabilities on TRAINING data for threshold selection
        try:
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        except Exception:
            y_train_proba = model.predict(X_train_scaled).astype(float)
        
        # Find optimal threshold on TRAINING data
        from shared.evaluation import find_optimal_threshold
        optimal_threshold, _ = find_optimal_threshold(
            y_train, y_train_proba,
            method=threshold_method,
            min_recall=min_recall,
            max_fpr=max_fpr
        )
        
        # ========== STEP 5: TEST ON HELD-OUT SUBJECT ==========
        # Predict on TEST data
        try:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        except Exception:
            y_proba = model.predict(X_test_scaled).astype(float)
        
        # Apply optimal threshold from training
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        # Store
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_test))
        
        # Fold metrics with optimal threshold
        fold_metric = evaluate_predictions(y_test, y_pred, y_proba, model_name, threshold=optimal_threshold)
        fold_metric["subject"] = test_subject
        fold_metric["optimal_threshold"] = optimal_threshold
        fold_metrics.append(fold_metric)
        
        # Update progress bar with key metrics
        fold_gmean = fold_metric.get("gmean", float("nan"))
        fold_recall = fold_metric.get("recall", float("nan"))
        pbar.set_postfix({
            "Gmean": f"{fold_gmean:.3f}", 
            "Recall": f"{fold_recall:.3f}",
            "Test": len(y_test)
        })
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/n_subjects:.1f}s per fold)")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # Check if any folds succeeded
    if len(fold_metrics) == 0:
        logger.error(f"All folds failed for {model_name}")
        return {
            "metrics": {"auroc": float("nan"), "pr_auc": float("nan"), 
                       "f1": float("nan"), "accuracy": float("nan"),
                       "precision": float("nan"), "recall": float("nan"),
                       "specificity": float("nan"), "gmean": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "y_proba": np.array([]),
            "subjects": np.array([]),
            "fold_metrics": []
        }
    
    # Aggregate metrics using fold-level predictions (no re-thresholding)
    # This ensures: aggregate ≈ average(fold_metrics)
    aggregate_metrics = evaluate_predictions(
        all_y_true, 
        all_y_pred,  # Predictions made with fold-specific thresholds
        all_y_proba, 
        model_name
        # NO threshold - using pre-computed predictions!
    )
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    # Report threshold statistics (each fold used different threshold)
    fold_thresholds = [f.get("optimal_threshold", 0.5) for f in fold_metrics]
    if len(fold_thresholds) > 0:
        aggregate_metrics["threshold_mean"] = float(np.mean(fold_thresholds))
        aggregate_metrics["threshold_std"] = float(np.std(fold_thresholds))
        aggregate_metrics["threshold_min"] = float(np.min(fold_thresholds))
        aggregate_metrics["threshold_max"] = float(np.max(fold_thresholds))
    else:
        aggregate_metrics["threshold_mean"] = 0.5
        aggregate_metrics["threshold_std"] = 0.0
        aggregate_metrics["threshold_min"] = 0.5
        aggregate_metrics["threshold_max"] = 0.5
    
    # Log results
    log_model_results(logger, model_name, aggregate_metrics)
    
    # Log hyperparameter summary if tuning was enabled
    if enable_tuning and fold_best_params:
        logger.info(f"\n{'='*60}")
        logger.info(f"HYPERPARAMETER TUNING SUMMARY")
        logger.info(f"{'='*60}")
        
        # Count most common parameter values
        from collections import Counter
        param_keys = set()
        for params in fold_best_params:
            param_keys.update(params.keys())
        
        for key in sorted(param_keys):
            values = [params.get(key) for params in fold_best_params if key in params]
            if values:
                counter = Counter(values)
                most_common = counter.most_common(3)
                logger.info(f"  {key}:")
                for val, count in most_common:
                    logger.info(f"    {val}: {count}/{len(fold_best_params)} folds")
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": all_subjects,
        "fold_metrics": fold_metrics,
        "best_params_per_fold": fold_best_params  # NEW: track best params
    }


def log_missing_analysis(logger, analysis: Dict):
    """Log missing data analysis results."""
    logger.info("\n" + "-"*50)
    logger.info("MISSING DATA ANALYSIS BY MODALITY")
    logger.info("-"*50)
    
    for modality, info in analysis.items():
        status_icon = "✓" if info["status"] == "OK" else "⚠"
        logger.info(f"  {status_icon} {modality.upper()}")
        logger.info(f"      Features: {info['n_features']}")
        logger.info(f"      Missing rate: {info['missing_rate']*100:.1f}%")
        if "rows_with_missing" in info:
            logger.info(f"      Rows with missing: {info['rows_with_missing']} ({info['pct_rows_missing']:.1f}%)")


def log_class_distribution(logger, class_stats: Dict):
    """Log class distribution."""
    logger.info("\n" + "-"*50)
    logger.info("CLASS DISTRIBUTION")
    logger.info("-"*50)
    logger.info(f"  Total samples: {class_stats['n_total']}")
    logger.info(f"  Emotional Stress (1): {class_stats['n_emotional_stress']} ({class_stats['pct_emotional_stress']:.1f}%)")
    logger.info(f"  No Stress/Physical (0): {class_stats['n_no_stress_or_physical']} ({class_stats['pct_no_stress']:.1f}%)")
    logger.info(f"  Class ratio: {class_stats['class_ratio']}")
    
    # Warning if highly imbalanced
    if class_stats['imbalance_ratio'] > 5:
        logger.warning(f"  ⚠ HIGHLY IMBALANCED - consider using class weights or resampling")


def train_and_save_final_model(X: np.ndarray,
                                y: np.ndarray,
                                feature_names: List[str],
                                model_class,
                                model_name: str,
                                model_kwargs: Dict,
                                output_dir: Path,
                                logger) -> None:
    """
    Train a final model on ALL data and save weights/parameters.
    
    Saves:
    - model weights (.joblib)
    - scaler (.joblib)
    - model parameters (.json)
    - feature names (.json)
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        model_class: Model class to instantiate
        model_name: Name for saving
        model_kwargs: Model hyperparameters
        output_dir: Directory to save models
        logger: Logger instance
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare model kwargs with class weights
    model_kwargs_copy = model_kwargs.copy()
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    
    if "XGB" in str(model_class):
        model_kwargs_copy["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
    elif "class_weight" in str(model_class.__init__.__code__.co_varnames):
        model_kwargs_copy["class_weight"] = "balanced"
    
    # Handle any remaining NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = model_class(**model_kwargs_copy)
    model.fit(X_scaled, y)
    
    # Save model
    model_path = models_dir / f"{model_name}_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to: {model_path}")
    
    # Save scaler
    scaler_path = models_dir / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to: {scaler_path}")
    
    # Save model parameters
    params_to_save = {
        "model_class": str(model_class.__name__),
        "hyperparameters": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                           for k, v in model_kwargs_copy.items()},
        "n_features": X.shape[1],
        "n_samples_trained": X.shape[0],
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "feature_names": feature_names,
    }
    
    params_path = models_dir / f"{model_name}_params.json"
    with open(params_path, "w") as f:
        json.dump(params_to_save, f, indent=2)
    logger.info(f"Saved parameters to: {params_path}")


def main():
    """Main training pipeline."""
    
    # Setup
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        "classical_ml",
        log_file=results_dir / "training.log"
    )
    
    log_experiment_start(logger, "CLASSICAL ML BASELINE TRAINING")
    
    # Initialize feature extractor (sampling rate calculated dynamically)
    extractor = BasicFeatureExtractor()
    logger.info(f"Feature extractor initialized")
    logger.info(f"Expected features: {len(FEATURE_NAMES)}")
    logger.info(f"Sampling rate: Calculated dynamically from data (never hardcoded)")
    
    # Check HeartPy
    if HEARTPY_AVAILABLE:
        logger.info("HeartPy: Available - will extract HR/HRV from raw PPG ✓")
    else:
        logger.warning("HeartPy: Not available - will skip PPG-based HR/HRV features")
    
    # Check XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("XGBoost: Available ✓")
    else:
        logger.warning("XGBoost: Not available - will skip XGBoost training")
    
    # Get all subjects
    subjects = get_all_subjects(config.data_path)
    logger.info(f"Found {len(subjects)} subject folders")
    
    # Process all subjects
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Loading and preprocessing data")
    logger.info("-"*50)
    
    all_dfs = []
    all_hrv_dfs = []  # Collect HRV-only datasets
    successful = 0
    failed = 0
    
    # Track HRV extraction statistics across all subjects
    overall_hrv_stats = {
        "total_windows": 0,
        "hrv_attempted": 0,
        "hrv_successful": 0,
        "hrv_failed": 0,
        "hrv_partial": 0,
        "no_ppg": 0
    }
    
    pbar = tqdm(subjects, desc="Loading subjects", unit="subject",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for subject_folder in pbar:
        result = process_subject(subject_folder, config, extractor, logger)
        
        if result is None:
            failed += 1
            pbar.set_postfix({"OK": successful, "Fail": failed})
            continue
            
        df, hrv_df, subject_id, hrv_stats = result
        
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            all_hrv_dfs.append(hrv_df)  # Collect HRV data
            successful += 1
            
            # Accumulate HRV statistics
            for key in overall_hrv_stats:
                overall_hrv_stats[key] += hrv_stats.get(key, 0)
            
            pbar.set_postfix({"OK": successful, "Fail": failed, "Windows": len(df)})
        else:
            failed += 1
            pbar.set_postfix({"OK": successful, "Fail": failed})
    
    pbar.close()
    
    logger.info(f"Loaded: {successful}/{len(subjects)} subjects ({failed} failed)")
    
    # Log overall HRV extraction statistics
    if overall_hrv_stats["total_windows"] > 0:
        logger.info("\n" + "="*60)
        logger.info("OVERALL HRV EXTRACTION STATISTICS")
        logger.info("="*60)
        
        total = overall_hrv_stats["total_windows"]
        success_pct = 100 * overall_hrv_stats["hrv_successful"] / total
        fail_pct = 100 * overall_hrv_stats["hrv_failed"] / total
        partial_pct = 100 * overall_hrv_stats["hrv_partial"] / total
        no_ppg_pct = 100 * overall_hrv_stats["no_ppg"] / total
        
        logger.info(f"Total windows:          {total}")
        logger.info(f"HRV attempted:          {overall_hrv_stats['hrv_attempted']} ({100*overall_hrv_stats['hrv_attempted']/total:.1f}%)")
        logger.info(f"  ✓ Successful:         {overall_hrv_stats['hrv_successful']} ({success_pct:.1f}%)")
        logger.info(f"  ✗ Failed:             {overall_hrv_stats['hrv_failed']} ({fail_pct:.1f}%)")
        if overall_hrv_stats["hrv_partial"] > 0:
            logger.info(f"  ⚠ Partial:            {overall_hrv_stats['hrv_partial']} ({partial_pct:.1f}%)")
        if overall_hrv_stats["no_ppg"] > 0:
            logger.info(f"No PPG available:       {overall_hrv_stats['no_ppg']} ({no_ppg_pct:.1f}%)")
        
        # Calculate usable HRV rate (successful + partial)
        usable = overall_hrv_stats["hrv_successful"] + overall_hrv_stats["hrv_partial"]
        usable_pct = 100 * usable / total
        logger.info(f"\nUsable HRV data:        {usable}/{total} ({usable_pct:.1f}%)")
        logger.info(f"Missing HRV data:       {total - usable}/{total} ({100 - usable_pct:.1f}%)")
        logger.info("="*60 + "\n")
    
    if not all_dfs:
        logger.error("No data loaded! Check data path and file formats.")
        return
    
    # Combine all subjects
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # =========================================================================
    # MISSING DATA ANALYSIS
    # =========================================================================
    
    feature_cols = [col for col in combined_df.columns 
                   if col not in ["subject_id", "window_id", "window_start", 
                                  "window_end", "context", "label_3min", 
                                  "label_5min", "label_10min"]]
    
    missing_analysis = analyze_missing_data(combined_df, feature_cols)
    log_missing_analysis(logger, missing_analysis)
    
    # Save missing analysis to JSON
    missing_path = results_dir / "missing_analysis.json"
    with open(missing_path, "w") as f:
        # Convert to serializable format
        serializable = {}
        for mod, info in missing_analysis.items():
            serializable[mod] = {
                "n_features": info["n_features"],
                "missing_rate": float(info["missing_rate"]),
                "status": info["status"],
                "rows_with_missing": int(info.get("rows_with_missing", 0)),
                "pct_rows_missing": float(info.get("pct_rows_missing", 0))
            }
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved missing analysis to: {missing_path}")
    
    # =========================================================================
    # CLASS DISTRIBUTION
    # =========================================================================
    
    class_stats = calculate_class_ratio(combined_df[config.target_label].values)
    log_class_distribution(logger, class_stats)
    
    # Save class distribution
    class_dist_path = results_dir / "class_distribution.json"
    with open(class_dist_path, "w") as f:
        json.dump({k: v if not isinstance(v, np.integer) else int(v) 
                   for k, v in class_stats.items()}, f, indent=2)
    logger.info(f"Saved class distribution to: {class_dist_path}")
    
    # =========================================================================
    # SAVE FULL DATASET
    # =========================================================================
    
    features_path = results_dir / "features_dataset.csv"
    combined_df.to_csv(features_path, index=False)
    logger.info(f"\nSaved full dataset to: {features_path}")
    logger.info(f"  Shape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
    logger.info(f"  Subjects: {combined_df['subject_id'].nunique()}")
    logger.info(f"  Features: {len(feature_cols)}")
    
    # Save HRV-only dataset
    if all_hrv_dfs:
        combined_hrv_df = pd.concat(all_hrv_dfs, ignore_index=True)
        hrv_path = results_dir / "hrv_features_dataset.csv"
        combined_hrv_df.to_csv(hrv_path, index=False)
        
        # Calculate statistics (exclude dropped features)
        hrv_feature_cols = [col for col in combined_hrv_df.columns 
                           if col.startswith(('hr_', 'hrv_')) and 
                           col not in ['hr_peak_rejection_rate', 'hrv_lf', 'hrv_hf', 
                                      'hrv_lf_hf_ratio'] and 
                           col != 'breathing_rate']
        
        logger.info(f"\nSaved HRV-only dataset to: {hrv_path}")
        logger.info(f"  Shape: {combined_hrv_df.shape[0]} rows × {combined_hrv_df.shape[1]} columns")
        logger.info(f"  Subjects: {combined_hrv_df['subject_id'].nunique()}")
        logger.info(f"  HRV features: {len(hrv_feature_cols)}")
        
        # Report missing data per feature
        logger.info(f"\n  HRV Feature Completeness:")
        for col in hrv_feature_cols:
            valid_count = combined_hrv_df[col].notna().sum()
            valid_pct = 100 * valid_count / len(combined_hrv_df)
            logger.info(f"    {col:25s}: {valid_count:4d}/{len(combined_hrv_df)} ({valid_pct:5.1f}%)")
    
    # Prepare data with 2-stage quality filtering
    X, y, subjects_arr, feature_names = prepare_data(combined_df, config.target_label, logger)
    
    # Log data summary
    log_data_summary(
        logger,
        n_subjects=combined_df["subject_id"].nunique(),
        n_windows=len(combined_df),
        n_features=X.shape[1],
        n_positive=int(np.sum(y == 1)),
        n_negative=int(np.sum(y == 0))
    )
    
    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Model Training (LOSO CV)")
    logger.info("-"*50)
    
    models = {
        "logistic_regression": (
            LogisticRegression,
            {"max_iter": 1000, "random_state": config.random_seed}
        ),
        "random_forest": (
            RandomForestClassifier,
            {"n_estimators": 100, "max_depth": 10, "random_state": config.random_seed, "n_jobs": -1}
        ),
        "svm": (
            SVC,
            {"kernel": "rbf", "probability": True, "random_state": config.random_seed, "C": 1.0}
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models["xgboost"] = (
            xgb.XGBClassifier,
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
             "random_state": config.random_seed, "eval_metric": "logloss", 
             "n_jobs": -1, "verbosity": 0}
        )
    
    logger.info(f"Models to train: {list(models.keys())}")
    
    # Train each model
    all_results = {}
    
    for model_key, (model_class, model_kwargs) in models.items():
        # Get hyperparameter grid for tuning
        param_grid = get_hyperparameter_grid(model_key)
        
        result = loso_cross_validation(
            X, y, subjects_arr, feature_names,
            model_class, model_key, model_kwargs, logger,
            enable_tuning=True,  # Enable hyperparameter tuning
            param_grid=param_grid
        )
        
        all_results[model_key] = result
        
        # Save results
        save_results(result["metrics"], results_dir, model_key)
        save_predictions(
            result["y_true"], result["y_pred"], result["y_proba"],
            result["subjects"], results_dir, model_key
        )
        
        if len(np.unique(result["y_true"])) > 1:
            plot_results(result["y_true"], result["y_proba"], model_key, results_dir,
                        y_pred=result["y_pred"])
        
        fold_df = pd.DataFrame(result["fold_metrics"])
        fold_df.to_csv(results_dir / f"{model_key}_fold_metrics.csv", index=False)
        
        # Train and save final model on ALL data
        train_and_save_final_model(
            X, y, feature_names,
            model_class, model_key, model_kwargs,
            results_dir, logger
        )
        
        logger.info(f"Saved {model_key} results to {results_dir}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("="*70)
    
    comparison_data = []
    for model_key, result in all_results.items():
        metrics = result["metrics"]
        comparison_data.append({
            "Model": model_key,
            "AUROC": metrics.get("auroc", float("nan")),
            "PR-AUC": metrics.get("pr_auc", float("nan")),
            "F1": metrics.get("f1", float("nan")),
            "Accuracy": metrics.get("accuracy", float("nan"))
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("AUROC", ascending=False)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Best model
    best_model = comparison_df.iloc[0]["Model"]
    best_auroc = comparison_df.iloc[0]["AUROC"]
    logger.info(f"\nBest model: {best_model} (AUROC: {best_auroc:.4f})")
    
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
    
    # Generate combined confusion matrix plot
    plot_all_confusion_matrices(all_results, results_dir)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("OUTPUT FILES")
    logger.info("="*70)
    logger.info(f"  Dataset: features_dataset.csv ({combined_df.shape[0]} samples)")
    logger.info(f"  Missing analysis: missing_analysis.json")
    logger.info(f"  Class distribution: class_distribution.json")
    logger.info(f"  Model comparison: model_comparison.csv")
    logger.info(f"  Per-model results: <model>_metrics.json, <model>_predictions.csv")
    
    log_experiment_end(logger, "CLASSICAL ML BASELINE TRAINING")
    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
