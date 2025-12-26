"""
TabPFN Foundation Model for Stress Prediction

Uses TabPFN-2.5 pre-trained model for tabular classification.
Similar to classical ML but uses a foundation model instead of traditional classifiers.

TabPFN advantages:
- Pre-trained on synthetic tabular data
- No hyperparameter tuning needed
- Fast inference
- Handles missing values automatically
- Works well on small datasets

Reference: https://github.com/PriorLabs/TabPFN
Paper: https://arxiv.org/abs/2207.01848
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
    plot_all_confusion_matrices, find_optimal_threshold
)
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import (
    setup_logger, log_experiment_start, log_experiment_end,
    log_data_summary, log_model_results
)
from feature_extractor import (
    BasicFeatureExtractor, FEATURE_NAMES,
    analyze_missing_data, calculate_class_ratio)
# from hrv_extractor import extract_hrv_from_window, HRV_FEATURE_NAMES
# Import feature extraction from classical ML experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "01_classical_ml"))



# Import HRV extractor
try:
    from hrv_extractor import extract_hrv_from_window, HRV_FEATURE_NAMES
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False
    HRV_FEATURE_NAMES = []
    logging.warning("HeartPy/HRV extractor not available - will skip PPG features")

# Import TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.error("TabPFN not available! Install with: pip install tabpfn")


def process_subject(subject_folder: Path, 
                    config: Config,
                    extractor: BasicFeatureExtractor,
                    logger) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict]:
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
        return None, None, subject_id, {}
    
    # Stage 1: Align to 1Hz for windowing
    aligned = align_to_1hz(signals, start, end)
    
    if aligned is None or len(aligned) == 0:
        logger.warning(f"Subject {subject_id[:8]}...: No aligned data")
        return None, None, subject_id, {}
    
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
        return None, None, subject_id, {}
    
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
    
    NOTE: TabPFN can handle missing values, but we apply same filtering as classical ML
    for fair comparison.
    
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


def loso_cross_validation_tabpfn(X: np.ndarray,
                                   y: np.ndarray,
                                   subjects: np.ndarray,
                                   feature_names: List[str],
                                   logger,
                                   device: str = "cpu",
                                   threshold_method: str = "geometric_mean",
                                   min_recall: float = 0.0,
                                   max_fpr: float = 1.0) -> Dict:
    """
    Perform Leave-One-Subject-Out cross-validation with TabPFN.
    
    Key differences from classical ML:
    1. No hyperparameter tuning (TabPFN is pre-trained)
    2. No manual scaling (TabPFN handles this)
    3. No class weights (TabPFN handles imbalance)
    4. Threshold selection still on training data
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        subjects: Subject IDs (n_samples,)
        feature_names: List of feature names
        logger: Logger instance
        device: "cuda" or "cpu" (default: "cpu")
        threshold_method: Method for threshold selection
        min_recall: Minimum recall constraint
        max_fpr: Maximum FPR constraint
    
    Returns:
        Dictionary with metrics and predictions
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: TabPFN Foundation Model")
    logger.info(f"{'='*60}")
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    logger.info(f"LOSO CV with {n_subjects} folds (subjects)")
    logger.info(f"Device: {device}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Total samples: {X.shape[0]}")
    
    # Check TabPFN limitations
    if X.shape[1] > 100:
        logger.warning(f"⚠️  TabPFN works best with <100 features. You have {X.shape[1]}.")
        logger.warning("   Consider feature selection for better performance.")
    
    if X.shape[0] > 50000:
        logger.warning(f"⚠️  TabPFN works best with <50K samples. You have {X.shape[0]}.")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    
    start_time = time.time()
    
    # Progress bar for LOSO
    pbar = tqdm(enumerate(unique_subjects), total=n_subjects, 
                desc=f"  TabPFN", unit="fold",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for fold_idx, test_subject in pbar:
        # ========== STEP 1: SUBJECT-WISE SPLIT ==========
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(X_test) == 0:
            continue
        
        # Check dataset size for TabPFN
        if len(X_train) > 50000:
            logger.warning(f"Fold {fold_idx+1}: Training set too large ({len(X_train)}). TabPFN works best with <50K samples.")
        
        # ========== STEP 2: TRAIN TabPFN ==========
        try:
            # Initialize TabPFN (no hyperparameters needed!)
            model = TabPFNClassifier(
                device=device,
                n_estimators=8  # Ensemble size (default: 8)
            )
            
            # Train (fit)
            model.fit(X_train, y_train)
            
        except Exception as e:
            logger.warning(f"Fold {fold_idx+1}: Training failed for {test_subject[:8]}... - {e}")
            continue
        
        # ========== STEP 3: THRESHOLD SELECTION (TRAIN DATA) ==========
        # Get probabilities on TRAINING data for threshold selection
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
        except Exception as e:
            logger.warning(f"Fold {fold_idx+1}: Prediction failed - {e}")
            continue
        
        # Find optimal threshold on TRAINING data
        optimal_threshold, _ = find_optimal_threshold(
            y_train, y_train_proba,
            method=threshold_method,
            min_recall=min_recall,
            max_fpr=max_fpr
        )
        
        # ========== STEP 4: TEST ON HELD-OUT SUBJECT ==========
        # Predict on TEST data
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.warning(f"Fold {fold_idx+1}: Test prediction failed - {e}")
            continue
        
        # Apply optimal threshold from training
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        # Store
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_test))
        
        # Fold metrics
        fold_metric = evaluate_predictions(y_test, y_pred, y_proba, "tabpfn", threshold=optimal_threshold)
        fold_metric["subject"] = test_subject
        fold_metric["optimal_threshold"] = optimal_threshold
        fold_metrics.append(fold_metric)
        
        # Update progress bar
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
        logger.error(f"All folds failed for TabPFN")
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
    aggregate_metrics = evaluate_predictions(
        all_y_true, 
        all_y_pred,  # Predictions made with fold-specific thresholds
        all_y_proba, 
        "tabpfn"
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
    log_model_results(logger, "tabpfn", aggregate_metrics)
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": all_subjects,
        "fold_metrics": fold_metrics
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
        logger.warning(f"  ⚠ HIGHLY IMBALANCED - TabPFN handles this automatically")


def main():
    """Main training pipeline."""
    
    # Check TabPFN availability
    if not TABPFN_AVAILABLE:
        print("="*70)
        print("ERROR: TabPFN not installed!")
        print("="*70)
        print("\nInstallation steps:")
        print("  1. Install TabPFN: pip install tabpfn")
        print("  2. Login to HuggingFace: huggingface-cli login")
        print("  3. Accept license at: https://huggingface.co/Prior-Labs/tabpfn_2_5")
        print("="*70)
        return
    
    # Setup
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        "tabpfn",
        log_file=results_dir / "training.log"
    )
    
    log_experiment_start(logger, "TabPFN FOUNDATION MODEL TRAINING")
    
    # Check HuggingFace authentication and model access
    logger.info("Checking TabPFN requirements...")
    try:
        # Test if model can be loaded (will auto-download on first use)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_model = TabPFNClassifier(device=device)
        logger.info("✓ TabPFN model accessible")
        logger.info(f"✓ Using device: {device}")
        if device == "cpu":
            logger.warning("⚠️  Running on CPU - this will be slower!")
            logger.warning("   Consider running on GPU for faster training")
    except Exception as e:
        logger.error(f"✗ TabPFN model not accessible: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Run: huggingface-cli login")
        logger.error("  2. Accept license at: https://huggingface.co/Prior-Labs/tabpfn_2_5")
        logger.error("  3. Ensure you have internet connection for first download (~500MB)")
        return
    
    # Initialize feature extractor (same as classical ML)
    extractor = BasicFeatureExtractor()
    logger.info(f"Feature extractor initialized")
    logger.info(f"Expected features: {len(FEATURE_NAMES)}")
    logger.info(f"Sampling rate: Calculated dynamically from data (never hardcoded)")
    
    # Check HeartPy
    if HEARTPY_AVAILABLE:
        logger.info("HeartPy: Available - will extract HR/HRV from raw PPG ✓")
    else:
        logger.warning("HeartPy: Not available - will skip PPG-based HR/HRV features")
    
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
        
        if result[0] is None:
            failed += 1
            pbar.set_postfix({"OK": successful, "Fail": failed})
            continue
            
        df, hrv_df, subject_id, hrv_stats = result
        
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            all_hrv_dfs.append(hrv_df)
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
    # PHASE 2: TabPFN TRAINING
    # =========================================================================
    
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: TabPFN Training (LOSO CV)")
    logger.info("-"*50)
    
    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Train TabPFN with LOSO
    result = loso_cross_validation_tabpfn(
        X, y, subjects_arr, feature_names,
        logger,
        device=device,
        threshold_method="geometric_mean",
        min_recall=0.0,
        max_fpr=1.0
    )
    
    # Save results
    save_results(result["metrics"], results_dir, "tabpfn")
    save_predictions(
        result["y_true"], result["y_pred"], result["y_proba"],
        result["subjects"], results_dir, "tabpfn"
    )
    
    if len(np.unique(result["y_true"])) > 1:
        plot_results(result["y_true"], result["y_proba"], "tabpfn", results_dir,
                    y_pred=result["y_pred"])
    
    fold_df = pd.DataFrame(result["fold_metrics"])
    fold_df.to_csv(results_dir / "tabpfn_fold_metrics.csv", index=False)
    
    logger.info(f"Saved TabPFN results to {results_dir}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("TabPFN TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"  AUROC:       {result['metrics'].get('auroc', float('nan')):.4f}")
    logger.info(f"  PR-AUC:      {result['metrics'].get('pr_auc', float('nan')):.4f}")
    logger.info(f"  F1:          {result['metrics'].get('f1', float('nan')):.4f}")
    logger.info(f"  Accuracy:    {result['metrics'].get('accuracy', float('nan')):.4f}")
    logger.info(f"  Precision:   {result['metrics'].get('precision', float('nan')):.4f}")
    logger.info(f"  Recall:      {result['metrics'].get('recall', float('nan')):.4f}")
    logger.info(f"  Specificity: {result['metrics'].get('specificity', float('nan')):.4f}")
    logger.info(f"  Gmean:       {result['metrics'].get('gmean', float('nan')):.4f}")
    
    # Create comparison with classical ML if available
    classical_ml_results_dir = Path(__file__).parent.parent / "01_classical_ml" / "results"
    if (classical_ml_results_dir / "model_comparison.csv").exists():
        logger.info("\n" + "="*70)
        logger.info("COMPARISON WITH CLASSICAL ML")
        logger.info("="*70)
        
        classical_df = pd.read_csv(classical_ml_results_dir / "model_comparison.csv")
        
        # Add TabPFN result
        tabpfn_row = {
            "Model": "tabpfn",
            "AUROC": result['metrics'].get('auroc', float('nan')),
            "PR-AUC": result['metrics'].get('pr_auc', float('nan')),
            "F1": result['metrics'].get('f1', float('nan')),
            "Accuracy": result['metrics'].get('accuracy', float('nan'))
        }
        
        comparison_df = pd.concat([classical_df, pd.DataFrame([tabpfn_row])], ignore_index=True)
        comparison_df = comparison_df.sort_values("AUROC", ascending=False)
        
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = results_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSaved comparison to: {comparison_path}")
    
    log_experiment_end(logger, "TabPFN FOUNDATION MODEL TRAINING")
    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()

