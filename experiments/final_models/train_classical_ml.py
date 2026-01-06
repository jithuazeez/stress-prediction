"""
Train classical ML models (LR, RF, SVM) with Ablation B (3 threshold strategies).

Fixed hyperparameters - no tuning.
Per-fold imputation - zero leakage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import time
import warnings

warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Imports from shared modules
from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_signals
from shared.windowing import create_labeled_windows, parse_stress_events, compute_subject_stats
from shared.evaluation import evaluate_predictions, find_optimal_threshold
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import setup_logger

# Import feature extraction from classical_ml
sys.path.insert(0, str(Path(__file__).parent.parent / "classical_ml"))
from feature_extraction import BasicFeatureExtractor

# Import HRV extractor
try:
    from shared.hrv_extractor import extract_hrv_from_window
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# Local config
import config as exp_config


def process_subject(subject_folder: Path, config: Config, extractor: BasicFeatureExtractor, logger) -> Tuple[pd.DataFrame, str]:
    """Process a single subject: align, window, extract features."""
    signals = load_raw_signals(subject_folder)
    subject_id = signals["subject_id"]
    
    try:
        start, end = get_experiment_time_range(signals)
    except ValueError:
        return None, subject_id
    
    # Align to 4Hz
    aligned = align_signals(signals, start, end, target_hz=exp_config.TARGET_HZ)
    if aligned is None or len(aligned) == 0:
        return None, subject_id
    
    # Compute subject stats
    subject_stats = compute_subject_stats(aligned)
    
    # Parse stress events
    event_info = parse_stress_events(
        signals.get("annotation"),
        config.stress_start_events,
        config.stress_stop_events
    )
    
    # Create windows
    windows = create_labeled_windows(
        aligned,
        event_info,
        window_size_sec=exp_config.WINDOW_SIZE_SEC,
        overlap_ratio=exp_config.OVERLAP_RATIO,
        horizons_minutes=[3, 5, 10],
        skip_first_minutes=config.skip_first_minutes,
        subject_stats=subject_stats,
    )
    
    if len(windows) == 0:
        return None, subject_id
    
    # Extract features with HRV (if available)
    rows = []
    hr_hrv_cache = {}
    
    # Extract HRV features first if available
    if HEARTPY_AVAILABLE and signals.get("ppg") is not None:
        from shared.hrv_extractor import extract_hrv_from_window as extract_hrv
        ppg_df = signals["ppg"]
        
        for window in windows:
            window_key = (window["window_start"], window["window_end"])
            hrv_features = extract_hrv(
                ppg_df=ppg_df,
                window_start=window["window_start"],
                window_end=window["window_end"],
                sample_rate=64.0
            )
            hr_hrv_cache[window_key] = hrv_features
    
    # Extract features from aligned data
    for window in windows:
        window_data = window["window_data"]
        window_key = (window["window_start"], window["window_end"])
        hrv_features = hr_hrv_cache.get(window_key)
        subject_stats_win = window.get("subject_stats", None)
        
        # Extract basic features
        features = extractor.extract_from_window(
            window_data,
            hr_hrv_features=hrv_features,
            subject_stats=subject_stats_win
        )
        
        # Add metadata and labels
        row = {
            "subject_id": subject_id,
            "window_start": window["window_start"],
            "window_end": window["window_end"],
            "label_5min": window.get("label_5min", 0),
            **features
        }
        rows.append(row)
    
    return pd.DataFrame(rows), subject_id


def prepare_data(df: pd.DataFrame, logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and labels with quality filtering.
    Returns data WITHOUT imputation (done per-fold).
    """
    metadata_cols = ["subject_id", "window_start", "window_end", "label_5min"]
    
    # Stage 1: Quality filtering - keep only windows where HRV succeeded
    if 'hr_bpm' in df.columns:
        initial_count = len(df)
        df_filtered = df[df['hr_bpm'].notna()].copy()
        logger.info(f"Quality filtering: {len(df_filtered)}/{initial_count} windows retained ({100*len(df_filtered)/initial_count:.1f}%)")
    else:
        df_filtered = df.copy()
        logger.info("No HRV column - skipping quality filtering")
    
    # Get feature columns
    feature_cols = [col for col in df_filtered.columns if col not in metadata_cols]
    X_df = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # Drop entirely NaN columns
    cols_to_drop = X_df.columns[X_df.isna().all()].tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} entirely NaN columns")
        X_df = X_df.drop(columns=cols_to_drop)
        feature_cols = [col for col in feature_cols if col not in cols_to_drop]
    
    X = X_df.values.astype(float)
    y = df_filtered["label_5min"].values
    subjects = df_filtered["subject_id"].values
    
    logger.info(f"Dataset: {X.shape[0]} windows, {X.shape[1]} features, {len(np.unique(subjects))} subjects")
    logger.info(f"Class distribution: {np.sum(y==0)} negative, {np.sum(y==1)} positive ({100*np.mean(y):.1f}% positive)")
    
    return X, y, subjects, feature_cols


def loso_with_ablation_b(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                          model_class, model_name: str, model_params: Dict,
                          logger) -> Dict:
    """
    LOSO CV with Ablation B: 3 threshold strategies per fold.
    
    For each test subject:
    1. Split: train vs test
    2. Impute: fit on train, transform test
    3. Train model
    4. Select 3 thresholds (B1, B2, B3) on train probabilities
    5. Apply all 3 thresholds to test probabilities
    6. Evaluate each threshold strategy
    """
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Params: {model_params}")
    logger.info(f"{'='*60}")
    logger.info(f"LOSO CV: {n_subjects} folds")
    
    # Storage for 3 threshold strategies
    results_b1 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    results_b2 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    results_b3 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    
    pbar = tqdm(unique_subjects, desc=f"  {model_name}", unit="fold")
    
    for test_subject in pbar:
        # Split
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(X_test) == 0:
            continue
        
        # Per-fold imputation (fit on train only)
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Handle any remaining NaNs
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Get probabilities
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Select 3 thresholds on TRAIN data
        thr_b1, _ = find_optimal_threshold(y_train, y_train_proba, method="geometric_mean")
        thr_b2, _ = find_optimal_threshold(y_train, y_train_proba, method="constrained_gmean", 
                                          min_recall=0.70, max_fpr=1.0)
        thr_b3, _ = find_optimal_threshold(y_train, y_train_proba, method="constrained_gmean",
                                          min_recall=0.0, max_fpr=0.30)
        
        # Apply thresholds to TEST data
        y_pred_b1 = (y_test_proba >= thr_b1).astype(int)
        y_pred_b2 = (y_test_proba >= thr_b2).astype(int)
        y_pred_b3 = (y_test_proba >= thr_b3).astype(int)
        
        # Evaluate each threshold strategy
        metrics_b1 = evaluate_predictions(y_test, y_pred_b1, y_test_proba, model_name, threshold=thr_b1)
        metrics_b1["subject"] = test_subject
        
        metrics_b2 = evaluate_predictions(y_test, y_pred_b2, y_test_proba, model_name, threshold=thr_b2)
        metrics_b2["subject"] = test_subject
        
        metrics_b3 = evaluate_predictions(y_test, y_pred_b3, y_test_proba, model_name, threshold=thr_b3)
        metrics_b3["subject"] = test_subject
        
        # Store results for each strategy
        for result, y_pred, metrics in [(results_b1, y_pred_b1, metrics_b1),
                                         (results_b2, y_pred_b2, metrics_b2),
                                         (results_b3, y_pred_b3, metrics_b3)]:
            result["y_true"].extend(y_test)
            result["y_pred"].extend(y_pred)
            result["y_proba"].extend(y_test_proba)
            result["subjects"].extend([test_subject] * len(y_test))
            result["fold_metrics"].append(metrics)
        
        pbar.set_postfix({
            "B1_Rec": f"{metrics_b1['recall']:.2f}",
            "B2_Rec": f"{metrics_b2['recall']:.2f}",
            "B3_FAR": f"{metrics_b3.get('false_alarm_rate', 0.0):.2f}"
        })
    
    pbar.close()
    
    # Aggregate results for each threshold strategy
    aggregated_results = {}
    for strategy, result in [("b1", results_b1), ("b2", results_b2), ("b3", results_b3)]:
        y_true = np.array(result["y_true"])
        y_pred = np.array(result["y_pred"])
        y_proba = np.array(result["y_proba"])
        
        # Aggregate metrics
        agg_metrics = evaluate_predictions(y_true, y_pred, y_proba, model_name)
        
        # Add FAR alias
        if "false_alarm_rate" in agg_metrics:
            agg_metrics["far"] = agg_metrics["false_alarm_rate"]
        
        # Add fold-level statistics
        fold_df = pd.DataFrame(result["fold_metrics"])
        # Map false_alarm_rate to far for easier access
        if "false_alarm_rate" in fold_df.columns:
            fold_df["far"] = fold_df["false_alarm_rate"]
        
        for metric in ["recall", "far", "specificity", "precision", "gmean", "f1"]:
            if metric in fold_df.columns:
                agg_metrics[f"{metric}_mean"] = float(fold_df[metric].mean())
                agg_metrics[f"{metric}_std"] = float(fold_df[metric].std())
        
        aggregated_results[strategy] = {
            "metrics": agg_metrics,
            "fold_metrics": result["fold_metrics"],
            "predictions": {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist(),
                "subjects": result["subjects"]
            }
        }
        
        logger.info(f"\n{strategy.upper()}: Recall={agg_metrics['recall']:.3f}, FAR={agg_metrics['far']:.3f}, G-mean={agg_metrics['gmean']:.3f}")
    
    return aggregated_results


def main():
    # Setup logging
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    logger = setup_logger("train_classical_ml", results_dir / "training_classical_ml.log")
    
    logger.info("="*60)
    logger.info("CLASSICAL ML TRAINING - ABLATION B")
    logger.info("="*60)
    logger.info(f"Models: LR, RF, SVM")
    logger.info(f"Fixed hyperparameters (no tuning)")
    logger.info(f"Threshold strategies: B1 (G-Mean), B2 (Recall≥70%), B3 (FAR≤30%)")
    
    # Load and prepare data
    logger.info("\nLoading data...")
    config = Config(**{**vars(DEFAULT_CONFIG), 
                      'target_sample_rate': exp_config.TARGET_HZ,
                      'overlap_ratio': exp_config.OVERLAP_RATIO})
    
    dataset_path = exp_config.DATASET_PATH
    subjects = get_all_subjects(dataset_path)
    logger.info(f"Found {len(subjects)} subjects")
    
    # Process all subjects
    extractor = BasicFeatureExtractor()
    all_dfs = []
    
    for subject_folder in tqdm(subjects, desc="Processing subjects"):
        df, _ = process_subject(subject_folder, config, extractor, logger)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
    
    if len(all_dfs) == 0:
        logger.error("No data loaded")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined_df)} windows")
    
    # Prepare data
    X, y, subjects_arr, feature_names = prepare_data(combined_df, logger)
    
    # Train models
    models = {
        "lr": (LogisticRegression, exp_config.LR_PARAMS),
        "rf": (RandomForestClassifier, exp_config.RF_PARAMS),
        "svm": (SVC, exp_config.SVM_PARAMS)
    }
    
    for model_name, (model_class, model_params) in models.items():
        start_time = time.time()
        
        # Run LOSO with Ablation B
        results = loso_with_ablation_b(X, y, subjects_arr, model_class, model_name.upper(), model_params, logger)
        
        # Save results for each threshold strategy
        model_dir = results_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        for strategy in ["b1", "b2", "b3"]:
            strategy_results = results[strategy]
            
            # Save metrics
            with open(model_dir / f"{model_name}_{strategy}_metrics.json", 'w') as f:
                json.dump(strategy_results["metrics"], f, indent=2)
            
            # Save fold metrics
            fold_df = pd.DataFrame(strategy_results["fold_metrics"])
            fold_df.to_csv(model_dir / f"{model_name}_{strategy}_fold_metrics.csv", index=False)
            
            # Save predictions
            pred_df = pd.DataFrame(strategy_results["predictions"])
            pred_df.to_csv(model_dir / f"{model_name}_{strategy}_predictions.csv", index=False)
        
        elapsed = time.time() - start_time
        logger.info(f"\n{model_name.upper()} completed in {elapsed/60:.1f} minutes")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()

