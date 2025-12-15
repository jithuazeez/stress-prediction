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

warnings.filterwarnings("ignore")

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_to_1hz
from shared.windowing import create_labeled_windows, parse_stress_events
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

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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
    Process a single subject: load signals, align, window, extract features.
    """
    signals = load_raw_signals(subject_folder)
    subject_id = signals["subject_id"]
    
    try:
        start, end = get_experiment_time_range(signals)
    except ValueError as e:
        logger.warning(f"Subject {subject_id[:8]}...: Could not determine time range - {e}")
        return None, subject_id
    
    # Align to 1Hz
    aligned = align_to_1hz(signals, start, end)
    
    if aligned is None or len(aligned) == 0:
        logger.warning(f"Subject {subject_id[:8]}...: No aligned data")
        return None, subject_id
    
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
        skip_first_minutes=config.skip_first_minutes
    )
    
    if not windows:
        logger.warning(f"Subject {subject_id[:8]}...: No windows created")
        return None, subject_id
    
    # Extract features for each window
    rows = []
    for window in windows:
        features = extractor.extract_from_window(window["window_data"])
        
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
    
    duration = (end - start).total_seconds() / 60
    logger.debug(f"Subject {subject_id[:8]}...: {len(windows)} windows from {duration:.1f} min recording")
    
    return pd.DataFrame(rows), subject_id


def prepare_data(df: pd.DataFrame, 
                 label_col: str = "label_5min") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare features and labels from DataFrame."""
    metadata_cols = ["subject_id", "window_id", "window_start", "window_end", 
                    "context", "label_3min", "label_5min", "label_10min"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df[label_col].values
    subjects = df["subject_id"].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, subjects, feature_cols


def loso_cross_validation(X: np.ndarray,
                          y: np.ndarray,
                          subjects: np.ndarray,
                          feature_names: List[str],
                          model_class,
                          model_name: str,
                          model_kwargs: Dict,
                          logger) -> Dict:
    """Perform Leave-One-Subject-Out cross-validation."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    if model_kwargs is None:
        model_kwargs = {}
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    logger.info(f"LOSO CV with {n_subjects} folds (subjects)")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    
    start_time = time.time()
    
    # Progress bar for LOSO
    pbar = tqdm(enumerate(unique_subjects), total=n_subjects, 
                desc=f"  {model_name}", unit="fold",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for fold_idx, test_subject in pbar:
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(X_test) == 0:
            continue
        
        # Class weight
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        
        # Model setup
        model_kwargs_copy = model_kwargs.copy()
        if "XGB" in str(model_class):
            model_kwargs_copy["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
        elif "class_weight" in str(model_class.__init__.__code__.co_varnames):
            model_kwargs_copy["class_weight"] = "balanced"
        
        model = model_class(**model_kwargs_copy)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            logger.warning(f"Fold {fold_idx+1}: Training failed for {test_subject[:8]}... - {e}")
            continue
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        try:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)
        
        # Store
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_test))
        
        # Fold metrics
        fold_metric = evaluate_predictions(y_test, y_pred, y_proba, model_name)
        fold_metric["subject"] = test_subject
        fold_metrics.append(fold_metric)
        
        # Update progress bar
        fold_auroc = fold_metric.get("auroc", float("nan"))
        pbar.set_postfix({"AUROC": f"{fold_auroc:.3f}", "Test": len(y_test)})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/n_subjects:.1f}s per fold)")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # Aggregate metrics
    aggregate_metrics = evaluate_predictions(all_y_true, all_y_pred, all_y_proba, model_name)
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    # Log results
    log_model_results(logger, model_name, aggregate_metrics)
    
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
    successful = 0
    failed = 0
    
    pbar = tqdm(subjects, desc="Loading subjects", unit="subject",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for subject_folder in pbar:
        df, subject_id = process_subject(subject_folder, config, extractor, logger)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            successful += 1
            pbar.set_postfix({"OK": successful, "Fail": failed, "Windows": len(df)})
        else:
            failed += 1
            pbar.set_postfix({"OK": successful, "Fail": failed})
    
    pbar.close()
    
    logger.info(f"Loaded: {successful}/{len(subjects)} subjects ({failed} failed)")
    
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
    
    # Prepare data
    X, y, subjects_arr, feature_names = prepare_data(combined_df, config.target_label)
    
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
        result = loso_cross_validation(
            X, y, subjects_arr, feature_names,
            model_class, model_key, model_kwargs, logger
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
