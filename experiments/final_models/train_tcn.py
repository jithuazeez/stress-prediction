"""
Train TCN model with Ablation B (3 threshold strategies).

Fixed hyperparameters - no tuning.
Subject-wise normalization, forward-fill for missing data.
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tcn"))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_signals
from shared.windowing import create_labeled_windows, parse_stress_events, compute_subject_stats
from shared.evaluation import evaluate_predictions, find_optimal_threshold
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import setup_logger

# TCN imports
from tcn.dataset import VitaStressTCNDataset
from tcn.model import create_tcn_model, FocalLoss

# Local config
import config as exp_config


def load_all_windows(config: Config, logger) -> Dict[str, List[Dict]]:
    """Load and process all subjects, returning windows by subject."""
    subjects = get_all_subjects(config.data_path)
    windows_by_subject = {}
    hr_timeseries_data = []  # Collect HR time series for saving
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Processing subjects", unit="subject")
    
    for subject_folder in pbar:
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            continue
        
        aligned = align_signals(signals, start, end, target_hz=exp_config.TARGET_HZ)
        if aligned is None or len(aligned) == 0:
            continue
        
        subject_stats = compute_subject_stats(aligned)
        
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events
        )
        
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=exp_config.WINDOW_SIZE_SEC,
            overlap_ratio=exp_config.OVERLAP_RATIO,
            horizons_minutes=[3, 5, 10],
            skip_first_minutes=config.skip_first_minutes,
            subject_stats=subject_stats,
        )
        
        # Extract HR time series from windows for saving
        if windows:
            for window in windows:
                window_data = window.get("window_data")
                if window_data is not None and "hr_bpm" in window_data.columns:
                    # Extract HR data with timestamps
                    hr_subset = window_data[["timestamp", "hr_bpm", "rmssd"]].copy()
                    hr_subset["subject_id"] = subject_id
                    hr_subset["window_id"] = window.get("window_id", -1)
                    
                    # Only save if HR data is valid (not all NaN)
                    hr_timeseries_data.append(hr_subset)
        
        if windows:
            windows_by_subject[subject_id] = windows
    
    pbar.close()
    logger.info(f"Loaded {len(windows_by_subject)} subjects")
    
    # Save extracted HR time series to CSV
    if hr_timeseries_data:
        hr_df = pd.concat(hr_timeseries_data, ignore_index=True)
        reports_dir = Path(__file__).parent.parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        hr_output_path = reports_dir / "hr_extracted_4hz.csv"
        hr_df.to_csv(hr_output_path, index=False)
        logger.info(f"Saved extracted 4Hz HR time series to: {hr_output_path}")
        logger.info(f"  Total HR samples: {len(hr_df)}")
        logger.info(f"  Subjects with HR: {hr_df['subject_id'].nunique()}")
        logger.info(f"  Windows with HR: {len(hr_timeseries_data)}")
    else:
        logger.warning("No HR time series data extracted to save")
    
    return windows_by_subject


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_model(model, data_loader, device):
    """Get predictions and probabilities."""
    model.eval()
    all_y_true = []
    all_y_proba = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            proba = torch.softmax(logits, dim=1)[:, 1]
            
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_proba.extend(proba.cpu().numpy())
    
    return np.array(all_y_true), np.array(all_y_proba)


def loso_with_ablation_b(windows_by_subject: Dict[str, List[Dict]],
                          config: Config, device: torch.device, logger) -> Dict:
    """
    LOSO CV with Ablation B: 3 threshold strategies per fold.
    """
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    # TCN hyperparameters from config
    tcn_params = exp_config.TCN_PARAMS
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: TCN")
    logger.info(f"Params: {tcn_params}")
    logger.info(f"{'='*60}")
    logger.info(f"LOSO CV: {n_subjects} folds")
    
    # Storage for 3 threshold strategies
    results_b1 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    results_b2 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    results_b3 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    
    pbar = tqdm(subjects, desc="  TCN", unit="fold")
    
    for test_subject in pbar:
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
        
        # Create datasets (480 timesteps for 4Hz * 120s)
        train_dataset = VitaStressTCNDataset(train_windows, config.target_label,
                                            seq_len=480, normalize=True,
                                            normalization_mode="subject")
        test_dataset = VitaStressTCNDataset(test_windows, config.target_label,
                                           seq_len=480, normalize=True,
                                           normalization_mode="subject")
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=tcn_params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=tcn_params["batch_size"], shuffle=False)
        
        # Create model
        num_channels_input = train_dataset.get_n_channels()
        model = create_tcn_model(
            num_inputs=num_channels_input,
            num_classes=2,
            num_channels=tcn_params["channels"],
            kernel_size=3,
            dilations=tcn_params["dilations"],
            dropout=tcn_params["dropout"],
            fc_hidden_dim=128,
            use_last_timestep=True
        ).to(device)
        
        # Loss and optimizer
        criterion = FocalLoss(alpha=0.88, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=tcn_params["lr"])
        
        # Train with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(tcn_params["epochs"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= tcn_params["patience"]:
                break
        
        # Get probabilities
        y_train, y_train_proba = evaluate_model(model, train_loader, device)
        y_test, y_test_proba = evaluate_model(model, test_loader, device)
        
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
        metrics_b1 = evaluate_predictions(y_test, y_pred_b1, y_test_proba, "TCN", threshold=thr_b1)
        metrics_b1["subject"] = test_subject
        
        metrics_b2 = evaluate_predictions(y_test, y_pred_b2, y_test_proba, "TCN", threshold=thr_b2)
        metrics_b2["subject"] = test_subject
        
        metrics_b3 = evaluate_predictions(y_test, y_pred_b3, y_test_proba, "TCN", threshold=thr_b3)
        metrics_b3["subject"] = test_subject
        
        # Store results
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
    
    # Aggregate results
    aggregated_results = {}
    for strategy, result in [("b1", results_b1), ("b2", results_b2), ("b3", results_b3)]:
        y_true = np.array(result["y_true"])
        y_pred = np.array(result["y_pred"])
        y_proba = np.array(result["y_proba"])
        
        agg_metrics = evaluate_predictions(y_true, y_pred, y_proba, "TCN")
        
        # Add FAR alias
        if "false_alarm_rate" in agg_metrics:
            agg_metrics["far"] = agg_metrics["false_alarm_rate"]
        
        # Add fold-level statistics
        fold_df = pd.DataFrame(result["fold_metrics"])
        # Map false_alarm_rate to far
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
    # Setup
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    logger = setup_logger("train_tcn", results_dir / "training_tcn.log")
    
    logger.info("="*60)
    logger.info("TCN TRAINING - ABLATION B")
    logger.info("="*60)
    logger.info(f"Fixed hyperparameters (no tuning)")
    logger.info(f"Threshold strategies: B1 (G-Mean), B2 (Recall≥70%), B3 (FAR≤30%)")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info("\nLoading data...")
    config = Config(**{**vars(DEFAULT_CONFIG),
                      'target_sample_rate': exp_config.TARGET_HZ,
                      'overlap_ratio': exp_config.OVERLAP_RATIO,
                      'data_path': exp_config.DATASET_PATH})
    
    windows_by_subject = load_all_windows(config, logger)
    
    if len(windows_by_subject) == 0:
        logger.error("No data loaded")
        return
    
    # Train TCN
    start_time = time.time()
    
    results = loso_with_ablation_b(windows_by_subject, config, device, logger)
    
    # Save results
    tcn_dir = results_dir / "tcn"
    tcn_dir.mkdir(exist_ok=True)
    
    for strategy in ["b1", "b2", "b3"]:
        strategy_results = results[strategy]
        
        # Save metrics
        with open(tcn_dir / f"tcn_{strategy}_metrics.json", 'w') as f:
            json.dump(strategy_results["metrics"], f, indent=2)
        
        # Save fold metrics
        fold_df = pd.DataFrame(strategy_results["fold_metrics"])
        fold_df.to_csv(tcn_dir / f"tcn_{strategy}_fold_metrics.csv", index=False)
        
        # Save predictions
        pred_df = pd.DataFrame(strategy_results["predictions"])
        pred_df.to_csv(tcn_dir / f"tcn_{strategy}_predictions.csv", index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"\nTCN completed in {elapsed/60:.1f} minutes")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()

