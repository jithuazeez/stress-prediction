"""
Training script for TS2Vec classifier with LOSO cross-validation.

Uses pretrained TS2Vec encoder to extract embeddings, then trains
a simple classifier on top.

References:
- https://github.com/zhihanyue/ts2vec
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import time
import pickle

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_to_1hz
from shared.windowing import create_labeled_windows, parse_stress_events
from shared.evaluation import (
    evaluate_predictions, aggregate_fold_metrics,
    save_results, save_predictions, plot_results
)
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import (
    setup_logger, log_experiment_start, log_experiment_end,
    log_data_summary, log_model_results
)

from dataset import VitaStressTS2VecDataset
from pretrain import TS2VecEncoder


def load_pretrained_encoder(checkpoint_path: Path,
                            config_path: Path,
                            device: torch.device) -> Tuple[TS2VecEncoder, Dict]:
    """Load pretrained encoder from checkpoint."""
    # Load config
    with open(config_path, "rb") as f:
        pretrain_config = pickle.load(f)
    
    # Create encoder with saved config
    encoder = TS2VecEncoder(
        input_channels=pretrain_config["n_channels"],
        output_dim=pretrain_config["output_dim"],
        hidden_dim=pretrain_config["hidden_dim"]
    )
    
    # Load weights
    encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder, pretrain_config


def extract_embeddings(encoder: TS2VecEncoder,
                       windows: List[Dict],
                       device: torch.device,
                       batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from windows using pretrained encoder.
    
    Returns:
        Tuple of (embeddings, labels)
    """
    dataset = VitaStressTS2VecDataset(windows, "label_5min")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            
            # Get encoder output (batch, time, dim)
            z = encoder(x)
            
            # Pool over time to get fixed-length embedding
            embedding = z.mean(dim=1)  # (batch, dim)
            
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.extend(y.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    return embeddings, labels


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


def loso_cross_validation(encoder: TS2VecEncoder,
                          windows_by_subject: Dict[str, List[Dict]],
                          config: Config,
                          device: torch.device,
                          logger) -> Dict:
    """Perform LOSO CV using extracted embeddings."""
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    logger.info(f"\nLOSO CV with {n_subjects} subjects")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    
    start_time = time.time()
    
    pbar = tqdm(enumerate(subjects), total=n_subjects, desc="LOSO folds", unit="fold")
    
    for fold_idx, test_subject in pbar:
        pbar.set_description(f"Fold {fold_idx+1}/{n_subjects} ({test_subject[:8]}...)")
        
        # Split windows
        train_windows = []
        test_windows = []
        
        for subject_id, windows in windows_by_subject.items():
            if subject_id == test_subject:
                test_windows.extend(windows)
            else:
                train_windows.extend(windows)
        
        if len(train_windows) == 0 or len(test_windows) == 0:
            continue
        
        # Extract embeddings
        X_train, y_train = extract_embeddings(encoder, train_windows, device)
        X_test, y_test = extract_embeddings(encoder, test_windows, device)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Scale embeddings
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train simple classifier
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.random_seed
        )
        classifier.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test_scaled)
        y_proba = classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_test))
        
        # Fold metrics
        fold_metric = evaluate_predictions(y_test, y_pred, y_proba, "ts2vec")
        fold_metric["subject"] = test_subject
        fold_metrics.append(fold_metric)
        
        fold_auroc = fold_metric.get("auroc", float("nan"))
        pbar.set_postfix({"AUROC": f"{fold_auroc:.3f}", "Test": len(y_test)})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # Aggregate metrics
    aggregate_metrics = evaluate_predictions(all_y_true, all_y_pred, all_y_proba, "ts2vec")
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": all_subjects,
        "fold_metrics": fold_metrics
    }


def main():
    """Main classifier training pipeline."""
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("ts2vec_classify", log_file=results_dir / "classify.log")
    
    log_experiment_start(logger, "TS2VEC CLASSIFIER TRAINING")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load pretrained encoder
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Loading pretrained encoder")
    logger.info("-"*50)
    
    encoder_path = results_dir / "ts2vec_encoder.pt"
    config_path = results_dir / "pretrain_config.pkl"
    
    if not encoder_path.exists():
        logger.error(f"Pretrained encoder not found at: {encoder_path}")
        logger.error("Run pretrain.py first!")
        return
    
    encoder, pretrain_config = load_pretrained_encoder(encoder_path, config_path, device)
    logger.info(f"Loaded encoder from: {encoder_path}")
    logger.info(f"  Output dim: {pretrain_config['output_dim']}")
    
    # Load all windows
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Loading data")
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
        n_features=pretrain_config["output_dim"],
        n_positive=n_pos,
        n_negative=total_windows - n_pos
    )
    
    # Run LOSO CV
    logger.info("\n" + "-"*50)
    logger.info("PHASE 3: Classifier Training (LOSO CV)")
    logger.info("-"*50)
    
    results = loso_cross_validation(
        encoder,
        windows_by_subject,
        config,
        device,
        logger
    )
    
    # Log results
    log_model_results(logger, "TS2Vec + LogReg", results["metrics"])
    
    # Save results
    save_results(results["metrics"], results_dir, "ts2vec")
    save_predictions(
        results["y_true"], results["y_pred"], results["y_proba"],
        results["subjects"], results_dir, "ts2vec"
    )
    
    if len(np.unique(results["y_true"])) > 1:
        plot_results(results["y_true"], results["y_proba"], "ts2vec", results_dir)
    
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / "ts2vec_fold_metrics.csv", index=False)
    
    log_experiment_end(logger, "TS2VEC CLASSIFIER TRAINING")
    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
