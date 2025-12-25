"""
Logging utilities for experiments.

Provides consistent, colorful logging across all experiments.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter with colors for terminal output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m"
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(name: str, 
                 log_file: Optional[Path] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent propagation to root logger (avoids duplicate logs)
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (without colors)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_experiment_start(logger: logging.Logger, experiment_name: str):
    """Log experiment start with header."""
    logger.info("=" * 70)
    logger.info(f"  {experiment_name}")
    logger.info(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


def log_experiment_end(logger: logging.Logger, experiment_name: str):
    """Log experiment completion."""
    logger.info("=" * 70)
    logger.info(f"  {experiment_name} - COMPLETE")
    logger.info(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


def log_data_summary(logger: logging.Logger,
                     n_subjects: int,
                     n_windows: int,
                     n_features: int,
                     n_positive: int,
                     n_negative: int):
    """Log data summary statistics."""
    logger.info("-" * 50)
    logger.info("DATA SUMMARY")
    logger.info("-" * 50)
    logger.info(f"  Subjects:        {n_subjects}")
    logger.info(f"  Total windows:   {n_windows}")
    logger.info(f"  Features:        {n_features}")
    logger.info(f"  Positive (1):    {n_positive} ({100*n_positive/n_windows:.1f}%)")
    logger.info(f"  Negative (0):    {n_negative} ({100*n_negative/n_windows:.1f}%)")
    logger.info(f"  Class ratio:     1:{n_negative/max(n_positive,1):.1f}")
    logger.info("-" * 50)


def log_model_results(logger: logging.Logger,
                      model_name: str,
                      metrics: dict):
    """
    Log model evaluation results with emphasis on clinical metrics.
    
    Highlights:
    - G-Mean (primary metric for balanced performance)
    - Sensitivity/Recall (stress detection rate)
    - Specificity (false alarm control)
    - Precision (positive predictive value)
    """
    logger.info("\n" + "=" * 50)
    logger.info(f"RESULTS: {model_name.upper()}")
    logger.info("=" * 50)
    
    # Primary clinical metrics
    logger.info("\nClinical Performance:")
    gmean = metrics.get("gmean", float("nan"))
    sensitivity = metrics.get("sensitivity", metrics.get("recall", float("nan")))
    specificity = metrics.get("specificity", float("nan"))
    precision = metrics.get("precision", float("nan"))
    
    logger.info(f"  G-Mean (Balanced):     {gmean:.4f}")
    logger.info(f"  Sensitivity (Recall):  {sensitivity:.4f}")
    logger.info(f"  Specificity:           {specificity:.4f}")
    logger.info(f"  Precision:             {precision:.4f}")
    
    # Standard metrics
    logger.info("\nStandard Metrics:")
    auroc = metrics.get("auroc", float("nan"))
    pr_auc = metrics.get("pr_auc", float("nan"))
    f1 = metrics.get("f1", float("nan"))
    accuracy = metrics.get("accuracy", float("nan"))
    
    logger.info(f"  AUROC:                 {auroc:.4f}")
    logger.info(f"  PR-AUC:                {pr_auc:.4f}")
    logger.info(f"  F1 Score:              {f1:.4f}")
    logger.info(f"  Accuracy:              {accuracy:.4f}")
    
    # Confusion matrix
    tp = metrics.get("true_positive", 0)
    tn = metrics.get("true_negative", 0)
    fp = metrics.get("false_positive", 0)
    fn = metrics.get("false_negative", 0)
    
    if tp + fp + fn + tn > 0:
        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Positive (Hit):   {tp}")
        logger.info(f"  True Negative (CR):    {tn}")
        logger.info(f"  False Positive (FA):   {fp}")
        logger.info(f"  False Negative (Miss): {fn}")
    
    # Threshold info
    if "threshold_mean" in metrics:
        threshold_mean = metrics.get("threshold_mean", 0.5)
        threshold_std = metrics.get("threshold_std", 0.0)
        threshold_min = metrics.get("threshold_min", 0.5)
        threshold_max = metrics.get("threshold_max", 0.5)
        logger.info(f"\nThreshold Statistics:")
        logger.info(f"  Mean: {threshold_mean:.4f} ± {threshold_std:.4f}")
        logger.info(f"  Range: [{threshold_min:.4f}, {threshold_max:.4f}]")
    
    logger.info("-" * 50)


def log_fold_progress(logger: logging.Logger,
                      fold_num: int,
                      total_folds: int,
                      subject_id: str,
                      train_size: int,
                      test_size: int,
                      auroc: Optional[float] = None):
    """Log LOSO fold progress."""
    progress = f"[{fold_num:2d}/{total_folds}]"
    msg = f"{progress} Subject: {subject_id[:8]}... | Train: {train_size} | Test: {test_size}"
    
    if auroc is not None:
        msg += f" | AUROC: {auroc:.3f}"
    
    logger.info(msg)

