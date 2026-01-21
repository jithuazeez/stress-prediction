"""
Fixed configuration for final models experiment.
All hyperparameters are frozen - no tuning.
"""

# ===== FIXED HYPERPARAMETERS =====

LR_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "saga",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "max_features": "log2",
    "min_samples_leaf": 4,
    "min_samples_split": 10,
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": 'scale',
    "degree": 2,
    "probability": True,
    "class_weight": "balanced",
    "random_state": 42
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss"
}

TCN_PARAMS = {
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 100,
    "dropout": 0.3,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
    "channels": [16] * 8,
    "patience": 15
}

# ===== PIPELINE CONFIGURATION =====

TARGET_HZ = 4.0
OVERLAP_RATIO = 0.5
WINDOW_SIZE_SEC = 120
LABEL_COL = "label_5min"
RANDOM_SEED = 42

# ===== THRESHOLD STRATEGIES =====

THRESHOLD_CONFIGS = {
    "B1": {"method": "gmean", "name": "Unconstrained G-Mean"},
    "B2": {"method": "recall", "min_recall": 0.70, "name": "Recall-Constrained (≥70%)"},
    "B3": {"method": "fpr", "max_fpr": 0.30, "name": "FAR-Constrained (≤30%)"}
}

# ===== PATHS =====

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "Datasets" / "VitaStress" / "data"
RESULTS_PATH = Path(__file__).parent / "results"

# ===== EVALUATION METRICS =====

METRICS_TO_REPORT = [
    "recall",
    "far",
    "specificity",
    "precision",
    "gmean",
    "auroc",
    "pr_auc",
    "balanced_accuracy",
    "f1"
]

