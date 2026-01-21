"""
Train fusion models (Logical OR, Cascade, Stacked) with Ablation B.

Fuses LR + TCN probabilities using 3 strategies:
- Logical OR: pred = (lr >= thr_lr) OR (tcn >= thr_tcn)
- Cascade: if lr < thr_lr → 0, else → (tcn >= thr_tcn)
- Stacked: Meta-LR trained on [lr_proba, tcn_proba]

For each fusion strategy, applies 3 threshold strategies (B1, B2, B3).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.evaluation import evaluate_predictions, find_optimal_threshold
from shared.logging_utils import setup_logger

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Local config
import config as exp_config


def load_predictions(model_name: str, strategy: str, results_dir: Path) -> pd.DataFrame:
    """Load predictions from a trained model."""
    pred_path = results_dir / model_name / f"{model_name}_{strategy}_predictions.csv"
    return pd.read_csv(pred_path)


# def logical_or_fusion(lr_proba: np.ndarray, tcn_proba: np.ndarray,
#                      lr_thr: float, tcn_thr: float) -> np.ndarray:
#     """Logical OR: predict stress if either model predicts stress."""
#     return ((lr_proba >= lr_thr) | (tcn_proba >= tcn_thr)).astype(int)


# def cascade_fusion(lr_proba: np.ndarray, tcn_proba: np.ndarray,
#                   lr_thr: float, tcn_thr: float) -> np.ndarray:
#     """Cascade: LR screens, TCN confirms."""
#     pred = np.zeros(len(lr_proba), dtype=int)
#     # If LR predicts stress, check TCN
#     lr_positive = lr_proba >= lr_thr
#     pred[lr_positive] = (tcn_proba[lr_positive] >= tcn_thr).astype(int)
#     return pred


def train_stacked_meta_model(lr_proba: np.ndarray, tcn_proba: np.ndarray,
                            y_true: np.ndarray, subjects: np.ndarray,
                            logger) -> Dict:
    """
    Train stacked ensemble using nested LOSO.
    
    For each test subject:
    1. Hold out test subject
    2. Train meta-LR on out-of-fold base predictions
    3. Predict on test subject
    4. Select 3 thresholds on train, apply to test
    """
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    logger.info(f"\nStacked Ensemble: Nested LOSO with {n_subjects} folds")
    
    # Storage for 3 threshold strategies
    results_b1 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    results_b2 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    results_b3 = {"y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_metrics": []}
    
    pbar = tqdm(unique_subjects, desc="  Stacked", unit="fold")
    
    for test_subject in pbar:
        # Split
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        # Training data for meta-model
        X_train = np.column_stack([lr_proba[train_mask], tcn_proba[train_mask]])
        y_train = y_true[train_mask]
        
        # Test data
        X_test = np.column_stack([lr_proba[test_mask], tcn_proba[test_mask]])
        y_test = y_true[test_mask]
        
        if len(X_test) == 0:
            continue
        
        # Train meta-model (Logistic Regression)
        meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_model.fit(X_train, y_train)
        
        # Get probabilities
        y_train_proba = meta_model.predict_proba(X_train)[:, 1]
        y_test_proba = meta_model.predict_proba(X_test)[:, 1]
        
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
        
        # Evaluate
        metrics_b1 = evaluate_predictions(y_test, y_pred_b1, y_test_proba, "Stacked", threshold=thr_b1)
        metrics_b1["subject"] = test_subject
        
        metrics_b2 = evaluate_predictions(y_test, y_pred_b2, y_test_proba, "Stacked", threshold=thr_b2)
        metrics_b2["subject"] = test_subject
        
        metrics_b3 = evaluate_predictions(y_test, y_pred_b3, y_test_proba, "Stacked", threshold=thr_b3)
        metrics_b3["subject"] = test_subject
        
        # Store
        for result, y_pred, metrics in [(results_b1, y_pred_b1, metrics_b1),
                                         (results_b2, y_pred_b2, metrics_b2),
                                         (results_b3, y_pred_b3, metrics_b3)]:
            result["y_true"].extend(y_test)
            result["y_pred"].extend(y_pred)
            result["y_proba"].extend(y_test_proba)
            result["subjects"].extend([test_subject] * len(y_test))
            result["fold_metrics"].append(metrics)
        
        pbar.set_postfix({
            "B1_Rec": f"{metrics_b1.get('recall', 0.0):.2f}",
            "B2_Rec": f"{metrics_b2.get('recall', 0.0):.2f}"
        })
    
    pbar.close()
    
    # Aggregate
    aggregated_results = {}
    for strategy, result in [("b1", results_b1), ("b2", results_b2), ("b3", results_b3)]:
        y_true_agg = np.array(result["y_true"])
        y_pred_agg = np.array(result["y_pred"])
        y_proba_agg = np.array(result["y_proba"])
        
        agg_metrics = evaluate_predictions(y_true_agg, y_pred_agg, y_proba_agg, "Stacked")
        
        fold_df = pd.DataFrame(result["fold_metrics"])
        for metric in ["recall", "far", "specificity", "precision", "gmean", "f1"]:
            if metric in fold_df.columns:
                agg_metrics[f"{metric}_mean"] = float(fold_df[metric].mean())
                agg_metrics[f"{metric}_std"] = float(fold_df[metric].std())
        
        aggregated_results[strategy] = {
            "metrics": agg_metrics,
            "fold_metrics": result["fold_metrics"],
            "predictions": {
                "y_true": y_true_agg.tolist(),
                "y_pred": y_pred_agg.tolist(),
                "y_proba": y_proba_agg.tolist(),
                "subjects": result["subjects"]
            }
        }
        
        logger.info(f"{strategy.upper()}: Recall={agg_metrics['recall']:.3f}, FAR={agg_metrics['false_alarm_rate']:.3f}")
    
    return aggregated_results


def train_rule_based_fusion(lr_proba: np.ndarray, tcn_proba: np.ndarray,
                            y_true: np.ndarray, subjects: np.ndarray,
                            fusion_func, fusion_name: str, logger) -> Dict:
    """
    Train rule-based fusion (OR or Cascade) with Ablation B.
    
    For each test subject:
    1. Hold out test subject
    2. Select thresholds for LR and TCN on training data
    3. Apply fusion rule with thresholds to test data
    4. Evaluate 3 threshold selection strategies
    """
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    logger.info(f"\n{fusion_name}: LOSO with {n_subjects} folds")
    
    # Storage for 3 threshold strategies
    results_b1 = {"y_true": [], "y_pred": [], "subjects": [], "fold_metrics": []}
    results_b2 = {"y_true": [], "y_pred": [], "subjects": [], "fold_metrics": []}
    results_b3 = {"y_true": [], "y_pred": [], "subjects": [], "fold_metrics": []}
    
    pbar = tqdm(unique_subjects, desc=f"  {fusion_name}", unit="fold")
    
    for test_subject in pbar:
        # Split
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        lr_train = lr_proba[train_mask]
        tcn_train = tcn_proba[train_mask]
        y_train = y_true[train_mask]
        
        lr_test = lr_proba[test_mask]
        tcn_test = tcn_proba[test_mask]
        y_test = y_true[test_mask]
        
        if len(y_test) == 0:
            continue
        
        # Select thresholds for base models on TRAIN data
        # B1: Unconstrained G-Mean
        lr_thr_b1, _ = find_optimal_threshold(y_train, lr_train, method="geometric_mean")
        tcn_thr_b1, _ = find_optimal_threshold(y_train, tcn_train, method="geometric_mean")
        
        # B2: Recall-constrained
        lr_thr_b2, _ = find_optimal_threshold(y_train, lr_train, method="constrained_gmean",
                                             min_recall=0.70, max_fpr=1.0)
        tcn_thr_b2, _ = find_optimal_threshold(y_train, tcn_train, method="constrained_gmean",
                                              min_recall=0.70, max_fpr=1.0)
        
        # B3: FAR-constrained
        lr_thr_b3, _ = find_optimal_threshold(y_train, lr_train, method="constrained_gmean",
                                             min_recall=0.0, max_fpr=0.30)
        tcn_thr_b3, _ = find_optimal_threshold(y_train, tcn_train, method="constrained_gmean",
                                              min_recall=0.0, max_fpr=0.30)
        
        # Apply fusion to TEST data
        y_pred_b1 = fusion_func(lr_test, tcn_test, lr_thr_b1, tcn_thr_b1)
        y_pred_b2 = fusion_func(lr_test, tcn_test, lr_thr_b2, tcn_thr_b2)
        y_pred_b3 = fusion_func(lr_test, tcn_test, lr_thr_b3, tcn_thr_b3)
        
        # For rule-based fusion, we don't have probabilities, use average of base probabilities
        y_test_proba = (lr_test + tcn_test) / 2.0
        
        # Evaluate
        metrics_b1 = evaluate_predictions(y_test, y_pred_b1, y_test_proba, fusion_name)
        metrics_b1["subject"] = test_subject
        
        metrics_b2 = evaluate_predictions(y_test, y_pred_b2, y_test_proba, fusion_name)
        metrics_b2["subject"] = test_subject
        
        metrics_b3 = evaluate_predictions(y_test, y_pred_b3, y_test_proba, fusion_name)
        metrics_b3["subject"] = test_subject
        
        # Store
        for result, y_pred, metrics in [(results_b1, y_pred_b1, metrics_b1),
                                         (results_b2, y_pred_b2, metrics_b2),
                                         (results_b3, y_pred_b3, metrics_b3)]:
            result["y_true"].extend(y_test)
            result["y_pred"].extend(y_pred)
            result["subjects"].extend([test_subject] * len(y_test))
            result["fold_metrics"].append(metrics)
        
        pbar.set_postfix({
            "B1_Rec": f"{metrics_b1['recall']:.2f}",
            "B3_FAR": f"{metrics_b3['false_alarm_rate']:.2f}"
        })
    
    pbar.close()
    
    # Aggregate
    aggregated_results = {}
    for strategy, result in [("b1", results_b1), ("b2", results_b2), ("b3", results_b3)]:
        y_true_agg = np.array(result["y_true"])
        y_pred_agg = np.array(result["y_pred"])
        
        # Use predictions directly (no probability thresholding)
        # Compute threshold-independent metrics
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        agg_metrics = {
            "recall": float(np.sum((y_pred_agg == 1) & (y_true_agg == 1)) / max(np.sum(y_true_agg == 1), 1)),
            "specificity": float(np.sum((y_pred_agg == 0) & (y_true_agg == 0)) / max(np.sum(y_true_agg == 0), 1)),
            "precision": float(np.sum((y_pred_agg == 1) & (y_true_agg == 1)) / max(np.sum(y_pred_agg == 1), 1)),
        }
        agg_metrics["far"] = 1.0 - agg_metrics["specificity"]
        agg_metrics["gmean"] = np.sqrt(agg_metrics["recall"] * agg_metrics["specificity"])
        
        # Threshold-independent metrics computed from averaged probabilities
        y_proba_avg = (lr_proba + tcn_proba) / 2.0
        if len(np.unique(y_true)) > 1:
            agg_metrics["auroc"] = float(roc_auc_score(y_true, y_proba_avg))
            precision, recall, _ = precision_recall_curve(y_true, y_proba_avg)
            agg_metrics["pr_auc"] = float(auc(recall, precision))
        else:
            agg_metrics["auroc"] = 0.0
            agg_metrics["pr_auc"] = 0.0
        
        # Fold statistics
        fold_df = pd.DataFrame(result["fold_metrics"])
        for metric in ["recall", "far", "specificity", "precision", "gmean"]:
            if metric in fold_df.columns:
                agg_metrics[f"{metric}_mean"] = float(fold_df[metric].mean())
                agg_metrics[f"{metric}_std"] = float(fold_df[metric].std())
        
        aggregated_results[strategy] = {
            "metrics": agg_metrics,
            "fold_metrics": result["fold_metrics"],
            "predictions": {
                "y_true": y_true_agg.tolist(),
                "y_pred": y_pred_agg.tolist(),
                "subjects": result["subjects"]
            }
        }
        
        logger.info(f"{strategy.upper()}: Recall={agg_metrics['recall']:.3f}, FAR={agg_metrics['far']:.3f}")
    
    return aggregated_results


def main():
    # Setup
    results_dir = Path(__file__).parent / "results"
    logger = setup_logger("train_fusion", results_dir / "training_fusion.log")
    
    logger.info("="*60)
    logger.info("FUSION MODELS TRAINING - ABLATION B & C")
    logger.info("="*60)
    logger.info("Models: Logical OR, Cascade, Stacked")
    logger.info("Threshold strategies: B1 (G-Mean), B2 (Recall≥70%), B3 (FAR≤30%)")
    
    # Load base model predictions (use B1 for base probabilities)
    logger.info("\nLoading base model predictions...")
    lr_pred = load_predictions("lr", "b1", results_dir)
    tcn_pred = load_predictions("tcn", "b1", results_dir)
    
    # Verify alignment
    assert all(lr_pred["subjects"] == tcn_pred["subjects"]), "Subject mismatch!"
    assert all(lr_pred["y_true"] == tcn_pred["y_true"]), "Label mismatch!"
    
    logger.info(f"Loaded {len(lr_pred)} windows")
    
    # Extract arrays
    lr_proba = lr_pred["y_proba"].values
    tcn_proba = tcn_pred["y_proba"].values
    y_true = lr_pred["y_true"].values
    subjects = lr_pred["subjects"].values
    
    # Train fusion models
    fusion_configs = [
        ("or", logical_or_fusion, "Logical OR"),
        ("cascade", cascade_fusion, "Cascade"),
    ]
    
    for model_dir_name, fusion_func, fusion_display_name in fusion_configs:
        results = train_rule_based_fusion(lr_proba, tcn_proba, y_true, subjects,
                                         fusion_func, fusion_display_name, logger)
        
        # Save results
        model_dir = results_dir / model_dir_name
        model_dir.mkdir(exist_ok=True)
        
        for strategy in ["b1", "b2", "b3"]:
            with open(model_dir / f"{model_dir_name}_{strategy}_metrics.json", 'w') as f:
                json.dump(results[strategy]["metrics"], f, indent=2)
            
            fold_df = pd.DataFrame(results[strategy]["fold_metrics"])
            fold_df.to_csv(model_dir / f"{model_dir_name}_{strategy}_fold_metrics.csv", index=False)
            
            pred_df = pd.DataFrame(results[strategy]["predictions"])
            pred_df.to_csv(model_dir / f"{model_dir_name}_{strategy}_predictions.csv", index=False)
    
    # Train stacked ensemble (uses nested LOSO, produces its own probabilities)
    logger.info("\n" + "="*60)
    stacked_results = train_stacked_meta_model(lr_proba, tcn_proba, y_true, subjects, logger)
    
    # Save stacked results
    stacked_dir = results_dir / "stacked"
    stacked_dir.mkdir(exist_ok=True)
    
    for strategy in ["b1", "b2", "b3"]:
        with open(stacked_dir / f"stacked_{strategy}_metrics.json", 'w') as f:
            json.dump(stacked_results[strategy]["metrics"], f, indent=2)
        
        fold_df = pd.DataFrame(stacked_results[strategy]["fold_metrics"])
        fold_df.to_csv(stacked_dir / f"stacked_{strategy}_fold_metrics.csv", index=False)
        
        pred_df = pd.DataFrame(stacked_results[strategy]["predictions"])
        pred_df.to_csv(stacked_dir / f"stacked_{strategy}_predictions.csv", index=False)
    
    logger.info("\n" + "="*60)
    logger.info("FUSION TRAINING COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()

