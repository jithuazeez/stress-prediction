"""
Evaluation utilities for stress prediction experiments.

Provides metrics calculation, result saving, and plotting functions.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, balanced_accuracy_score
)


def find_optimal_threshold(y_true: np.ndarray,
                            y_proba: np.ndarray,
                            method: str = "youden") -> Tuple[float, Dict[str, float]]:
    """
    Find optimal decision threshold from training data.
    
    This should be called on TRAINING data only, then applied to test data.
    This avoids data leakage from peeking at test labels.
    
    Args:
        y_true: True binary labels (from training data)
        y_proba: Predicted probabilities for positive class (from training data)
        method: Method to find threshold
            - "youden": Maximizes Youden's J statistic (TPR - FPR) - balances sensitivity/specificity
            - "f1": Maximizes F1 score
            - "balanced": Threshold where sensitivity ≈ specificity
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    if len(y_true) == 0 or len(y_proba) == 0:
        return 0.5, {}
    
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    if n_positive == 0 or n_negative == 0:
        return 0.5, {}
    
    # Get ROC curve: fpr = false alarm rate, tpr = sensitivity
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    if method == "youden":
        # Youden's J statistic: maximizes (sensitivity + specificity - 1)
        # Equivalent to maximizing (TPR - FPR)
        j_scores = tpr - fpr
        idx = np.argmax(j_scores)
    elif method == "f1":
        # Find threshold that maximizes F1 score
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
        idx = np.argmax(f1_scores[:-1])  # Last element is for recall=0
        # Map back to ROC thresholds (approximately)
        optimal_thresh = pr_thresholds[idx] if idx < len(pr_thresholds) else 0.5
        # Find closest ROC threshold
        idx = np.argmin(np.abs(thresholds - optimal_thresh))
    elif method == "balanced":
        # Find threshold where sensitivity ≈ specificity
        specificity = 1 - fpr
        diff = np.abs(tpr - specificity)
        idx = np.argmin(diff)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Handle edge case where thresholds array might be shorter
    threshold_idx = min(idx, len(thresholds) - 1)
    optimal_threshold = float(thresholds[threshold_idx])
    
    metrics = {
        "sensitivity": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
        "false_alarm_rate": float(fpr[idx]),
    }
    
    return optimal_threshold, metrics


def evaluate_predictions(y_true: np.ndarray,
                         y_pred: Optional[np.ndarray] = None,
                         y_proba: Optional[np.ndarray] = None,
                         model_name: str = "model",
                         threshold: Optional[float] = None) -> Dict:
    """
    Evaluate model predictions with comprehensive metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (optional if y_proba and threshold provided)
        y_proba: Predicted probabilities (optional)
        model_name: Name of the model for logging
        threshold: Decision threshold to use (default: 0.5)
                  Should be computed on TRAINING data, not test data!
    
    Returns:
        Dictionary of evaluation metrics including:
        - threshold: The decision threshold used
        - sensitivity/recall: True positive rate (what % of stress events detected)
        - specificity: True negative rate (1 - false alarm rate)
        - false_alarm_rate: False positive rate
        - precision, f1, accuracy, auroc, pr_auc, etc.
    """
    metrics = {"model_name": model_name}
    
    # Handle edge cases
    if len(y_true) == 0:
        return {"error": "Empty predictions"}
    
    # Basic counts
    n_samples = len(y_true)
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    metrics["n_samples"] = int(n_samples)
    metrics["n_positive"] = int(n_positive)
    metrics["n_negative"] = int(n_negative)
    
    # Determine threshold and compute y_pred if needed
    if threshold is None:
        threshold = 0.5
    
    metrics["threshold"] = float(threshold)
    
    # Compute y_pred from y_proba if not provided
    if y_pred is None:
        if y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            return {"error": "Either y_pred or y_proba must be provided"}
    
    # Classification metrics
    try:
        metrics["accuracy"] = float(np.mean(y_true == y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["sensitivity"] = metrics["recall"]  # Alias for clinical clarity
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        metrics["classification_error"] = str(e)
    
    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["true_negative"] = int(tn)
        metrics["false_positive"] = int(fp)
        metrics["false_negative"] = int(fn)
        metrics["true_positive"] = int(tp)
        
        # Derived metrics
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["detection_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    except Exception as e:
        metrics["confusion_error"] = str(e)
    
    # Probability-based metrics (threshold-independent)
    if y_proba is not None:
        try:
            # Only compute if both classes present
            if n_positive > 0 and n_negative > 0:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
            else:
                metrics["auroc"] = np.nan
                metrics["pr_auc"] = np.nan
        except Exception as e:
            metrics["proba_error"] = str(e)
            metrics["auroc"] = np.nan
            metrics["pr_auc"] = np.nan
    
    return metrics


def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across LOSO folds.
    
    Args:
        fold_metrics: List of metric dictionaries from each fold
    
    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}
    
    # Collect numeric metrics
    metric_keys = set()
    for m in fold_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not np.isnan(v) if isinstance(v, float) else True:
                metric_keys.add(k)
    
    for key in metric_keys:
        values = [m.get(key) for m in fold_metrics if m.get(key) is not None]
        values = [v for v in values if isinstance(v, (int, float)) and (not np.isnan(v) if isinstance(v, float) else True)]
        
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
    
    aggregated["n_folds"] = len(fold_metrics)
    
    return aggregated


def save_results(metrics: Dict,
                 output_dir: Path,
                 experiment_name: str) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save results
        experiment_name: Name of the experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{experiment_name}_metrics.json"
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    metrics_converted = convert(metrics)
    
    with open(output_file, "w") as f:
        json.dump(metrics_converted, f, indent=2)
    
    print(f"Saved results to {output_file}")


def save_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray],
                     subjects: np.ndarray,
                     output_dir: Path,
                     experiment_name: str) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        subjects: Subject IDs for each sample
        output_dir: Directory to save results
        experiment_name: Name of the experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        "subject_id": subjects,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    
    if y_proba is not None:
        df["y_proba"] = y_proba
    
    output_file = output_dir / f"{experiment_name}_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str,
                          output_dir: Path) -> None:
    """
    Generate and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum() * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        classes = ['No Stress (0)', 'Stress (1)']
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=classes, yticklabels=classes,
               ylabel='Actual', xlabel='Predicted',
               title=f'Confusion Matrix - {model_name}')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                text = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black',
                       fontsize=12, fontweight='bold')
        
        # Add metrics annotation
        tn, fp, fn, tp = cm.ravel()
        metrics_text = (
            f"TP={tp} (Hit)\n"
            f"TN={tn} (Correct Rejection)\n"
            f"FP={fp} (False Alarm)\n"
            f"FN={fn} (Miss)"
        )
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_confusion_matrix.png", dpi=150, 
                   bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {figures_dir / f'{model_name}_confusion_matrix.png'}")
    except Exception as e:
        print(f"Warning: Could not plot confusion matrix: {e}")


def plot_results(y_true: np.ndarray,
                 y_proba: np.ndarray,
                 model_name: str,
                 output_dir: Path,
                 y_pred: np.ndarray = None) -> None:
    """
    Generate and save evaluation plots.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        output_dir: Directory to save plots
        y_pred: Predicted labels (for confusion matrix)
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix (if y_pred provided)
    if y_pred is not None:
        plot_confusion_matrix(y_true, y_pred, model_name, output_dir)
    
    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_roc.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot ROC curve: {e}")
    
    # Precision-Recall curve
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_pr.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot PR curve: {e}")


def plot_all_confusion_matrices(all_results: Dict[str, Dict],
                                output_dir: Path) -> None:
    """
    Generate a combined confusion matrix plot for all models.
    
    Args:
        all_results: Dictionary mapping model name to results dict containing y_true, y_pred
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    n_models = len(all_results)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, result) in zip(axes, all_results.items()):
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_pct = cm.astype('float') / cm.sum() * 100
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Labels
        classes = ['No Stress', 'Stress']
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=classes, yticklabels=classes,
               ylabel='Actual', xlabel='Predicted',
               title=model_name.replace('_', ' ').title())
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                text = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black',
                       fontsize=10, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "all_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined confusion matrices to {figures_dir / 'all_confusion_matrices.png'}")


def compare_models(results: Dict[str, Dict],
                   output_dir: Path) -> pd.DataFrame:
    """
    Compare results across multiple models.
    
    Args:
        results: Dictionary mapping model name to metrics
        output_dir: Directory to save comparison
    
    Returns:
        DataFrame with model comparison
    """
    rows = []
    
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        
        for key in ["auroc", "pr_auc", "f1", "precision", "recall", "accuracy"]:
            # Check for aggregated mean values first
            if f"{key}_mean" in metrics:
                row[key.upper()] = metrics[f"{key}_mean"]
            elif key in metrics:
                row[key.upper()] = metrics[key]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save comparison
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create comparison bar plot
    try:
        metrics_to_plot = ["AUROC", "PR_AUC", "F1"]
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if available_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df))
            width = 0.25
            
            for i, metric in enumerate(available_metrics):
                ax.bar(x + i * width, df[metric], width, label=metric)
            
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.set_title("Model Comparison")
            ax.set_xticks(x + width)
            ax.set_xticklabels(df["Model"])
            ax.legend()
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"Warning: Could not create comparison plot: {e}")
    
    return df


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation utilities...")
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = evaluate_predictions(y_true, y_pred, y_proba, "test_model")
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

