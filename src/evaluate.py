"""
Evaluation framework for stress prediction models.

Includes metrics, visualization, and LOSO cross-validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve
from pathlib import Path
import json


def evaluate_predictions(y_true, y_pred, y_proba, model_name="Model"):
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name of the model
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'model_name': model_name,
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(len(y_true) - np.sum(y_true)),
    }
    
    # Classification metrics
    metrics['auroc'] = float(roc_auc_score(y_true, y_proba))
    metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['brier_score'] = float(brier_score_loss(y_true, y_proba))
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # Operational metrics
    metrics['false_alarm_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics['detection_rate'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall']
    
    # Precision at different recall thresholds
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    for target_recall in [0.7, 0.8, 0.9]:
        idx = np.where(recall_vals >= target_recall)[0]
        if len(idx) > 0:
            metrics[f'precision_at_recall_{int(target_recall*100)}'] = float(precision_vals[idx[0]])
        else:
            metrics[f'precision_at_recall_{int(target_recall*100)}'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Stress', 'Stress'],
                yticklabels=['No Stress', 'Stress'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_proba, model_name, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUROC = {auroc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_proba, model_name, save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(y_true, y_proba, model_name, n_bins=10, save_path=None):
    """Plot calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    brier = brier_score_loss(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
             label=f'{model_name} (Brier = {brier:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(importance_df, model_name, top_n=20, save_path=None):
    """Plot feature importance."""
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models.
    
    Args:
        results_dict: Dictionary of model_name -> metrics_dict
        save_path: Optional path to save comparison plot
    """
    # Extract metrics for comparison
    models = list(results_dict.keys())
    metrics_to_compare = ['auroc', 'pr_auc', 'f1', 'precision', 'recall']
    
    # Create DataFrame
    data = []
    for model in models:
        row = {'Model': model}
        for metric in metrics_to_compare:
            row[metric.upper()] = results_dict[model].get(metric, 0)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(18, 4))
    
    for idx, metric in enumerate(metrics_to_compare):
        ax = axes[idx]
        ax.bar(df['Model'], df[metric.upper()])
        ax.set_title(metric.upper())
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return df


def save_results(metrics, output_dir, model_name):
    """Save evaluation results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_path / f"{model_name.replace(' ', '_').lower()}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics to {metrics_file}")


def print_evaluation_report(metrics):
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {metrics['model_name']}")
    print(f"{'='*60}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {metrics['n_samples']}")
    print(f"  Positive samples: {metrics['n_positive']} ({metrics['n_positive']/metrics['n_samples']*100:.1f}%)")
    print(f"  Negative samples: {metrics['n_negative']} ({metrics['n_negative']/metrics['n_samples']*100:.1f}%)")
    
    print(f"\nClassification Metrics:")
    print(f"  AUROC:          {metrics['auroc']:.4f}")
    print(f"  PR-AUC:         {metrics['pr_auc']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1-Score:       {metrics['f1']:.4f}")
    print(f"  Brier Score:    {metrics['brier_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negative:  {metrics['true_negative']}")
    print(f"  False Positive: {metrics['false_positive']}")
    print(f"  False Negative: {metrics['false_negative']}")
    print(f"  True Positive:  {metrics['true_positive']}")
    
    print(f"\nOperational Metrics:")
    print(f"  Detection Rate:      {metrics['detection_rate']:.4f}")
    print(f"  False Alarm Rate:    {metrics['false_alarm_rate']:.4f}")
    print(f"  Specificity:         {metrics['specificity']:.4f}")
    print(f"  Sensitivity:         {metrics['sensitivity']:.4f}")
    
    print(f"\nPrecision at Recall Thresholds:")
    print(f"  Precision @ 70% Recall: {metrics.get('precision_at_recall_70', 0):.4f}")
    print(f"  Precision @ 80% Recall: {metrics.get('precision_at_recall_80', 0):.4f}")
    print(f"  Precision @ 90% Recall: {metrics.get('precision_at_recall_90', 0):.4f}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    # Test evaluation functions
    print("Testing evaluation framework...")
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = evaluate_predictions(y_true, y_pred, y_proba, "Test Model")
    print_evaluation_report(metrics)
    
    print("✓ Evaluation framework tested successfully!")








