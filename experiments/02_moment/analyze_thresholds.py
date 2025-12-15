"""
Analyze different decision thresholds on trained MOMENT model.

Loads saved predictions and tests various thresholds to find optimal balance
between precision and recall without retraining.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_predictions(results_dir: Path):
    """Load saved predictions from CSV."""
    pred_file = results_dir / "moment_predictions.csv"
    df = pd.read_csv(pred_file)
    
    y_true = df['y_true'].values
    y_proba = df['y_proba'].values
    subjects = df['subject_id'].values if 'subject_id' in df.columns else df.get('subject', None)
    
    return y_true, y_proba, subjects


def evaluate_threshold(y_true, y_proba, threshold):
    """Evaluate metrics at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    # Handle edge cases
    if len(np.unique(y_pred)) == 1:
        # All predictions are the same class
        precision = 0.0 if y_pred[0] == 1 else 0.0
        recall = 0.0 if np.sum(y_true) > 0 else 0.0
        f1 = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0
    }


def find_optimal_thresholds(y_true, y_proba):
    """Find optimal thresholds for different criteria."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    for thresh in thresholds:
        metrics = evaluate_threshold(y_true, y_proba, thresh)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Find optimal thresholds
    optimal = {}
    
    # Best F1 score
    best_f1_idx = df['f1'].idxmax()
    optimal['best_f1'] = df.loc[best_f1_idx, 'threshold']
    
    # Balanced (closest to precision=recall)
    df['precision_recall_diff'] = abs(df['precision'] - df['recall'])
    balanced_idx = df['precision_recall_diff'].idxmin()
    optimal['balanced'] = df.loc[balanced_idx, 'threshold']
    
    # High precision (precision > 0.4)
    high_prec = df[df['precision'] >= 0.4]
    if len(high_prec) > 0:
        optimal['high_precision'] = high_prec.loc[high_prec['f1'].idxmax(), 'threshold']
    else:
        optimal['high_precision'] = 0.7
    
    # High recall (recall > 0.7)
    high_rec = df[df['recall'] >= 0.7]
    if len(high_rec) > 0:
        optimal['high_recall'] = high_rec.loc[high_rec['f1'].idxmax(), 'threshold']
    else:
        optimal['high_recall'] = 0.3
    
    return df, optimal


def plot_threshold_analysis(df, optimal, output_dir: Path):
    """Plot threshold vs metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1 vs Threshold
    ax = axes[0, 0]
    ax.plot(df['threshold'], df['precision'], label='Precision', linewidth=2)
    ax.plot(df['threshold'], df['recall'], label='Recall', linewidth=2)
    ax.plot(df['threshold'], df['f1'], label='F1-Score', linewidth=2, linestyle='--')
    
    # Mark optimal thresholds
    for name, thresh in optimal.items():
        ax.axvline(thresh, color='red', alpha=0.3, linestyle=':')
        ax.text(thresh, 0.95, name.replace('_', '\n'), rotation=90, 
                verticalalignment='top', fontsize=8)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1 vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: False Positives and False Negatives
    ax = axes[0, 1]
    ax.plot(df['threshold'], df['fp'], label='False Positives', linewidth=2, color='orange')
    ax.plot(df['threshold'], df['fn'], label='False Negatives', linewidth=2, color='red')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    ax.set_title('False Positives & False Negatives vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: FPR and FNR
    ax = axes[1, 0]
    ax.plot(df['threshold'], df['fpr'], label='False Positive Rate', linewidth=2, color='orange')
    ax.plot(df['threshold'], df['fnr'], label='False Negative Rate', linewidth=2, color='red')
    ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='10% target')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('False Positive Rate & False Negative Rate vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Threshold
    ax = axes[1, 1]
    ax.plot(df['threshold'], df['accuracy'], linewidth=2, color='purple')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Threshold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved threshold analysis plot to: {output_dir / 'threshold_analysis.png'}")


def print_comparison_table(y_true, y_proba, thresholds_to_test):
    """Print comparison table for specific thresholds."""
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON TABLE")
    print("="*80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FP':<8} {'FN':<8}")
    print("-"*80)
    
    for thresh in thresholds_to_test:
        metrics = evaluate_threshold(y_true, y_proba, thresh)
        print(f"{thresh:<12.2f} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
              f"{metrics['f1']:<12.3f} {metrics['fp']:<8} {metrics['fn']:<8}")
    
    print("="*80)


def main():
    """Main analysis function."""
    results_dir = Path(__file__).parent / "results"
    
    print("\n" + "="*80)
    print("MOMENT MODEL - THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Load predictions
    print("\n📂 Loading saved predictions...")
    y_true, y_proba, subjects = load_predictions(results_dir)
    
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    
    print(f"   Total samples: {len(y_true)}")
    print(f"   Positive: {n_positive} ({100*n_positive/len(y_true):.1f}%)")
    print(f"   Negative: {n_negative} ({100*n_negative/len(y_true):.1f}%)")
    
    # Current performance (threshold = 0.5)
    print("\n" + "="*80)
    print("CURRENT PERFORMANCE (Threshold = 0.5)")
    print("="*80)
    current = evaluate_threshold(y_true, y_proba, 0.5)
    print(f"   Precision: {current['precision']:.4f}")
    print(f"   Recall:    {current['recall']:.4f}")
    print(f"   F1-Score:  {current['f1']:.4f}")
    print(f"   False Positives: {current['fp']} ({100*current['fpr']:.1f}%)")
    print(f"   False Negatives: {current['fn']} ({100*current['fnr']:.1f}%)")
    
    # Find optimal thresholds
    print("\n🔍 Analyzing thresholds from 0.1 to 0.9...")
    df, optimal = find_optimal_thresholds(y_true, y_proba)
    
    # Save results
    df.to_csv(results_dir / 'threshold_sweep.csv', index=False)
    print(f"✅ Saved threshold sweep to: {results_dir / 'threshold_sweep.csv'}")
    
    # Print optimal thresholds
    print("\n" + "="*80)
    print("RECOMMENDED OPTIMAL THRESHOLDS")
    print("="*80)
    
    for name, thresh in optimal.items():
        metrics = evaluate_threshold(y_true, y_proba, thresh)
        print(f"\n🎯 {name.upper().replace('_', ' ')} (threshold = {thresh:.2f})")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    # Test specific thresholds
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    print_comparison_table(y_true, y_proba, thresholds_to_test)
    
    # Plot analysis
    print("\n📊 Generating threshold analysis plots...")
    plot_threshold_analysis(df, optimal, results_dir)
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    
    best_f1_metrics = evaluate_threshold(y_true, y_proba, optimal['best_f1'])
    
    if best_f1_metrics['precision'] < 0.3:
        print(f"⚠️  Current model has too many false alarms (precision = {best_f1_metrics['precision']:.3f})")
        print(f"   Try threshold = {optimal['high_precision']:.2f} to reduce false positives")
        print(f"   Or retrain with lower class weight (3.0 instead of 6.8)")
    elif best_f1_metrics['recall'] < 0.5:
        print(f"⚠️  Current model misses too many stress cases (recall = {best_f1_metrics['recall']:.3f})")
        print(f"   Try threshold = {optimal['high_recall']:.2f} to catch more stress")
    else:
        print(f"✅ Use threshold = {optimal['best_f1']:.2f} for best F1-score")
        print(f"   This gives precision = {best_f1_metrics['precision']:.3f}, recall = {best_f1_metrics['recall']:.3f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

