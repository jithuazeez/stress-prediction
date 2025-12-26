#!/usr/bin/env python3
"""
Threshold Optimization for Stress Detection
Find optimal threshold for 80% Recall with ≤20% FAR
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def calculate_metrics(y_true, y_pred):
    """Calculate key metrics for threshold analysis."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    gmean = np.sqrt(sensitivity * specificity)
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'False Alarm Rate': far,
        'G-Mean': gmean
    }

def main():
    # Load predictions
    print("Loading predictions...")
    predictions_path = '../experiments/01_classical_ml/results/logistic_regression_predictions.csv'
    df = pd.read_csv(predictions_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Stress samples (y=1): {(df['y_true'] == 1).sum()}")
    print(f"Non-stress samples (y=0): {(df['y_true'] == 0).sum()}")
    print(f"Class distribution: {(df['y_true'] == 1).sum() / len(df) * 100:.1f}% stress\n")
    
    # Extract labels and probabilities
    y_true = df['y_true'].values
    y_proba = df['y_proba'].values
    
    # Threshold sweep
    print("Performing threshold sweep...")
    thresholds = np.arange(0.01, 1.00, 0.01)
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred_thresh)
        metrics['Threshold'] = threshold
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    print(f"✓ Tested {len(thresholds)} thresholds\n")
    
    # Find optimal threshold
    target_recall = 0.80
    max_far = 0.20
    
    meets_recall = results_df['Sensitivity (Recall)'] >= target_recall
    meets_far = results_df['False Alarm Rate'] <= max_far
    feasible = results_df[meets_recall & meets_far].copy()
    
    print("="*70)
    print("CONSTRAINT-BASED THRESHOLD OPTIMIZATION")
    print("="*70)
    print(f"Target: Recall ≥ {target_recall*100:.0f}% AND FAR ≤ {max_far*100:.0f}%")
    print(f"Feasible thresholds found: {len(feasible)}\n")
    
    if len(feasible) > 0:
        # Pick threshold with best G-Mean
        best_idx = feasible['G-Mean'].idxmax()
        best_threshold = feasible.loc[best_idx]
        
        print("="*70)
        print(f"🎯 OPTIMAL THRESHOLD: {best_threshold['Threshold']:.4f}")
        print("="*70)
        print(f"Recall (Sensitivity):    {best_threshold['Sensitivity (Recall)']*100:.2f}%")
        print(f"False Alarm Rate:        {best_threshold['False Alarm Rate']*100:.2f}%")
        print(f"Specificity:             {best_threshold['Specificity']*100:.2f}%")
        print(f"Precision:               {best_threshold['Precision']*100:.2f}%")
        print(f"G-Mean:                  {best_threshold['G-Mean']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {best_threshold['TP']:.0f}  |  FN: {best_threshold['FN']:.0f}")
        print(f"  FP: {best_threshold['FP']:.0f}  |  TN: {best_threshold['TN']:.0f}")
        print("="*70)
        print(f"\nFeasible threshold range: [{feasible['Threshold'].min():.3f}, {feasible['Threshold'].max():.3f}]")
        
        # Save configuration
        optimal_config = {
            'threshold': float(best_threshold['Threshold']),
            'target_recall': target_recall,
            'max_far': max_far,
            'achieved_recall': float(best_threshold['Sensitivity (Recall)']),
            'achieved_far': float(best_threshold['False Alarm Rate']),
            'precision': float(best_threshold['Precision']),
            'specificity': float(best_threshold['Specificity']),
            'gmean': float(best_threshold['G-Mean']),
            'confusion_matrix': {
                'TP': int(best_threshold['TP']),
                'TN': int(best_threshold['TN']),
                'FP': int(best_threshold['FP']),
                'FN': int(best_threshold['FN'])
            },
            'validation_samples': len(df),
            'stress_samples': int((df['y_true']==1).sum()),
            'notes': 'Optimized for 80% recall with maximum 20% false alarm rate'
        }
        
        config_path = '../experiments/01_classical_ml/results/optimal_threshold_config.json'
        with open(config_path, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        
        print(f"\n✓ Configuration saved to: {config_path}")
        
        # Practical interpretation
        print("\n" + "="*70)
        print("📊 PRACTICAL INTERPRETATION")
        print("="*70)
        print(f"Out of 100 stress episodes:")
        print(f"  ✓ You will catch: {best_threshold['Sensitivity (Recall)']*100:.0f}")
        print(f"  ✗ You will miss: {(1-best_threshold['Sensitivity (Recall)'])*100:.0f}")
        print(f"\nOut of 100 non-stress periods:")
        print(f"  ✓ Correctly identified: {(1-best_threshold['False Alarm Rate'])*100:.0f}")
        print(f"  ✗ False alarms: {best_threshold['False Alarm Rate']*100:.0f}")
        print(f"\nWhen system alerts 'stress':")
        print(f"  ✓ It's correct {best_threshold['Precision']*100:.0f}% of the time")
        print("="*70)
        
    else:
        print("\n⚠️  WARNING: No threshold satisfies BOTH constraints!")
        print("\nWhat's achievable:")
        
        within_far = results_df[meets_far]
        if len(within_far) > 0:
            best_recall = within_far['Sensitivity (Recall)'].max()
            best_recall_row = within_far.loc[within_far['Sensitivity (Recall)'].idxmax()]
            print(f"  Best recall with FAR ≤ 20%: {best_recall*100:.1f}% (threshold={best_recall_row['Threshold']:.3f})")
        
        within_recall = results_df[meets_recall]
        if len(within_recall) > 0:
            best_far = within_recall['False Alarm Rate'].min()
            best_far_row = within_recall.loc[within_recall['False Alarm Rate'].idxmin()]
            print(f"  Lowest FAR with recall ≥ 80%: {best_far*100:.1f}% (threshold={best_far_row['Threshold']:.3f})")
        
        best_threshold = None
    
    # Visualizations
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Recall vs FAR
    ax = axes[0, 0]
    ax.plot(results_df['Threshold'], results_df['Sensitivity (Recall)'], label='Recall', lw=2, color='#2ecc71')
    ax.plot(results_df['Threshold'], results_df['False Alarm Rate'], label='FAR', lw=2, color='#e74c3c')
    ax.axhline(0.80, color='#2ecc71', ls='--', alpha=0.5, label='Target Recall (80%)')
    ax.axhline(0.20, color='#e74c3c', ls='--', alpha=0.5, label='Max FAR (20%)')
    if len(feasible) > 0:
        ax.axvline(best_threshold['Threshold'], color='black', ls=':', lw=2, label=f"Optimal: {best_threshold['Threshold']:.3f}")
    ax.set_xlabel('Threshold'); ax.set_ylabel('Rate')
    ax.set_title('Recall vs False Alarm Rate', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1])
    
    # Plot 2: Precision vs Recall
    ax = axes[0, 1]
    ax.plot(results_df['Threshold'], results_df['Precision'], label='Precision', lw=2, color='#3498db')
    ax.plot(results_df['Threshold'], results_df['Sensitivity (Recall)'], label='Recall', lw=2, color='#2ecc71')
    ax.axhline(0.80, color='#2ecc71', ls='--', alpha=0.5)
    if len(feasible) > 0:
        ax.axvline(best_threshold['Threshold'], color='black', ls=':', lw=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
    ax.set_title('Precision-Recall Trade-off', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1])
    
    # Plot 3: G-Mean
    ax = axes[1, 0]
    ax.plot(results_df['Threshold'], results_df['G-Mean'], lw=2, color='#9b59b6', label='G-Mean')
    if len(feasible) > 0:
        feasible_gmean = results_df.loc[meets_recall & meets_far, 'G-Mean']
        feasible_thresh = results_df.loc[meets_recall & meets_far, 'Threshold']
        ax.fill_between(feasible_thresh, 0, feasible_gmean, alpha=0.3, color='green', label='Feasible Region')
        ax.axvline(best_threshold['Threshold'], color='black', ls=':', lw=2)
        ax.scatter([best_threshold['Threshold']], [best_threshold['G-Mean']], 
                  s=200, color='red', zorder=5, marker='*', edgecolors='black', lw=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('G-Mean')
    ax.set_title('G-Mean (Balance Metric)', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1])
    
    # Plot 4: Confusion Matrix Components
    ax = axes[1, 1]
    ax.plot(results_df['Threshold'], results_df['TP'], label='True Positives', lw=2)
    ax.plot(results_df['Threshold'], results_df['FN'], label='False Negatives', lw=2)
    ax.plot(results_df['Threshold'], results_df['FP'], label='False Positives', lw=2)
    if len(feasible) > 0:
        ax.axvline(best_threshold['Threshold'], color='black', ls=':', lw=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Count')
    ax.set_title('Confusion Matrix Components', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim([0,1])
    
    plt.tight_layout()
    output_path = '../experiments/01_classical_ml/results/figures/threshold_optimization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    
    print("\n" + "="*70)
    print("✓ THRESHOLD OPTIMIZATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()

