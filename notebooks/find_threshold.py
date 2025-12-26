#!/usr/bin/env python3
"""Simple threshold optimization without visualization"""

import pandas as pd
import numpy as np
import json

def calculate_metrics(y_true, y_pred):
    """Calculate key metrics."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    gmean = np.sqrt(recall * specificity)
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Recall': recall,
        'Specificity': specificity,
        'Precision': precision,
        'FAR': far,
        'GMean': gmean
    }

# Load data
df = pd.read_csv('../experiments/01_classical_ml/results/logistic_regression_predictions.csv')
y_true = df['y_true'].values
y_proba = df['y_proba'].values

print(f"Total samples: {len(df)}")
print(f"Stress: {(y_true==1).sum()} | Non-stress: {(y_true==0).sum()}\n")

# Threshold sweep
print("Sweeping thresholds...")
thresholds = np.arange(0.01, 1.00, 0.01)
results = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    m = calculate_metrics(y_true, y_pred)
    m['Threshold'] = t
    results.append(m)

results_df = pd.DataFrame(results)

# Find optimal
target_recall = 0.80
max_far = 0.20

feasible = results_df[(results_df['Recall'] >= target_recall) & (results_df['FAR'] <= max_far)]

print("="*70)
print(f"TARGET: Recall ≥ {target_recall*100:.0f}%, FAR ≤ {max_far*100:.0f}%")
print(f"Feasible thresholds: {len(feasible)}")
print("="*70)

if len(feasible) > 0:
    best = feasible.loc[feasible['GMean'].idxmax()]
    
    print(f"\n🎯 OPTIMAL THRESHOLD: {best['Threshold']:.4f}")
    print("="*70)
    print(f"Recall:      {best['Recall']*100:.2f}%")
    print(f"FAR:         {best['FAR']*100:.2f}%")
    print(f"Precision:   {best['Precision']*100:.2f}%")
    print(f"Specificity: {best['Specificity']*100:.2f}%")
    print(f"G-Mean:      {best['GMean']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {best['TP']:.0f}  |  FN: {best['FN']:.0f}")
    print(f"  FP: {best['FP']:.0f}  |  TN: {best['TN']:.0f}")
    print("="*70)
    
    # Practical interpretation
    print(f"\n📊 PRACTICAL INTERPRETATION")
    print("="*70)
    print(f"Out of 100 stress episodes, you catch:  {best['Recall']*100:.0f}")
    print(f"Out of 100 non-stress, false alarms:    {best['FAR']*100:.0f}")
    print(f"When alert fires, it's correct:          {best['Precision']*100:.0f}% of time")
    print("="*70)
    
    # Save config
    config = {
        'threshold': float(best['Threshold']),
        'target_recall': target_recall,
        'max_far': max_far,
        'achieved_recall': float(best['Recall']),
        'achieved_far': float(best['FAR']),
        'precision': float(best['Precision']),
        'specificity': float(best['Specificity']),
        'gmean': float(best['GMean']),
        'confusion_matrix': {
            'TP': int(best['TP']),
            'TN': int(best['TN']),
            'FP': int(best['FP']),
            'FN': int(best['FN'])
        },
        'validation_samples': len(df),
        'stress_samples': int((y_true==1).sum()),
        'notes': 'Optimized for 80% recall with max 20% false alarm rate'
    }
    
    with open('../experiments/01_classical_ml/results/optimal_threshold_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Config saved: optimal_threshold_config.json")
    
    # Save full results
    results_df.to_csv('../experiments/01_classical_ml/results/threshold_sweep_results.csv', index=False)
    print(f"✓ Full results saved: threshold_sweep_results.csv")
    
else:
    print("\n⚠️  No threshold satisfies both constraints!")
    
    # Show what's achievable
    far_constrained = results_df[results_df['FAR'] <= max_far]
    if len(far_constrained) > 0:
        best_recall = far_constrained.loc[far_constrained['Recall'].idxmax()]
        print(f"\nBest recall @ FAR≤20%: {best_recall['Recall']*100:.1f}% (threshold={best_recall['Threshold']:.3f})")
    
    recall_constrained = results_df[results_df['Recall'] >= target_recall]
    if len(recall_constrained) > 0:
        best_far = recall_constrained.loc[recall_constrained['FAR'].idxmin()]
        print(f"Lowest FAR @ Recall≥80%: {best_far['FAR']*100:.1f}% (threshold={best_far['Threshold']:.3f})")

print("\n✓ COMPLETE")

