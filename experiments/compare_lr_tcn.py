import pandas as pd
import numpy as np

# Load both fold metrics
lr_df = pd.read_csv('/Users/jithuazeez/Documents/Msc/Dissertation/results/baselines/logistic_regression_fold_metrics.csv')
tcn_df = pd.read_csv('/Users/jithuazeez/Documents/Msc/Dissertation/experiments/08_tcn/results/tcn_fold_metrics.csv')

# Remove "id_" prefix from LR subjects to match TCN format
lr_df['subject_clean'] = lr_df['subject'].str.replace('id_', '')
tcn_df['subject_clean'] = tcn_df['subject']

# Merge on subject
comparison = pd.merge(
    lr_df[['subject_clean', 'auroc', 'pr_auc', 'precision', 'recall', 'f1', 
           'specificity', 'false_alarm_rate']],
    tcn_df[['subject_clean', 'auroc', 'pr_auc', 'precision', 'recall', 'f1',
            'specificity', 'false_alarm_rate', 'gmean', 'threshold']],
    on='subject_clean',
    suffixes=('_lr', '_tcn')
)

# Calculate differences (TCN - LR)
comparison['auroc_diff'] = comparison['auroc_tcn'] - comparison['auroc_lr']
comparison['pr_auc_diff'] = comparison['pr_auc_tcn'] - comparison['pr_auc_lr']
comparison['f1_diff'] = comparison['f1_tcn'] - comparison['f1_lr']
comparison['recall_diff'] = comparison['recall_tcn'] - comparison['recall_lr']
comparison['specificity_diff'] = comparison['specificity_tcn'] - comparison['specificity_lr']
comparison['precision_diff'] = comparison['precision_tcn'] - comparison['precision_lr']

# Summary statistics
print("="*80)
print("TCN vs LOGISTIC REGRESSION - FOLD-BY-FOLD COMPARISON")
print("="*80)
print("\nSame dataset: 1,733 samples | 213 positive (12.3%) | 1,520 negative (87.7%)")
print("\n" + "="*80)
print("AVERAGE METRICS ACROSS 21 FOLDS")
print("="*80)

metrics = ['auroc', 'pr_auc', 'f1', 'recall', 'specificity', 'precision', 'false_alarm_rate']
print(f"\n{'Metric':<20} {'LR Mean':<12} {'TCN Mean':<12} {'Difference':<12} {'Winner':<10}")
print("-"*80)

for metric in metrics:
    lr_mean = comparison[f'{metric}_lr'].mean()
    tcn_mean = comparison[f'{metric}_tcn'].mean()
    diff = tcn_mean - lr_mean
    pct_change = (diff / lr_mean * 100) if lr_mean != 0 else 0
    winner = "TCN" if diff > 0 else "LR"
    
    print(f"{metric:<20} {lr_mean:.4f}      {tcn_mean:.4f}      "
          f"{diff:+.4f}      {winner:<10}")

# G-mean (only in TCN)
print(f"\nG-Mean (TCN only):   {comparison['gmean'].mean():.4f}")
print(f"Threshold (TCN):     {comparison['threshold'].mean():.4f} ± {comparison['threshold'].std():.4f}")

# Count wins per metric
print("\n" + "="*80)
print("FOLD-BY-FOLD WIN COUNT (out of 21 folds)")
print("="*80)

for metric in metrics:
    tcn_wins = (comparison[f'{metric}_diff'] > 0).sum()
    lr_wins = (comparison[f'{metric}_diff'] < 0).sum()
    ties = (comparison[f'{metric}_diff'] == 0).sum()
    print(f"{metric:<20} TCN: {tcn_wins:2d}  |  LR: {lr_wins:2d}  |  Ties: {ties:2d}")

# Significant differences
print("\n" + "="*80)
print("SUBJECTS WITH LARGEST DIFFERENCES")
print("="*80)

# AUROC improvements
print("\n📈 Top 5 AUROC Improvements (TCN > LR):")
top_auroc = comparison.nlargest(5, 'auroc_diff')[['subject_clean', 'auroc_lr', 'auroc_tcn', 'auroc_diff']]
for idx, row in top_auroc.iterrows():
    print(f"   {row['subject_clean'][:8]}: {row['auroc_lr']:.3f} → {row['auroc_tcn']:.3f} ({row['auroc_diff']:+.3f})")

print("\n📉 Top 5 AUROC Declines (TCN < LR):")
bottom_auroc = comparison.nsmallest(5, 'auroc_diff')[['subject_clean', 'auroc_lr', 'auroc_tcn', 'auroc_diff']]
for idx, row in bottom_auroc.iterrows():
    print(f"   {row['subject_clean'][:8]}: {row['auroc_lr']:.3f} → {row['auroc_tcn']:.3f} ({row['auroc_diff']:+.3f})")

# Standard deviations
print("\n" + "="*80)
print("CONSISTENCY (Standard Deviation Across Folds)")
print("="*80)

print(f"\n{'Metric':<20} {'LR Std':<12} {'TCN Std':<12} {'More Stable':<10}")
print("-"*80)

for metric in metrics:
    lr_std = comparison[f'{metric}_lr'].std()
    tcn_std = comparison[f'{metric}_tcn'].std()
    more_stable = "LR" if lr_std < tcn_std else "TCN"
    print(f"{metric:<20} {lr_std:.4f}      {tcn_std:.4f}      {more_stable:<10}")

print("\n" + "="*80)

# Statistical significance
from scipy import stats

print("\nSTATISTICAL SIGNIFICANCE (Paired t-test)")
print("="*80)

for metric in ['auroc', 'pr_auc', 'f1', 'recall', 'specificity']:
    t_stat, p_value = stats.ttest_rel(comparison[f'{metric}_tcn'], comparison[f'{metric}_lr'])
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"{metric:<20} t={t_stat:+.3f}, p={p_value:.4f} {sig}")

print("\n*** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

