"""
Generate confusion matrix visualizations for all models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

# Model names and their result files
models = {
    'LR': 'lr/lr_b1_metrics.json',
    'RF': 'rf/rf_b1_metrics.json',
    'SVM': 'svm/svm_b1_metrics.json',
    'TCN': 'tcn/tcn_b1_metrics.json',
    'Stacked': 'stacked/stacked_b1_metrics.json'
}

# Load metrics
results_dir = Path('results')
metrics_data = {}

for model_name, metrics_file in models.items():
    with open(results_dir / metrics_file, 'r') as f:
        metrics_data[model_name] = json.load(f)

# Create figure with 5 subplots
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

for idx, (model_name, metrics) in enumerate(metrics_data.items()):
    ax = axes[idx]
    
    # Extract confusion matrix values
    tn = metrics['true_negative']
    fp = metrics['false_positive']
    fn = metrics['false_negative']
    tp = metrics['true_positive']
    
    # Create confusion matrix (normalized by true class)
    n_negative = tn + fp
    n_positive = fn + tp
    
    conf_matrix_norm = np.array([
        [tn / n_negative, fp / n_negative],
        [fn / n_positive, tp / n_positive]
    ])
    
    # Raw counts for annotation
    conf_matrix_raw = np.array([
        [tn, fp],
        [fn, tp]
    ])
    
    # Create heatmap
    sns.heatmap(
        conf_matrix_norm,
        annot=False,  # We'll add custom annotations
        fmt='.2f',
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar=True,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Proportion'}
    )
    
    # Add custom annotations (proportion + count)
    for i in range(2):
        for j in range(2):
            text = f'{conf_matrix_norm[i, j]:.2f}\nn={conf_matrix_raw[i, j]}'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color='white' if conf_matrix_norm[i, j] > 0.5 else 'black',
                   fontsize=10, weight='bold')
    
    # Labels
    ax.set_xlabel('Predicted Label', fontsize=11)
    if idx == 0:
        ax.set_ylabel('True Label', fontsize=11)
    else:
        ax.set_ylabel('')
    
    ax.set_xticklabels(['No Stress', 'Stress'], fontsize=10)
    ax.set_yticklabels(['No Stress', 'Stress'], fontsize=10, rotation=0)
    
    # Title
    title = f'{model_name}\n(B1 - G-Mean)'
    ax.set_title(title, fontsize=12, weight='bold', pad=10)

plt.tight_layout()

# Save figure
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'confusion_matrices_all_models.pdf', bbox_inches='tight')

print("✓ Confusion matrices saved to results/figures/")
print(f"  - PNG: {output_dir / 'confusion_matrices_all_models.png'}")
print(f"  - PDF: {output_dir / 'confusion_matrices_all_models.pdf'}")

# Also print summary stats
print("\nConfusion Matrix Summary:")
print("-" * 70)
print(f"{'Model':<12} {'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6} {'Recall':<8} {'FAR':<8}")
print("-" * 70)

for model_name, metrics in metrics_data.items():
    tn = metrics['true_negative']
    fp = metrics['false_positive']
    fn = metrics['false_negative']
    tp = metrics['true_positive']
    recall = metrics['recall']
    far = metrics['far']
    
    print(f"{model_name:<12} {tn:<6} {fp:<6} {fn:<6} {tp:<6} {recall:<8.3f} {far:<8.3f}")

plt.close()
