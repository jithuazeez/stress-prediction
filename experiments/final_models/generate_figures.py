"""
Generate all required visualizations for final models experiment.

Creates 6 figures:
1. Model comparison table (CSV)
2. Recall vs FAR trade-off plot
3. Subject-wise recall distribution
4. Normalized confusion matrices
5. Precision-Recall curves
6. Threshold sensitivity plots
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_model_results(model_name: str, strategy: str, results_dir: Path) -> Dict:
    """Load metrics and predictions for a model-strategy combination."""
    model_dir = results_dir / model_name
    
    # Load metrics
    metrics_file = model_dir / f"{model_name}_{strategy}_metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load fold metrics
    fold_file = model_dir / f"{model_name}_{strategy}_fold_metrics.csv"
    if fold_file.exists():
        fold_metrics = pd.read_csv(fold_file)
    else:
        fold_metrics = None
    
    # Load predictions
    pred_file = model_dir / f"{model_name}_{strategy}_predictions.csv"
    if pred_file.exists():
        predictions = pd.read_csv(pred_file)
    else:
        predictions = None
    
    return {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "predictions": predictions
    }


def generate_comparison_table(results_dir: Path, output_dir: Path):
    """Generate main comparison table for all models and strategies."""
    print("\n" + "="*60)
    print("FIGURE 1: Model Comparison Table")
    print("="*60)
    
    models = ["lr", "rf", "svm", "tcn", "or", "cascade", "stacked"]
    strategies = ["b1", "b2", "b3"]
    strategy_names = {
        "b1": "G-Mean",
        "b2": "Recall≥70%",
        "b3": "FAR≤30%"
    }
    
    rows = []
    
    for model in models:
        for strategy in strategies:
            result = load_model_results(model, strategy, results_dir)
            
            if result is None:
                print(f"  ⚠️  {model.upper()} ({strategy}) - not available")
                continue
            
            metrics = result["metrics"]
            
            row = {
                "Model": model.upper(),
                "Strategy": strategy_names[strategy],
                "Recall": f"{metrics.get('recall', 0.0):.3f} ± {metrics.get('recall_std', 0.0):.3f}",
                "FAR": f"{metrics.get('far', metrics.get('false_alarm_rate', 0.0)):.3f}",
                "Specificity": f"{metrics.get('specificity', 0.0):.3f}",
                "Precision": f"{metrics.get('precision', 0.0):.3f}",
                "G-mean": f"{metrics.get('gmean', 0.0):.3f}",
                "AUROC": f"{metrics.get('auroc', 0.0):.3f}",
                "PR-AUC": f"{metrics.get('pr_auc', 0.0):.3f}",
                "Bal.Acc": f"{metrics.get('balanced_accuracy', 0.0):.3f}"
            }
            rows.append(row)
            print(f"  ✓ {model.upper()} ({strategy}): Recall={metrics.get('recall', 0.0):.3f}, FAR={metrics.get('far', metrics.get('false_alarm_rate', 0.0)):.3f}")
    
    if len(rows) == 0:
        print("  ⚠️  No data available")
        return
    
    df = pd.DataFrame(rows)
    output_file = output_dir / "comparison_table.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")


def generate_recall_vs_far_plot(results_dir: Path, output_dir: Path):
    """Generate Recall vs FAR trade-off plot."""
    print("\n" + "="*60)
    print("FIGURE 2: Recall vs FAR Trade-off")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = ["lr", "rf", "svm", "tcn", "or", "cascade", "stacked"]
    model_names = {"lr": "Logistic Regression", "rf": "Random Forest", "svm": "SVM",
                   "tcn": "TCN", "or": "Logical OR", "cascade": "Cascade", "stacked": "Stacked"}
    strategies = ["b1", "b2", "b3"]
    strategy_markers = {"b1": "o", "b2": "s", "b3": "^"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    plotted_models = []
    
    for idx, model in enumerate(models):
        fars = []
        recalls = []
        strategy_labels = []
        
        for strategy in strategies:
            result = load_model_results(model, strategy, results_dir)
            if result is None:
                continue
            
            metrics = result["metrics"]
            far = metrics.get('far', metrics.get('false_alarm_rate', 0.0))
            recall = metrics.get('recall', 0.0)
            
            fars.append(far)
            recalls.append(recall)
            strategy_labels.append(strategy)
        
        if len(fars) > 0:
            # Plot line connecting points
            ax.plot(fars, recalls, '-', color=colors[idx], alpha=0.3, linewidth=2)
            
            # Plot points with markers
            for far, rec, strat in zip(fars, recalls, strategy_labels):
                ax.plot(far, rec, marker=strategy_markers[strat], color=colors[idx],
                       markersize=10, label=model_names[model] if strat == "b1" else "")
            
            plotted_models.append(model)
            print(f"  ✓ {model_names[model]}: {len(fars)} strategies")
    
    if len(plotted_models) == 0:
        print("  ⚠️  No data available")
        plt.close()
        return
    
    # Formatting
    ax.set_xlabel('False Alarm Rate (FAR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title('Recall vs FAR Trade-off\n(Lower-right is better: High Recall, Low FAR)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=9)
    
    # Add strategy markers legend
    strategy_legend = [plt.Line2D([0], [0], marker=strategy_markers[s], color='gray',
                                   linestyle='', markersize=8, label=s.upper())
                      for s in strategies]
    ax.add_artist(ax.legend(handles=strategy_legend, loc='upper left', 
                           title='Threshold Strategy', fontsize=8))
    
    plt.tight_layout()
    output_file = output_dir / "recall_vs_far_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def generate_subject_recall_distribution(results_dir: Path, output_dir: Path):
    """Generate subject-wise recall distribution boxplots."""
    print("\n" + "="*60)
    print("FIGURE 3: Subject-wise Recall Distribution")
    print("="*60)
    
    models = ["lr", "tcn", "or", "cascade", "stacked"]
    model_names = {"lr": "LR", "tcn": "TCN", "or": "OR", "cascade": "Cascade", "stacked": "Stacked"}
    strategy = "b1"  # Use B1 (unconstrained) for comparison
    
    data_for_plot = []
    
    for model in models:
        result = load_model_results(model, strategy, results_dir)
        if result is None or result["fold_metrics"] is None:
            continue
        
        fold_metrics = result["fold_metrics"]
        if "recall" in fold_metrics.columns:
            for recall_val in fold_metrics["recall"]:
                data_for_plot.append({
                    "Model": model_names[model],
                    "Recall": recall_val
                })
            print(f"  ✓ {model_names[model]}: {len(fold_metrics)} subjects")
    
    if len(data_for_plot) == 0:
        print("  ⚠️  No fold-level data available")
        return
    
    df = pd.DataFrame(data_for_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Model", y="Recall", ax=ax, palette="Set2")
    sns.swarmplot(data=df, x="Model", y="Recall", ax=ax, color='black', alpha=0.5, size=4)
    
    ax.set_ylabel('Recall (per subject)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Subject-wise Recall Distribution (B1 - G-Mean Threshold)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    output_file = output_dir / "subject_recall_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def generate_confusion_matrices(results_dir: Path, output_dir: Path):
    """Generate normalized confusion matrices for key models."""
    print("\n" + "="*60)
    print("FIGURE 4: Normalized Confusion Matrices")
    print("="*60)
    
    models = ["lr", "tcn", "or", "cascade", "stacked"]
    model_names = {"lr": "Logistic Regression", "tcn": "TCN", "or": "Logical OR",
                   "cascade": "Cascade", "stacked": "Stacked"}
    strategy = "b1"
    
    n_models = len([m for m in models if load_model_results(m, strategy, results_dir) is not None])
    if n_models == 0:
        print("  ⚠️  No data available")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 3.5))
    if n_models == 1:
        axes = [axes]
    
    plot_idx = 0
    
    for model in models:
        result = load_model_results(model, strategy, results_dir)
        if result is None or result["predictions"] is None:
            continue
        
        predictions = result["predictions"]
        y_true = predictions["y_true"].values
        y_pred = predictions["y_pred"].values
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        ax = axes[plot_idx]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['No Stress', 'Stress'],
                   yticklabels=['No Stress', 'Stress'],
                   ax=ax, cbar=True, vmin=0, vmax=1)
        
        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_names[model]}\n(B1 - G-Mean)', fontsize=11, fontweight='bold')
        
        # Add counts as text
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.7, f'n={cm[i,j]}',
                       ha='center', va='center', fontsize=8, color='gray')
        
        plot_idx += 1
        print(f"  ✓ {model_names[model]}")
    
    plt.tight_layout()
    output_file = output_dir / "confusion_matrices.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def generate_precision_recall_curves(results_dir: Path, output_dir: Path):
    """Generate Precision-Recall curves for key models."""
    print("\n" + "="*60)
    print("FIGURE 5: Precision-Recall Curves")
    print("="*60)
    
    models = ["lr", "tcn", "or", "cascade", "stacked"]
    model_names = {"lr": "Logistic Regression", "tcn": "TCN", "or": "Logical OR",
                   "cascade": "Cascade", "stacked": "Stacked"}
    strategy = "b1"
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plotted = False
    
    for idx, model in enumerate(models):
        result = load_model_results(model, strategy, results_dir)
        if result is None or result["predictions"] is None:
            continue
        
        predictions = result["predictions"]
        y_true = predictions["y_true"].values
        y_proba = predictions["y_proba"].values
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = result["metrics"].get("pr_auc", 0.0)
        
        ax.plot(recall, precision, color=colors[idx], linewidth=2,
               label=f'{model_names[model]} (AUC={pr_auc:.3f})')
        plotted = True
        print(f"  ✓ {model_names[model]}: PR-AUC={pr_auc:.3f}")
    
    if not plotted:
        print("  ⚠️  No data available")
        plt.close()
        return
    
    # Baseline (random classifier)
    ax.axhline(y=0.12, color='gray', linestyle='--', linewidth=1, label='Baseline (12% positive)')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves\n(Higher curve = Better performance)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "precision_recall_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def generate_threshold_sensitivity_plots(results_dir: Path, output_dir: Path):
    """Generate threshold sensitivity plots for each model."""
    print("\n" + "="*60)
    print("FIGURE 6: Threshold Sensitivity Plots")
    print("="*60)
    
    models = ["lr", "tcn", "or", "cascade", "stacked"]
    model_names = {"lr": "Logistic Regression", "tcn": "TCN", "or": "Logical OR",
                   "cascade": "Cascade", "stacked": "Stacked"}
    
    for model in models:
        result = load_model_results(model, "b1", results_dir)
        if result is None or result["predictions"] is None:
            continue
        
        predictions = result["predictions"]
        y_true = predictions["y_true"].values
        y_proba = predictions["y_proba"].values
        
        # Generate threshold range
        thresholds = np.linspace(0, 1, 101)
        recalls = []
        fars = []
        gmeans = []
        
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(recall * specificity)
            
            recalls.append(recall)
            fars.append(far)
            gmeans.append(gmean)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall')
        ax.plot(thresholds, fars, 'r-', linewidth=2, label='FAR')
        ax.plot(thresholds, gmeans, 'g-', linewidth=2, label='G-mean')
        
        # Mark operating points from B1, B2, B3
        strategies = ["b1", "b2", "b3"]
        strategy_names = {"b1": "B1 (G-Mean)", "b2": "B2 (Recall≥70%)", "b3": "B3 (FAR≤30%)"}
        markers = {"b1": "o", "b2": "s", "b3": "^"}
        
        for strategy in strategies:
            result_strat = load_model_results(model, strategy, results_dir)
            if result_strat is None:
                continue
            
            metrics = result_strat["metrics"]
            recall = metrics.get("recall", 0.0)
            far = metrics.get("far", metrics.get("false_alarm_rate", 0.0))
            
            # Find closest threshold (approximation)
            fold_metrics = result_strat.get("fold_metrics")
            if fold_metrics is not None and "optimal_threshold" in fold_metrics.columns:
                thr_val = fold_metrics["optimal_threshold"].mean()
            else:
                thr_val = 0.5
            
            ax.plot(thr_val, recall, marker=markers[strategy], markersize=10,
                   color='blue', label=strategy_names[strategy])
        
        ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Threshold Sensitivity: {model_names[model]}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        output_file = output_dir / f"threshold_sensitivity_{model}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {model_names[model]}")
    
    print(f"  Saved: {output_dir}/threshold_sensitivity_*.png")


def main():
    """Generate all figures."""
    results_dir = Path(__file__).parent / "results"
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("GENERATING ALL FIGURES")
    print("="*60)
    
    # Generate all figures
    generate_comparison_table(results_dir, comparison_dir)
    generate_recall_vs_far_plot(results_dir, output_dir)
    generate_subject_recall_distribution(results_dir, output_dir)
    generate_confusion_matrices(results_dir, output_dir)
    generate_precision_recall_curves(results_dir, output_dir)
    generate_threshold_sensitivity_plots(results_dir, output_dir)
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print("="*60)
    print(f"\nFigures saved in: {output_dir}/")
    print(f"Tables saved in: {comparison_dir}/")


if __name__ == "__main__":
    main()

