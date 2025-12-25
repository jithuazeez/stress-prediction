"""
Compare results from all experiments.

Aggregates metrics, creates comparison tables and visualizations.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.logging_utils import setup_logger, log_experiment_start, log_experiment_end


# Matplotlib for plots
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_experiment_results(experiment_dir: Path, 
                            model_name: str,
                            logger) -> Dict:
    """Load results from an experiment directory."""
    results_dir = experiment_dir / "results"
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return None
    
    # Try to load metrics JSON
    metrics_file = results_dir / f"{model_name}_metrics.json"
    
    if not metrics_file.exists():
        # Try without model prefix
        possible_files = list(results_dir.glob("*_metrics.json"))
        if possible_files:
            metrics_file = possible_files[0]
        else:
            logger.warning(f"No metrics file found in: {results_dir}")
            return None
    
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        logger.debug(f"Loaded metrics from: {metrics_file}")
        return metrics
    except Exception as e:
        logger.warning(f"Failed to load {metrics_file}: {e}")
        return None


def collect_all_results(base_dir: Path, logger) -> Dict[str, Dict]:
    """Collect results from all experiments."""
    experiments = {
        "Classical ML (XGBoost)": ("01_classical_ml", "xgboost"),
        "Classical ML (RF)": ("01_classical_ml", "random_forest"),
        "Classical ML (LogReg)": ("01_classical_ml", "logistic_regression"),
        "MOMENT": ("02_moment", "moment"),
        "TS2Vec": ("03_ts2vec", "ts2vec"),
        "MAML": ("04_maml", "maml"),
        "TabPFN": ("07_tabpfn", "tabpfn"),
    }
    
    results = {}
    
    logger.info("Loading experiment results...")
    
    pbar = tqdm(experiments.items(), desc="Loading results", unit="model")
    
    for display_name, (subdir, model_name) in pbar:
        pbar.set_postfix({"Model": display_name[:15]})
        
        experiment_dir = base_dir / subdir
        metrics = load_experiment_results(experiment_dir, model_name, logger)
        
        if metrics:
            results[display_name] = metrics
            logger.info(f"  ✓ {display_name}: Loaded")
        else:
            logger.warning(f"  ✗ {display_name}: Not found")
    
    pbar.close()
    
    return results


def create_comparison_table(results: Dict[str, Dict], logger) -> pd.DataFrame:
    """Create a comparison table of all models."""
    rows = []
    
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "AUROC": metrics.get("auroc", float("nan")),
            "PR-AUC": metrics.get("pr_auc", float("nan")),
            "F1": metrics.get("f1", float("nan")),
            "Accuracy": metrics.get("accuracy", float("nan")),
            "Precision": metrics.get("precision", float("nan")),
            "Recall": metrics.get("recall", float("nan")),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by AUROC
    df = df.sort_values("AUROC", ascending=False)
    
    return df


def create_comparison_plot(results: Dict[str, Dict], 
                           output_path: Path,
                           logger):
    """Create bar chart comparing all models."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plots")
        return
    
    models = list(results.keys())
    metrics = ["auroc", "pr_auc", "f1", "accuracy"]
    metric_labels = ["AUROC", "PR-AUC", "F1", "Accuracy"]
    
    n_models = len(models)
    n_metrics = len(metrics)
    
    if n_models == 0:
        logger.warning("No models to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_models)
    width = 0.2
    
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[model].get(metric, 0) for model in models]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved comparison plot to: {output_path}")


def create_radar_plot(results: Dict[str, Dict],
                      output_path: Path,
                      logger):
    """Create radar/spider chart comparing models."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    metrics = ["auroc", "pr_auc", "f1", "precision", "recall"]
    metric_labels = ["AUROC", "PR-AUC", "F1", "Precision", "Recall"]
    
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (model_name, model_metrics), color in zip(results.items(), colors):
        values = [model_metrics.get(m, 0) for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved radar plot to: {output_path}")


def main():
    """Main comparison pipeline."""
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / "comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("compare_all", log_file=output_dir / "comparison.log")
    
    log_experiment_start(logger, "MULTI-MODEL COMPARISON")
    
    # Collect all results
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Collecting experiment results")
    logger.info("-"*50)
    
    results = collect_all_results(base_dir, logger)
    
    if not results:
        logger.error("No experiment results found!")
        logger.error("Run individual experiments first:")
        logger.error("  python experiments/01_classical_ml/train.py")
        logger.error("  python experiments/02_moment/train.py")
        logger.error("  python experiments/03_ts2vec/pretrain.py && python experiments/03_ts2vec/train_classifier.py")
        logger.error("  python experiments/04_maml/train.py")
        return
    
    logger.info(f"\nLoaded results from {len(results)} models")
    
    # Create comparison table
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Creating comparison table")
    logger.info("-"*50)
    
    comparison_df = create_comparison_table(results, logger)
    
    # Display table
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*70)
    
    # Format for display
    display_df = comparison_df.copy()
    for col in ["AUROC", "PR-AUC", "F1", "Accuracy", "Precision", "Recall"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    
    logger.info("\n" + display_df.to_string(index=False))
    
    # Save table
    table_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(table_path, index=False)
    logger.info(f"\nSaved comparison table to: {table_path}")
    
    # Best model
    best_idx = comparison_df["AUROC"].idxmax()
    best_model = comparison_df.loc[best_idx, "Model"]
    best_auroc = comparison_df.loc[best_idx, "AUROC"]
    
    logger.info("\n" + "-"*50)
    logger.info(f"BEST MODEL: {best_model}")
    logger.info(f"  AUROC: {best_auroc:.4f}")
    logger.info("-"*50)
    
    # Create visualizations
    logger.info("\n" + "-"*50)
    logger.info("PHASE 3: Creating visualizations")
    logger.info("-"*50)
    
    create_comparison_plot(results, output_dir / "comparison_bar.png", logger)
    create_radar_plot(results, output_dir / "comparison_radar.png", logger)
    
    # Summary statistics
    logger.info("\n" + "-"*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("-"*50)
    
    for metric in ["AUROC", "PR-AUC", "F1"]:
        values = comparison_df[metric].dropna()
        if len(values) > 0:
            logger.info(f"{metric}:")
            logger.info(f"  Mean: {values.mean():.4f}")
            logger.info(f"  Std:  {values.std():.4f}")
            logger.info(f"  Best: {values.max():.4f}")
    
    log_experiment_end(logger, "MULTI-MODEL COMPARISON")
    logger.info(f"\nAll comparison results saved to: {output_dir}")


if __name__ == "__main__":
    main()
