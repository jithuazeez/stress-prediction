#!/usr/bin/env python3
"""
Quick start script for TCN experiment.

Trains TCN model with LOSO cross-validation for stress prediction.
"""

import sys
from pathlib import Path

# Add experiment to path
sys.path.insert(0, str(Path(__file__).parent))

from train import main

if __name__ == "__main__":
    print("=" * 60)
    print("TCN STRESS PREDICTION - QUICK START")
    print("=" * 60)
    print("\nThis will train a Temporal Convolutional Network (TCN) model")
    print("for stress prediction using LOSO cross-validation.")
    print("\nExpected runtime: ~30-60 minutes (depending on hardware)")
    print("=" * 60)
    
    # Run training
    main()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print("\nResults saved to: experiments/08_tcn/results/")
    print("  - training.log: Detailed training logs")
    print("  - tcn_metrics.json: Overall metrics")
    print("  - tcn_predictions.csv: Predictions for all samples")
    print("  - tcn_fold_metrics.csv: Per-fold breakdown")
    print("  - checkpoints/best_model.pt: Best model checkpoint")
    print("  - figures/: Visualizations (ROC, PR, confusion matrix)")
    print("=" * 60)

