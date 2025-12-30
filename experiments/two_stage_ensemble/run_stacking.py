"""
Quick script to run stacked ensemble training.

This is a convenience wrapper around train.py that runs the stacked ensemble.

Usage:
    python experiments/two_stage_ensemble/run_stacking.py

Options can be configured in this file before running.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train import main

if __name__ == "__main__":
    print("="*60)
    print("STACKED ENSEMBLE TRAINING")
    print("="*60)
    print("\n⚠️  WARNING: This will take approximately 24 hours!")
    print("   Nested LOSO requires ~900 model trainings.\n")
    print("If you want faster results, use:")
    print("  python experiments/two_stage_ensemble/train.py")
    print("="*60)
    
    # Prompt for confirmation
    response = input("\nContinue with stacked ensemble? [y/N]: ")
    
    if response.lower() != 'y':
        print("Cancelled. Using simple ensemble instead...")
        main(use_stacking=False)
    else:
        print("\nStarting stacked ensemble training...")
        print("You can monitor progress in: results/training.log\n")
        
        # Run stacked ensemble
        main(
            use_stacking=True,
            meta_model_type="logistic_regression"  # Change to "xgboost" or "random_forest" if desired
        )

