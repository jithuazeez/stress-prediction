#!/usr/bin/env python3
"""
Quick start script for Two-Stage Ensemble training.

Usage:
    python run.py [--min-recall-lr 0.75] [--max-far-tcn 0.25] [--epochs 100]
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.two_stage_ensemble.train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage Ensemble: LR (Recall-First) → TCN (FAR-First)"
    )
    
    parser.add_argument(
        "--min-recall-lr",
        type=float,
        default=0.75,
        help="Minimum recall for LR threshold (default: 0.75)"
    )
    
    parser.add_argument(
        "--max-far-tcn",
        type=float,
        default=0.25,
        help="Maximum FAR for TCN threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for TCN training (default: 100)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("TWO-STAGE ENSEMBLE TRAINING")
    print("="*60)
    print(f"LR Threshold: Recall-first (min {args.min_recall_lr*100:.0f}%)")
    print(f"TCN Threshold: FAR-first (max {args.max_far_tcn*100:.0f}%)")
    print(f"TCN Epochs: {args.epochs}")
    print("="*60)
    print()
    
    # Run training
    train_main()

