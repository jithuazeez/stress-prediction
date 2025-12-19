#!/usr/bin/env python
"""
Run all Multi-Rate Fusion experiment configurations.

Runs training for all combinations of:
- Window sizes: 60s, 120s
- Horizons: 3min, 5min

Usage:
    python run_all.py
    python run_all.py --windows 120 --horizons 3 5
"""

import subprocess
import sys
from pathlib import Path
import argparse
from itertools import product


# Experiment configurations
WINDOW_SIZES = [120, 60]
HORIZONS = [3, 5]


def run_training(window_size: int, horizon: int, epochs: int = 100):
    """Run training for a specific configuration."""
    cmd = [
        sys.executable, "train.py",
        "--window_size", str(window_size),
        "--horizon", str(horizon),
        "--epochs", str(epochs)
    ]
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {window_size}s | {horizon}min | {epochs} epochs")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all Multi-Rate Fusion experiments")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOW_SIZES, help="Window sizes")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS, help="Horizons")
    
    args = parser.parse_args()
    
    results = []
    
    print("\n" + "="*70)
    print("MULTI-RATE LATE FUSION EXPERIMENTS")
    print("="*70)
    
    for window_size, horizon in product(args.windows, args.horizons):
        success = run_training(window_size, horizon, args.epochs)
        results.append({
            "window_size": window_size,
            "horizon": horizon,
            "success": success
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['window_size']}s | {r['horizon']}min")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\nTotal: {success_count}/{len(results)} successful")


if __name__ == "__main__":
    main()


