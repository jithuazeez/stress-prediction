#!/usr/bin/env python
"""
Run all SSL experiment configurations.

Runs pre-training and fine-tuning for all combinations of:
- Window sizes: 60s, 120s
- Horizons: 3min, 5min
- SSL modes: base, invariant, specific

Usage:
    python run_all.py --pretrain_only     # Only pre-training
    python run_all.py --finetune_only     # Only fine-tuning (requires pre-trained)
    python run_all.py                      # Both pre-training and fine-tuning
"""

import subprocess
import sys
from pathlib import Path
import argparse
from itertools import product


# Experiment configurations
WINDOW_SIZES = [120, 60]
HORIZONS = [3, 5]
SSL_MODES = ["base", "invariant", "specific"]


def run_pretrain(mode: str, window_size: int, epochs: int = 100):
    """Run pre-training for a specific configuration."""
    cmd = [
        sys.executable, "pretrain.py",
        "--mode", mode,
        "--window_size", str(window_size),
        "--epochs", str(epochs)
    ]
    
    print(f"\n{'='*60}")
    print(f"PRE-TRAINING: {mode} | {window_size}s | {epochs} epochs")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_finetune(mode: str, window_size: int, horizon: int, epochs: int = 50):
    """Run fine-tuning for a specific configuration."""
    cmd = [
        sys.executable, "train.py",
        "--mode", mode,
        "--window_size", str(window_size),
        "--horizon", str(horizon),
        "--epochs", str(epochs)
    ]
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING: {mode} | {window_size}s | {horizon}min | {epochs} epochs")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all SSL experiments")
    parser.add_argument("--pretrain_only", action="store_true", help="Only run pre-training")
    parser.add_argument("--finetune_only", action="store_true", help="Only run fine-tuning")
    parser.add_argument("--pretrain_epochs", type=int, default=100, help="Pre-training epochs")
    parser.add_argument("--finetune_epochs", type=int, default=50, help="Fine-tuning epochs")
    parser.add_argument("--modes", nargs="+", default=SSL_MODES, help="SSL modes to run")
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOW_SIZES, help="Window sizes")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS, help="Horizons")
    
    args = parser.parse_args()
    
    run_pretrain_flag = not args.finetune_only
    run_finetune_flag = not args.pretrain_only
    
    results = {"pretrain": [], "finetune": []}
    
    # Pre-training (one per mode + window_size)
    if run_pretrain_flag:
        print("\n" + "="*70)
        print("PHASE 1: PRE-TRAINING")
        print("="*70)
        
        for mode, window_size in product(args.modes, args.windows):
            success = run_pretrain(mode, window_size, args.pretrain_epochs)
            results["pretrain"].append({
                "mode": mode,
                "window_size": window_size,
                "success": success
            })
    
    # Fine-tuning (one per mode + window_size + horizon)
    if run_finetune_flag:
        print("\n" + "="*70)
        print("PHASE 2: FINE-TUNING")
        print("="*70)
        
        for mode, window_size, horizon in product(args.modes, args.windows, args.horizons):
            success = run_finetune(mode, window_size, horizon, args.finetune_epochs)
            results["finetune"].append({
                "mode": mode,
                "window_size": window_size,
                "horizon": horizon,
                "success": success
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results["pretrain"]:
        print("\nPre-training:")
        for r in results["pretrain"]:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['mode']} | {r['window_size']}s")
    
    if results["finetune"]:
        print("\nFine-tuning:")
        for r in results["finetune"]:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['mode']} | {r['window_size']}s | {r['horizon']}min")
    
    # Count successes
    pretrain_success = sum(1 for r in results["pretrain"] if r["success"])
    finetune_success = sum(1 for r in results["finetune"] if r["success"])
    
    print(f"\nPre-train: {pretrain_success}/{len(results['pretrain'])} successful")
    print(f"Fine-tune: {finetune_success}/{len(results['finetune'])} successful")


if __name__ == "__main__":
    main()
