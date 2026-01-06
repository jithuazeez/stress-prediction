#!/bin/bash
# Script to run remaining experiments in sequence

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Final Models - Remaining Experiments"
echo "=========================================="
echo ""

# Check if classical ML is done
if [ ! -f "results/lr/lr_b1_metrics.json" ]; then
    echo "❌ Classical ML not complete. Run: python train_classical_ml.py"
    exit 1
fi
echo "✅ Classical ML completed"

# Step 1: TCN Training
if [ ! -f "results/tcn/tcn_b1_metrics.json" ]; then
    echo ""
    echo "Step 1/3: Training TCN (2-4 hours)..."
    echo "----------------------------------------"
    python -u train_tcn.py 2>&1 | tee logs/tcn_training.log
    echo "✅ TCN training completed"
else
    echo "✅ TCN already trained"
fi

# Step 2: Fusion Training
if [ ! -f "results/stacked/stacked_b1_metrics.json" ]; then
    echo ""
    echo "Step 2/3: Training Fusion Models (30-60 min)..."
    echo "----------------------------------------"
    python -u train_fusion.py 2>&1 | tee logs/fusion_training.log
    echo "✅ Fusion training completed"
else
    echo "✅ Fusion models already trained"
fi

# Step 3: Generate Figures
echo ""
echo "Step 3/3: Generating Figures..."
echo "----------------------------------------"
if [ -f "generate_figures.py" ]; then
    python generate_figures.py
    echo "✅ Figures generated"
else
    echo "⚠️  generate_figures.py not yet created"
    echo "   Create this script to generate visualizations"
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: results/"
echo "Figures saved in: results/figures/"
echo ""
echo "Next: Review STATUS.md for analysis"

