# Quick Start Guide: TabPFN Experiment

## Prerequisites

1. **Python 3.9+** (TabPFN requires Python 3.9 or higher)
2. **HuggingFace Account** (free, required for model download)
3. **Completed Classical ML Experiment** (optional, for comparison)

## Installation Steps

### Step 1: Install TabPFN

```bash
# Navigate to experiments directory
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments

# Install TabPFN
pip install tabpfn

# Or update requirements and install all
pip install -r requirements.txt
```

### Step 2: HuggingFace Authentication

TabPFN requires authentication to download the pre-trained model (~500MB).

```bash
# Login to HuggingFace (one-time setup)
huggingface-cli login
```

When prompted:
1. Paste your HuggingFace token (get from: https://huggingface.co/settings/tokens)
2. Press Enter

### Step 3: Accept Model License

Visit: https://huggingface.co/Prior-Labs/tabpfn_2_5

Click "Agree and access repository"

## Running the Experiment

### Quick Run

```bash
# Navigate to TabPFN experiment
cd 07_tabpfn

# Run training
python train.py
```

### Expected Output

```
============================================================
EXPERIMENT START: TabPFN FOUNDATION MODEL TRAINING
============================================================

Checking TabPFN requirements...
✓ TabPFN model accessible
✓ Using device: cuda

Feature extractor initialized
Expected features: 65
HeartPy: Available - will extract HR/HRV from raw PPG ✓

Found 21 subject folders

--------------------------------------------------
PHASE 1: Loading and preprocessing data
--------------------------------------------------

Loading subjects: 100%|███████████| 21/21 [00:30<00:00]
OK: 21, Fail: 0

--------------------------------------------------
PHASE 2: TabPFN Training (LOSO CV)
--------------------------------------------------

  TabPFN: 100%|███████████| 21/21 [05:30<00:00]
  Gmean: 0.785  Recall: 0.823  Test: 67

Training completed in 330.5s (15.7s per fold)

============================================================
TabPFN TRAINING COMPLETE
============================================================
  AUROC:       0.8234
  PR-AUC:      0.7156
  F1:          0.7245
  Gmean:       0.7852

All results saved to: results/
```

### Runtime Expectations

- **With GPU**: ~10-20 seconds per fold (~5-8 minutes total)
- **With CPU**: ~30-60 seconds per fold (~15-20 minutes total)

## Output Files

After completion, check `07_tabpfn/results/`:

```
results/
├── training.log                    # Detailed execution log
├── tabpfn_metrics.json            # Overall performance metrics
├── tabpfn_fold_metrics.csv        # Per-fold (per-subject) metrics
├── tabpfn_predictions.csv         # All predictions with probabilities
├── features_dataset.csv           # Extracted features (same as classical ML)
├── missing_analysis.json          # Data quality report
├── class_distribution.json        # Class balance info
└── figures/
    ├── tabpfn_roc.png            # ROC curve
    ├── tabpfn_pr.png             # Precision-Recall curve
    └── tabpfn_confusion_matrix.png
```

## Comparing with Classical ML

To see how TabPFN compares to XGBoost, Random Forest, etc.:

```bash
# Go back to experiments root
cd ..

# Run comparison
python compare_all.py
```

This will generate:
- `comparison_results/model_comparison.csv` - All models ranked by AUROC
- `comparison_results/comparison_bar.png` - Bar chart comparison
- `comparison_results/comparison_radar.png` - Radar chart

## Troubleshooting

### Issue 1: "TabPFN not available"

**Solution:**
```bash
pip install tabpfn --upgrade
```

### Issue 2: "Access denied to model"

**Solution:**
```bash
# Ensure you're logged in
huggingface-cli login

# Accept license at:
# https://huggingface.co/Prior-Labs/tabpfn_2_5
```

### Issue 3: "CUDA out of memory"

**Solution:** TabPFN will automatically use CPU. If you want to force CPU:

Edit `train.py`, line ~1050:
```python
device = "cpu"  # Force CPU mode
```

### Issue 4: Import errors for feature extraction

**Solution:**
```bash
# Make sure you're in the experiments directory
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments

# Run from experiments root
python 07_tabpfn/train.py
```

### Issue 5: "HeartPy not available"

This is OK - the script will skip PPG-based HR/HRV features but will still work with other features (accelerometer, temperature, etc.).

To enable HR/HRV features:
```bash
pip install heartpy
```

## Key Differences from Classical ML

| Feature | Classical ML | TabPFN |
|---------|-------------|--------|
| Hyperparameter tuning | ✅ Nested CV (~30-60s per fold) | ❌ Not needed (~10-20s per fold) |
| Feature scaling | ✅ Required (StandardScaler) | ❌ Automatic |
| Class balancing | ✅ Manual weights | ❌ Automatic |
| Model size | Small (<1MB) | Large (~500MB) |
| Interpretability | High | Low |

## Next Steps

1. **Analyze Results**: Check `results/training.log` for detailed metrics
2. **Compare Models**: Run `compare_all.py` to see TabPFN vs. other approaches
3. **Tune Threshold**: If needed, adjust `threshold_method` in `train.py` (line ~1055)
4. **Feature Analysis**: Compare feature importance (TabPFN doesn't provide this, but classical ML does)

## Expected Performance

Based on the VitaStress dataset characteristics:
- **AUROC**: 0.75-0.85 (similar to XGBoost)
- **PR-AUC**: 0.60-0.75
- **F1**: 0.65-0.75
- **Gmean**: 0.70-0.80

TabPFN should perform competitively with or slightly better than classical ML, especially given its pre-training on diverse tabular data.

## Support

For TabPFN-specific issues:
- GitHub: https://github.com/PriorLabs/TabPFN
- HuggingFace: https://huggingface.co/Prior-Labs/tabpfn_2_5

For experiment-specific issues, check:
- `results/training.log` - Detailed execution trace
- Classical ML results for comparison baseline

