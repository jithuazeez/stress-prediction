# TCN Quick Start Guide

## What is TCN?

Temporal Convolutional Network (TCN) is a deep learning architecture that combines:
- **Feature engineering** from classical ML (interpretable features)
- **Deep learning** power (automatic pattern learning)

It uses dilated causal convolutions with residual connections to learn from physiological features extracted on-the-fly.

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
# Navigate to project root
cd /Users/jithuazeez/Documents/Msc/Dissertation

# Install requirements (if not already done)
pip install -r experiments/requirements.txt
```

### 2. Run Training

```bash
# Navigate to TCN experiment
cd experiments/08_tcn

# Run training
python run.py
```

**OR** using the train script directly:

```bash
python train.py
```

### 3. Check Results

After training completes (~30-60 minutes), check:

```bash
# View metrics
cat results/tcn_metrics.json

# View training log
less results/training.log

# View per-fold metrics
open results/tcn_fold_metrics.csv

# View plots
open results/figures/tcn_roc.png
open results/figures/tcn_pr.png
```

## Expected Output

### Terminal Output

```
==============================================================
TCN MODEL TRAINING
==============================================================
Device: cuda / mps / cpu

--------------------------------------------------
PHASE 1: Loading and preprocessing data
--------------------------------------------------
Loading 22 subjects...
Processing subjects: 100%|████████████| 22/22
Loaded 22 subjects with 2156 total windows

--------------------------------------------------
PHASE 2: Model Training (LOSO CV)
--------------------------------------------------
LOSO CV: 22 subjects | 50 epochs | batch=32 | lr=0.001
Model: 145,730 params | Trainable: 145,730 | RF: 31

LOSO folds: 100%|████████████| 22/22
Fold  1/22 | Loss: 0.234 | Gmean: 0.712 | F1: 0.689
Fold  2/22 | Loss: 0.198 | Gmean: 0.745 | F1: 0.721
...
```

### Final Metrics

```
==============================================================
OVERALL PERFORMANCE ACROSS ALL FOLDS
==============================================================
Total samples:     2156
Positive samples:  543 (25.2%)
Negative samples:  1613 (74.8%)

--- Threshold-Independent Metrics ---
  AUROC:           0.7856 ± 0.0234
  PR-AUC:          0.6123 ± 0.0312

--- Test Set Performance ---
  Gmean:           0.7234 ± 0.0456
  F1-Score:        0.6890 ± 0.0523
  Sensitivity:     0.7123 ± 0.0612
  Specificity:     0.7345 ± 0.0498
```

## File Structure

After training, you'll have:

```
08_tcn/
├── dataset.py                  # PyTorch dataset with feature extraction
├── model.py                    # TCN architecture
├── train.py                    # Training script
├── run.py                      # Quick start script
├── README.md                   # Full documentation
├── QUICKSTART.md              # This file
└── results/
    ├── training.log            # Training logs
    ├── tcn_metrics.json        # Overall metrics
    ├── tcn_predictions.csv     # All predictions
    ├── tcn_fold_metrics.csv    # Per-fold metrics
    ├── checkpoints/
    │   └── best_model.pt       # Best model
    ├── figures/
    │   ├── tcn_roc.png        # ROC curve
    │   ├── tcn_pr.png         # PR curve
    │   └── tcn_confusion_matrix.png
    └── config/
        └── training_config.json
```

## Customization

### Change Hyperparameters

Edit `train.py` main() function:

```python
results = loso_cross_validation(
    windows_by_subject,
    config,
    device,
    logger,
    n_epochs=100,              # More epochs
    batch_size=64,             # Larger batch
    learning_rate=5e-4,        # Different learning rate
    tcn_channels=[128, 128],   # Larger model
    kernel_size=5,             # Larger kernels
    dropout=0.4,               # More dropout
    fc_hidden_dim=256          # Larger FC layer
)
```

### Change Window Size

Edit `shared/config.py`:

```python
window_size_sec: int = 60      # 60 second windows
overlap_ratio: float = 0.75    # 75% overlap
target_label: str = "label_3min"  # 3-minute prediction
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'shared'"

Make sure you're running from the correct directory:

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments/08_tcn
python train.py
```

### "CUDA out of memory" (if using GPU)

Reduce batch size:

```python
batch_size=16  # or even 8
```

### Poor Performance

1. Check feature extraction is working:
   ```bash
   python dataset.py  # Test dataset
   ```

2. Check model architecture:
   ```bash
   python model.py    # Test model
   ```

3. Try different hyperparameters:
   - Increase model capacity: `tcn_channels=[128, 128, 128]`
   - Increase dropout: `dropout=0.5`
   - Change learning rate: `learning_rate=5e-4`

## Next Steps

1. **Compare with other models**: See `experiments/compare_all.py`

2. **Analyze results**: 
   - ROC/PR curves in `results/figures/`
   - Per-fold metrics in `results/tcn_fold_metrics.csv`
   - Predictions in `results/tcn_predictions.csv`

3. **Read full docs**: See `README.md` for complete documentation

## Key References

- [TCN Article by Unit8](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)
- [TCN Paper (Bai et al., 2018)](https://arxiv.org/pdf/1803.01271.pdf)

## Need Help?

Check the full README.md for:
- Architecture details
- Feature descriptions
- Comparison with other models
- Implementation details
- Advanced usage examples

