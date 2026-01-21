# Final Models Experiment

Systematic ablation study for final model comparison with fixed pipeline.

## Structure

```
final_models/
├── config.py                 # Fixed hyperparameters
├── train_classical_ml.py     # LR, RF, SVM with Ablation B
├── train_tcn.py              # TCN with Ablation B
├── train_fusion.py           # OR, Cascade, Stacked with Ablation B
├── generate_figures.py       # All visualizations
└── results/                  # All outputs
```

## Ablation Studies

- **Ablation A**: Model Family (LR/RF/SVM vs TCN)
- **Ablation B**: Thresholding Strategy (B1/B2/B3)
- **Ablation C**: Fusion Strategy (OR vs Cascade vs Stacked)

## Fixed Pipeline

- **Sampling Rate**: 4 Hz
- **Window**: 120s, 50% overlap
- **Label**: 5-minute prediction horizon
- **Normalization**: Subject-wise z-score
- **Missing Data**: Quality filtering + per-fold median imputation
- **Cross-validation**: LOSO (Leave-One-Subject-Out)

## Usage

```bash
# 1. Train classical ML models
python train_classical_ml.py

# 2. Train TCN
python train_tcn.py

# 3. Train fusion models
python train_fusion.py

# 4. Generate all figures
python generate_figures.py
```

## Results

See `results/comparison/RESULTS_SUMMARY.md` after running all experiments.


