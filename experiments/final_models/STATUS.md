# Final Models Experiment - Status Report

## ✅ Completed

### 1. Setup & Configuration
- ✅ Created folder structure: `experiments/final_models/`
- ✅ Written `config.py` with fixed hyperparameters (no tuning)
- ✅ Written `README.md` with usage instructions

### 2. Classical ML Training
- ✅ Written `train_classical_ml.py` (minimal, clean, modular)
- ✅ Trained 3 models with Ablation B (3 threshold strategies each):
  - **Logistic Regression**: Recall=0.591, FAR=0.207, G-mean=0.677
  - **Random Forest**: Recall=0.558, FAR=0.220, G-mean=0.660
  - **SVM**: Recall=0.034, FAR=0.024, G-mean=0.181 (poor performance)
- ✅ Saved all results (9 configurations total: 3 models × 3 thresholds)
  - Metrics: `lr_b1_metrics.json`, `lr_b2_metrics.json`, `lr_b3_metrics.json`, etc.
  - Fold metrics: `lr_b1_fold_metrics.csv`, etc.
  - Predictions: `lr_b1_predictions.csv`, etc.

**Training time**: ~1 minute total

## 🔄 In Progress / Pending

### 3. TCN Training
- ✅ Written `train_tcn.py` (minimal, clean, modular)
- ⏳ **To run**: 
  ```bash
  cd experiments/final_models
  python train_tcn.py
  ```
- **Estimated time**: 2-4 hours (100 epochs × 21 folds with early stopping)
- **Output**: `results/tcn/tcn_{b1,b2,b3}_*.{json,csv}`

### 4. Fusion Models Training
- ✅ Written `train_fusion.py` (Logical OR, Cascade, Stacked)
- ⏳ **To run** (after TCN completes):
  ```bash
  cd experiments/final_models
  python train_fusion.py
  ```
- **Estimated time**: 30-60 minutes (Stacked uses nested LOSO)
- **Output**: 9 more configurations (3 fusion × 3 thresholds)

### 5. Visualization & Analysis
- ⏳ **To write**: `generate_figures.py`
- ⏳ **To run** (after fusion completes):
  ```bash
  cd experiments/final_models
  python generate_figures.py
  ```
- **Required figures**:
  1. Model comparison table (all models × all thresholds)
  2. Recall vs FAR trade-off plot
  3. Subject-wise recall distribution (boxplot/violin)
  4. Normalized confusion matrices (5 key models)
  5. Precision-Recall curves
  6. Threshold sensitivity plots (5 models)

### 6. Documentation
- ⏳ **To write**: `results/comparison/RESULTS_SUMMARY.md`
- **Content**:
  - Ablation A findings (Model Family: LR vs RF vs SVM vs TCN)
  - Ablation B findings (Thresholding: B1 vs B2 vs B3)
  - Ablation C findings (Fusion: OR vs Cascade vs Stacked)
  - Recommendations for deployment

---

## Current Results (Classical ML Only)

### Model Comparison (B1 - Unconstrained G-Mean)

| Model | Recall | FAR | Specificity | Precision | G-mean | AUROC | PR-AUC |
|-------|--------|-----|-------------|-----------|--------|-------|--------|
| **LR** | 0.591 | 0.207 | 0.793 | 0.344 | 0.677 | 0.750 | 0.389 |
| **RF** | 0.558 | 0.220 | 0.780 | 0.323 | 0.660 | 0.743 | 0.378 |
| **SVM** | 0.034 | 0.024 | 0.976 | 0.188 | 0.181 | 0.674 | 0.165 |

**Key observations**:
- LR performs best overall (highest recall, G-mean, AUROC)
- SVM has extremely low recall (only 3.4% of stress detected) - unsuitable for this task
- All models show class imbalance challenge (12% positive class)
- AUROC ~0.75 indicates moderate discrimination ability

---

## Design Principles Followed

### ✅ Code Quality
- **Minimal**: 3 training scripts instead of 7+
- **Clean**: No fallback functions, fail-fast error handling
- **Modular**: Reuses `experiments/shared/` extensively
- **Readable**: Clear structure, no unnecessary abstractions

### ✅ Zero Data Leakage
- **Per-fold imputation**: `fit(train)` → `transform(test)` inside LOSO loop
- **Threshold selection**: Always on training data only
- **Subject-wise normalization**: Stats computed per-subject before windowing
- **Fixed hyperparameters**: No tuning (prevents optimization bias)

### ✅ Reproducibility
- Fixed random seed (42)
- Identical LOSO folds across all models
- Same preprocessing pipeline for all classical ML models
- Saved predictions for fusion models

---

## Next Steps (Sequential)

1. **Run TCN training** (~2-4 hours)
   ```bash
   cd experiments/final_models
   python train_tcn.py
   ```

2. **Run fusion training** (~30-60 min, after TCN)
   ```bash
   python train_fusion.py
   ```

3. **Write visualization script**
   - Create `generate_figures.py`
   - Load all results (6 models × 3 thresholds = 18 configs)
   - Generate 6 required visualizations
   - Save to `results/figures/`

4. **Write results summary**
   - Create `results/comparison/RESULTS_SUMMARY.md`
   - Analyze all 3 ablations (A, B, C)
   - Provide deployment recommendations

5. **Quality check**
   - Verify all metrics make sense
   - Check for any LOSO fold mismatches
   - Ensure threshold-independent metrics (AUROC, PR-AUC) consistent

---

## File Structure

```
experiments/final_models/
├── config.py                           # Fixed hyperparameters
├── train_classical_ml.py               # LR, RF, SVM (DONE)
├── train_tcn.py                        # TCN (WRITTEN, NOT RUN)
├── train_fusion.py                     # OR, Cascade, Stacked (WRITTEN, NOT RUN)
├── generate_figures.py                 # TODO: Write
├── README.md                           # Usage instructions
├── STATUS.md                           # This file
└── results/
    ├── lr/                             # ✅ 9 files
    ├── rf/                             # ✅ 9 files
    ├── svm/                            # ✅ 9 files
    ├── tcn/                            # ⏳ To be created
    ├── or/                             # ⏳ To be created
    ├── cascade/                        # ⏳ To be created
    ├── stacked/                        # ⏳ To be created
    ├── comparison/                     # ⏳ Final tables & summary
    └── figures/                        # ⏳ All visualizations
```

---

## Estimated Remaining Time

- TCN training: 2-4 hours
- Fusion training: 30-60 minutes
- Write generate_figures.py: 30 minutes
- Generate all figures: 5 minutes
- Write RESULTS_SUMMARY.md: 30 minutes

**Total remaining: ~4-6 hours** (mostly training time)

---

## Notes

- **SVM performance**: Very poor (3.4% recall). Consider excluding from final comparison or investigating hyperparameters.
- **LR is strongest classical ML**: Best across all metrics. Use LR for fusion (not RF or SVM).
- **Threshold strategies**: B1, B2, B3 are producing identical results for some models. This suggests the constraints (recall≥70%, FAR≤30%) are not binding. May need to adjust constraints or investigate threshold selection.
- **Class imbalance**: 12% positive class is challenging. PR-AUC (~0.38) is more informative than AUROC (~0.75) for this imbalanced dataset.

