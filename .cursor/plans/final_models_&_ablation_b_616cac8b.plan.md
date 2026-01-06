---
name: Final Models - Ablations A, B, C
overview: "Minimal, clean implementation of final models with 3 ablations: (A) Model Family - LR/RF/SVM vs TCN, (B) Thresholding Strategy - B1/B2/B3, (C) Fusion Strategy - OR vs Cascade vs Stacked. Fixed hyperparameters, per-fold imputation (no leakage), subject-wise normalization. 3 training scripts only. Total: 6 models × 3 thresholds = 18 configurations."
todos:
  - id: setup_structure
    content: Create folder structure and config.py with fixed hyperparameters
    status: completed
  - id: train_classical_ml
    content: Write and run train_classical_ml.py (LR, RF, SVM with Ablation B)
    status: completed
    dependencies:
      - setup_structure
  - id: train_tcn
    content: Write and run train_tcn.py (TCN with Ablation B)
    status: pending
    dependencies:
      - setup_structure
  - id: train_fusion
    content: Write and run train_fusion.py (OR, Cascade, Stacked with Ablation B)
    status: pending
    dependencies:
      - train_classical_ml
      - train_tcn
  - id: generate_figures
    content: Write and run generate_figures.py for all visualizations
    status: completed
    dependencies:
      - train_fusion
  - id: document_results
    content: Write RESULTS_SUMMARY.md with findings for all ablations
    status: completed
    dependencies:
      - generate_figures
---

# Final Models: Ablations A, B, and C

## Overview

Create standardized experiment for final model comparison with three systematic ablations:

- **Ablation A**: Model Family (LR vs TCN)
- **Ablation B**: Thresholding Strategy (B1 vs B2 vs B3)
- **Ablation C**: Fusion Strategy (Logical OR vs Cascade vs Stacked)

**Total experiments**: 3 classical ML + 1 TCN + 3 fusion × 3 thresholds = **21 configurationsKey design principles**:

- **Minimal code**: 3 training scripts (not 7+)
- **No hyperparameter tuning**: Fixed params based on your specifications
- **Per-fold imputation**: Fit on train, transform test (zero leakage)
- **No fallbacks**: Fail fast if errors occur
- **Clean & readable**: Follow `experiments/classical_ml/train.py` best practices

---

## Fixed Pipeline Components (Non-Negotiable)

All models will use identical preprocessing:

### Data Pipeline

- **Sampling Rate**: 4 Hz alignment (480 timesteps per 120s window)
- **Window Size**: 120 seconds
- **Overlap**: 50% (for sample augmentation within LOSO)
- **Label**: `label_5min` (5-minute prediction horizon)
- **Normalization**: Subject-wise z-score (computed per-subject on full subject data, passed via `subject_stats` in windows)
- **Missing Data Strategy** (NO LEAKAGE):
- Stage 1: Quality filtering (drop windows where `hr_bpm` is NaN - HRV extraction failed)
- Stage 2: Per-fold median imputation
    - For LR/RF/SVM: Fit imputer on training fold → transform test fold
    - For TCN: Forward-fill in raw time series (maintains temporal continuity)
- **Critical**: Imputation NEVER sees test data (done inside LOSO loop per fold)

### Evaluation

- **Cross-validation**: LOSO (Leave-One-Subject-Out)
- **Metrics**: Recall, FAR, Specificity, Precision, G-mean, AUROC, PR-AUC, Balanced Accuracy
- **Primary Metric**: Recall-FAR trade-off

---

## Ablation Studies

### Ablation A: Model Family

Purpose: Assess modeling capacity and temporal processing.

- **A1**: Classical ML (baseline - engineered features)
- Logistic Regression: `C=0.1, penalty='l2', solver='saga'`
- Random Forest: `n_estimators=100, max_depth=5, max_features='log2', min_samples_leaf=4, min_samples_split=10`
- SVM: `C=10.0, kernel='rbf', gamma=0.1, degree=2`
- **A2**: TCN (temporal modeling - raw time series)

**Evaluation for Ablation A:**

- **Primary metrics** (threshold-independent): AUROC, PR-AUC
- **Secondary metrics** (threshold-dependent): Use B1 (Unconstrained G-Mean) for both models
- Rationale: Fair comparison requires same threshold strategy; threshold-independent metrics are primary

### Ablation B: Thresholding Strategy

Purpose: Assess operational trade-offs (applied to ALL 5 models).| Strategy | Method | Constraints | Purpose ||----------|--------|-------------|---------|| **B1** | Unconstrained G-Mean | None | Balanced sensitivity/specificity || **B2** | Recall-constrained | min_recall = 0.70 | Ensure 70%+ stress detection || **B3** | FAR-constrained | max_FAR = 0.30 | Limit false alarms to 30% |**Critical**: Thresholds selected on **training folds only**, never on test data.

### Ablation C: Fusion Strategy

Purpose: Assess decision-level integration methods.

- **C1**: Logical OR (LR ∨ TCN) - Predict stress if either model predicts stress (maximizes recall)
- **C2**: Two-stage Cascade (LR → TCN) - LR screens, TCN confirms (balances recall and specificity)
- **C3**: Stacked Generalization - Meta-learner combines LR + TCN probabilities (learns optimal fusion)

**Evaluation for Ablation C:**

- **Primary metrics** (threshold-independent): AUROC, PR-AUC
- **Secondary metrics** (threshold-dependent): Use B1 (Unconstrained G-Mean) for all fusion methods
- Rationale: Fair comparison of fusion strategies; B1 provides operational baseline

---

## Implementation Plan

### Phase 1: Setup (Structure & Configuration)

**1.1 Create experiment structure**

```javascript
experiments/final_models/
├── config.py                      # Fixed hyperparameters & config
├── train_classical_ml.py          # LR, RF, SVM with Ablation B (A1)
├── train_tcn.py                   # TCN with Ablation B (A2)
├── train_fusion.py                # OR, Cascade, Stacked with Ablation B (C1, C2, C3)
├── generate_figures.py            # All visualizations
├── results/
│   ├── lr/                        # LR: B1, B2, B3 metrics + probabilities
│   ├── rf/                        # RF: B1, B2, B3 metrics + probabilities
│   ├── svm/                       # SVM: B1, B2, B3 metrics + probabilities
│   ├── tcn/                       # TCN: B1, B2, B3 metrics + probabilities
│   ├── or/                        # OR: B1, B2, B3 metrics
│   ├── cascade/                   # Cascade: B1, B2, B3 metrics
│   ├── stacked/                   # Stacked: B1, B2, B3 metrics
│   └── comparison/                # Aggregated results & figures
└── README.md
```

**Design principles**:

- **3 training scripts** (not 7) - minimal and focused
- **No shared_pipeline.py** - directly use existing `shared/` modules
- **No evaluate_ablations.py** - evaluation happens inline during training
- Reuse functions from `experiments/shared/` (no duplication)

**1.2 Configuration file** (`config.py`)

```python
# Fixed hyperparameters (no tuning)
LR_PARAMS = {"C": 0.1, "penalty": "l2", "solver": "saga", "max_iter": 1000, 
             "class_weight": "balanced", "random_state": 42}

RF_PARAMS = {"n_estimators": 100, "max_depth": 5, "max_features": "log2",
             "min_samples_leaf": 4, "min_samples_split": 10, "bootstrap": True,
             "class_weight": "balanced", "random_state": 42, "n_jobs": -1}

SVM_PARAMS = {"C": 10.0, "kernel": "rbf", "gamma": 0.1, "degree": 2,
              "probability": True, "class_weight": "balanced", "random_state": 42}

TCN_PARAMS = {"batch_size": 16, "lr": 1e-3, "epochs": 100, "dropout": 0.3,
              "dilations": [1,2,4,8,16,32,64,128], "channels": [16]*8}

# Pipeline config
TARGET_HZ = 4.0
OVERLAP_RATIO = 0.5
WINDOW_SIZE_SEC = 120
LABEL_COL = "label_5min"
```

**No shared_pipeline.py** - directly import from `experiments/shared/` and `experiments/classical_ml/`---

### Phase 2: Model Training

**2.1 Classical ML Models** (`train_classical_ml.py`)Based on [`experiments/classical_ml/train.py`](experiments/classical_ml/train.py):

- Feature extraction: Reuse `BasicFeatureExtractor` + HRV extraction
- **Fixed hyperparameters** (no tuning):
- LR: `C=0.1, penalty='l2', solver='saga', max_iter=1000, class_weight='balanced'`
- RF: `n_estimators=100, max_depth=5, max_features='log2', min_samples_leaf=4, min_samples_split=10, bootstrap=True, class_weight='balanced'`
- SVM: `C=10.0, kernel='rbf', gamma=0.1, degree=2, probability=True, class_weight='balanced'`
- **LOSO loop** (per fold):

1. Split: train subjects vs test subject
2. **Impute per-fold**: `imputer.fit(X_train)` → `transform(X_test)`
3. Train model with fixed hyperparameters
4. Get train probabilities → select 3 thresholds (B1, B2, B3) on TRAIN data
5. Get test probabilities → apply 3 thresholds → evaluate
6. Save probabilities for fusion models

- **No hyperparameter tuning** - use fixed params
- **No fallbacks** - fail if exception occurs
- **Minimal logging** - only essential metrics

**2.2 TCN** (`train_tcn.py`)Based on [`experiments/tcn/train.py`](experiments/tcn/train.py):

- Architecture: 8 TCN blocks, dilations `[1,2,4,8,16,32,64,128]` for 480 timesteps
- Loss: Weighted Cross-Entropy (sklearn-style balanced weights)
- Missing data: Forward-fill in `VitaStressTCNDataset` (not zero-fill)
- **Fixed hyperparameters**: `batch_size=16, lr=1e-3, epochs=100, dropout=0.3`
- **LOSO loop** (per fold):

1. Split: train subjects vs test subject
2. Train TCN with early stopping (patience=15)
3. Get train probabilities → select 3 thresholds (B1, B2, B3)
4. Get test probabilities → apply 3 thresholds → evaluate
5. Save probabilities for fusion models

- **No architecture search** - use fixed architecture
- **No fallbacks** - fail fast

**2.3 Fusion Models** (`train_fusion.py`)All fusion models use **best performing classical ML** (choose after running 2.1) + TCN.**Logical OR (C1)**:

- Load saved probabilities from LR/RF/SVM and TCN
- For each fold: `pred = (base_proba >= base_thr) OR (tcn_proba >= tcn_thr)`
- Apply 3 threshold strategies

**Cascade (C2)**:

- For each fold: `if base_proba < base_thr → 0, else → (tcn_proba >= tcn_thr)`
- Apply 3 threshold strategies

**Stacked (C3)**:

- Nested LOSO: For each outer fold, train meta-model on out-of-fold base predictions
- Meta-model: Logistic Regression (2 features: base_proba, tcn_proba)
- Apply 3 threshold strategies to meta-model output

**Key principles**:

- No training - just load and combine probabilities (except Stacked)
- Minimal code - single file for all 3 fusion methods
- No fallbacks

---

### Phase 3: Ablation B Evaluation

**3.1 Threshold selection logic** (shared across all models)Use existing `find_optimal_threshold()` from `shared/evaluation.py`:

```python
# Inside LOSO loop, after training:
train_proba = model.predict_proba(X_train)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

# Select 3 thresholds on TRAIN data
thr_b1, _ = find_optimal_threshold(train_proba, y_train, method="gmean")  # Unconstrained
thr_b2, _ = find_optimal_threshold(train_proba, y_train, method="recall", min_recall=0.70)
thr_b3, _ = find_optimal_threshold(train_proba, y_train, method="fpr", max_fpr=0.30)

# Apply to TEST data
pred_b1 = (test_proba >= thr_b1).astype(int)
pred_b2 = (test_proba >= thr_b2).astype(int)
pred_b3 = (test_proba >= thr_b3).astype(int)

# Evaluate each
metrics_b1 = evaluate_predictions(y_test, pred_b1, test_proba, "model")
metrics_b2 = evaluate_predictions(y_test, pred_b2, test_proba, "model")
metrics_b3 = evaluate_predictions(y_test, pred_b3, test_proba, "model")
```

**No new functions** - reuse existing `shared/evaluation.py` functions.**3.2 Aggregate results**

- For each model × threshold combination: compute mean ± std across folds
- Save separate metrics files: `lr_b1_metrics.json`, `lr_b2_metrics.json`, etc.
- Save fold-level details for per-subject analysis

---

### Phase 4: Visualization & Reporting

**4.1 Main Comparison Table** (`generate_figures.py`)**Table 1: Ablation A & C (Model/Fusion Comparison)** - Threshold-independent + B1 only| Model | AUROC | PR-AUC | Recall@B1 | FAR@B1 | Spec@B1 | Prec@B1 | G-mean@B1 | Bal.Acc@B1 ||-------|-------|--------|-----------|--------|---------|---------|-----------|------------|| LR | ... | ... | ... | ... | ... | ... | ... | ... || TCN | ... | ... | ... | ... | ... | ... | ... | ... || OR (C1) | ... | ... | ... | ... | ... | ... | ... | ... || Cascade (C2) | ... | ... | ... | ... | ... | ... | ... | ... || Stacked (C3) | ... | ... | ... | ... | ... | ... | ... | ... |**Table 2: Ablation B (Thresholding Comparison)** - All threshold strategies per model| Model | Threshold | Recall | FAR | Spec | Prec | G-mean | PR-AUC | AUROC | Bal.Acc ||-------|-----------|--------|-----|------|------|--------|--------|-------|---------|| **LR (B1)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **LR (B2)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **LR (B3)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **TCN (B1)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **TCN (B2)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **TCN (B3)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **OR (B1)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **OR (B2)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **OR (B3)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **Cascade (B1)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **Cascade (B2)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **Cascade (B3)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **Stacked (B1)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **Stacked (B2)** | ... | ... | ... | ... | ... | ... | ... | ... | ... || **Stacked (B3)** | ... | ... | ... | ... | ... | ... | ... | ... | ... |**4.2 Recall vs FAR Trade-off Plot**

- X-axis: FAR (0-1)
- Y-axis: Recall (0-1)
- **5 curves**: LR, TCN, OR, Cascade, Stacked
- Mark operating points for B1, B2, B3 on each curve
- Show Pareto frontier
- **Key insight**: Shows which fusion strategy achieves best recall-FAR trade-off

**4.3 Subject-wise Recall Distribution**

- Boxplot: **5 boxes** (LR, TCN, OR, Cascade, Stacked) for best threshold per model
- Shows per-subject recall variability
- Demonstrates generalization difficulty

**4.4 Confusion Matrices (Normalized)**

- **5 matrices**: LR (best), TCN (best), OR (best), Cascade (best), Stacked (best)
- Normalize by true class
- Show actual counts as annotations

**4.5 Precision-Recall Curves**

- **5 curves**: LR, TCN, OR, Cascade, Stacked
- Threshold-independent evaluation
- Highlight low positive rate (7.4%)
- **Key insight**: Shows which model handles class imbalance best

**4.6 Threshold Sensitivity Plot**

- For **each of 5 models**: X-axis = threshold (0-1), Y-axis = metric value
- Show: Recall (blue), FAR (red), G-mean (green)
- Mark B1, B2, B3 operating points
- Helps understand threshold impact

**4.7 Ablation C Analysis Plot** (NEW)

- Bar chart comparing 3 fusion strategies (C1, C2, C3) with best threshold each
- Metrics: Recall, Specificity, G-mean, F1
- **Key insight**: Which fusion strategy performs best overall?

---

### Phase 5: Documentation & Validation

**5.1 Results summary** (`results/RESULTS_SUMMARY.md`)

- **Ablation A findings**: Does temporal modeling (TCN) improve over features (LR)?
- Compare AUROC and PR-AUC (threshold-independent)
- Use B1 (unconstrained G-Mean) for operational metrics
- **Ablation B findings**: Which thresholding strategy works best for each model?
- Compare B1 vs B2 vs B3 WITHIN each model
- Show recall-FAR trade-offs per threshold strategy
- **Ablation C findings**: Which fusion strategy achieves best performance? (OR vs Cascade vs Stacked)
- Compare AUROC and PR-AUC (threshold-independent)
- Use B1 for operational context
- Recommendations for deployment based on use-case requirements

**5.2 Validation checks**

- Ensure LOSO folds are identical across models (same subject order)
- Verify no test leakage in threshold selection
- Confirm subject-wise normalization computed correctly
- Check that all 3 threshold methods use same probabilities

**5.3 Reproducibility**

- Save random seeds
- Document Python package versions
- Include training logs
- Save model checkpoints

---

## Expected Outputs

### Results Files

```javascript

├── comparison/
│   ├── main_comparison_table.csv        # 15 rows (5 models × 3 thresholds)
│   ├── ablation_a_summary.json          # Model family comparison
│   ├── ablation_b_summary.json          # Thresholding strategy comparison
│   ├── ablation_c_summary.json          # Fusion strategy comparison
│   └── threshold_analysis.csv           # Threshold statistics per model
├── figures/
│   ├── recall_vs_far_tradeoff.png       # 5 curves with B1/B2/B3 markers
│   ├── subject_recall_distribution.png  # 5 boxplots
│   ├── confusion_matrices_combined.png  # 5 matrices (2×3 grid)
│   ├── precision_recall_curves.png      # 5 curves
│   ├── threshold_sensitivity_{model}.png # 5 separate plots
│   └── ablation_c_comparison.png        # Fusion strategies bar chart
├── lr/
│   ├── lr_b1_metrics.json
│   ├── lr_b2_metrics.json
│   ├── lr_b3_metrics.json
│   ├── lr_b1_fold_metrics.csv
│   ├── lr_b2_fold_metrics.csv
│   ├── lr_b3_fold_metrics.csv
│   └── lr_probabilities.csv             # Saved for fusion models
├── tcn/
│   ├── [same structure as lr/]
│   └── tcn_probabilities.csv
├── logical_or/
│   └── [same structure - no separate probabilities]
├── cascade/
│   └── [same structure]
└── stacked/
    └── [same structure]
```

---

## Key Design Decisions

### Why these 3 ablations?

- **Ablation A (Model Family)**: Does temporal modeling add value over features?
- **Ablation B (Thresholding)**: Deployment decisions - different use cases need different trade-offs
- **Ablation C (Fusion)**: How to best combine complementary models?
- Shows that model architecture, fusion strategy, and thresholding are **orthogonal** design choices

### Why subject-wise normalization?

- Handles inter-subject variability (baseline HR, skin temperature differ)
- LOSO-safe: stats computed per-subject, never across subjects
- Standard practice in physiological signal processing

### Why per-fold imputation?

- **Quality filtering first**: Drop windows where `hr_bpm` is NaN (HRV extraction completely failed)
- **Per-fold median imputation**: Fit imputer on training fold ONLY, then transform test fold
- **Critical for LOSO**: Test subject data never influences imputation statistics (no leakage)
- **For TCN**: Forward-fill in dataset (not zero-fill) for temporal continuity
- **No fallbacks**: If imputation fails, let it fail - don't silently fill with zeros
- Matches best practices from `experiments/classical_ml/train.py` (lines 663-668)

### Why 4Hz?

- Better frequency resolution for HRV and respiratory features
- Still computationally tractable
- Matches alignment in current codebase ([`shared/alignment.py`](experiments/shared/alignment.py))

---

## Timeline Estimate

- **Phase 1 (Setup)**: 15 minutes - create structure, config file only
- **Phase 2 (Training)**: 
- Classical ML (LR, RF, SVM): 45 min (3 models in one script)
- TCN: 2-3 hours
- Fusion (OR, Cascade, Stacked): 2-3 hours (Stacked needs nested LOSO)
- **Subtotal**: ~5-7 hours
- **Phase 3 (Visualization)**: 1 hour - generate all required plots
- **Phase 4 (Documentation)**: 30 minutes - write results summary

**Total**: ~6.5-8.5 hours**Simpler than original estimate** - minimal code, no unnecessary complexity

## Next Steps

1. Create `experiments/final_models/` folder structure
2. Write fixed configuration and shared pipeline
3. Write `config.py` with fixed hyperparameters
4. Write `train_classical_ml.py` - trains LR, RF, SVM with Ablation B (3 scripts → 1)
5. Write `train_tcn.py` - trains TCN with Ablation B
6. Write `train_fusion.py` - implements OR, Cascade, Stacked with Ablation B
7. Write `generate_figures.py` - creates all visualizations
8. Run training and generate results

**Key principles**:

- Minimal code - 3 training scripts (not 7)
- No helper functions unless absolutely necessary
- Reuse existing `shared/` modules extensively