# Two-Stage Ensemble: LR + TCN

**Experiment 09**: Ensemble combining Logistic Regression and Temporal Convolutional Network with **multiple strategies**.

> 📖 **See [DECISION_STRATEGIES.md](./DECISION_STRATEGIES.md) for decision strategy comparison**  
> 📖 **See [STACKING_GUIDE.md](./STACKING_GUIDE.md) for stacked ensemble with nested LOSO**

## 🎯 Motivation

Individual models have complementary strengths:
- **Logistic Regression (LR)**: High recall (75%), but many false alarms (26% FAR)
- **Temporal Convolutional Network (TCN)**: High specificity (86%), but low recall (41%)

**Solution**: Use them in sequence to leverage both strengths:
1. **LR screens** with high sensitivity (catches most stress)
2. **TCN confirms** with high specificity (filters false alarms)

## 🏗️ Architecture

### Stage 1: Logistic Regression (Screening)
- **Input**: Extracted features from 1Hz aligned data + HRV metrics
- **Features**: ~100+ engineered features (statistical, temporal, frequency domain)
- **Threshold**: **Recall-first** - Minimum 75% recall
- **Role**: Wide net to catch potential stress events

### Stage 2: Temporal Convolutional Network (Confirmation)
- **Input**: Raw multichannel time series (8 channels × 120 timesteps)
- **Channels**: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd
- **Architecture**: 6-layer TCN with dilations [1,2,4,8,16,32], receptive field = 127
- **Threshold**: **FAR-first** - Maximum 25% false alarm rate
- **Role**: High-specificity filter to confirm stress

## 🔀 Decision Strategies

### Strategy A: LR-Only Decision (Default) ⭐

**Goal**: Maximize recall - catch all stress events

```python
# LR makes decision, TCN provides confidence
Final_Prediction = STRESS if LR_probability >= LR_threshold else NO_STRESS
Confidence = TCN_probability if Final_Prediction == STRESS else (1 - TCN_probability)
```

**When to use**: Your priority is catching all stress events (can tolerate false positives)

### Strategy B: Hard AND Cascade

**Goal**: Minimize false alarms - only predict stress when both agree

```python
if LR_probability < LR_threshold:
    Final_Prediction = NO STRESS
else:
    Final_Prediction = STRESS if TCN_probability >= TCN_threshold else NO_STRESS
```

**When to use**: Your priority is avoiding false alarms (can tolerate missed detections)

> 📖 **For detailed comparison and usage guide, see [DECISION_STRATEGIES.md](./DECISION_STRATEGIES.md)**

### Strategy C: Stacked Ensemble (Advanced) 🎓

**Goal**: Meta-learning for optimal combination

```python
# Meta-model learns optimal weights
meta_X = [LR_probability, TCN_probability]
Final_Prediction = meta_model.predict(meta_X)
```

**How it works:**
- Uses **nested LOSO** to generate out-of-fold predictions
- Trains meta-model (e.g., Logistic Regression, XGBoost) on these predictions
- Meta-model learns when to trust LR vs. TCN
- No data leakage (all predictions are out-of-fold)

**When to use:**
- You want maximum performance (+2-5% precision)
- You have time for longer training (~24 hours vs. 1.5 hours)
- You're doing final model evaluation

**Trade-offs:**
- ✅ Better precision and G-mean
- ⚠️ ~16× longer training time
- ⚠️ More complex (nested cross-validation)

> 📖 **For complete guide, see [STACKING_GUIDE.md](./STACKING_GUIDE.md)**

## ⚙️ Threshold Selection (Asymmetric)

### LR: Recall-First Threshold

**Objective**: Find **lowest** threshold that achieves minimum recall

```python
τ_LR = min{τ : Recall(τ) ≥ 0.75}
```

**Strategy**:
- Start from lowest threshold (highest recall)
- Find the highest threshold that still meets 75% recall
- Prioritizes **catching stress** over precision

**Example**:
- Target: 75% recall
- Selected: τ = 0.32 → Recall = 76%, FAR = 32%

### TCN: FAR-First Threshold

**Objective**: Find **highest** threshold that achieves maximum FAR

```python
τ_TCN = max{τ : FAR(τ) ≤ 0.25}
```

**Strategy**:
- Start from highest threshold (lowest FAR)
- Find the lowest threshold that still keeps FAR under 25%
- Prioritizes **avoiding false alarms** over recall

**Example**:
- Target: 25% FAR
- Selected: τ = 0.78 → FAR = 24%, Recall = 52%

## 📊 Expected Performance

Based on individual model performance:

| Metric | LR Alone | TCN Alone | **Strategy A<br>(LR-Only)** | **Strategy B<br>(AND Cascade)** | **Strategy C<br>(Stacked)** |
|--------|----------|-----------|---------------------------|------------------------------|---------------------------|
| **Recall** | 0.75 | 0.41 | **~0.85-0.90** ✓✓ | **~0.40-0.50** | **~0.75-0.88** ✓ |
| **Specificity** | 0.74 | 0.86 | **~0.74-0.76** | **~0.85-0.90** ✓✓ | **~0.76-0.82** ✓ |
| **Precision** | 0.31 | 0.30 | **~0.31-0.35** | **~0.40-0.55** ✓ | **~0.35-0.42** ✓✓ |
| **FAR** | 26% | 13.5% | **~24-26%** | **~10-15%** ✓✓ | **~18-24%** ✓ |
| **G-mean** | 0.74 | 0.60 | **~0.77-0.81** ✓ | **~0.60-0.67** | **~0.76-0.84** ✓✓ |
| **AUROC** | 0.828 | 0.727 | **~0.828** (same) | **~0.80-0.85** ✓ | **~0.82-0.85** ✓✓ |
| **Training Time** | 30 min | 1 hr | 1.5 hrs | 1.5 hrs | **24 hrs** ⚠️ |

**Strategy A Benefits (LR-Only):**
- ✅✅ **Highest recall** - catches 85-90% of stress events
- ✅ Provides **confidence scores** for interpretability
- ⚠️ Similar precision to LR (some false alarms)
- **Use when**: Missing stress is more costly than false alarms

**Strategy B Benefits (AND Cascade):**
- ✅✅ **Lowest false alarms** - high specificity filter
- ✅ **Better precision** - both models must agree
- ⚠️ Lower recall - TCN vetoes some true positives
- **Use when**: False alarms are more costly than missed detections

## 🔬 LOSO Cross-Validation

### Algorithm

```
FOR EACH test_subject IN subjects:
    1. Split: train = all others, test = current subject
    
    2. Train LR:
       - Extract features from train windows
       - Train with class_weight='balanced'
    
    3. Train TCN:
       - Use raw time series from train windows
       - Train with balanced class weights, early stopping
    
    4. Get TRAINING probabilities:
       - p_train_lr  = LR.predict_proba(train_data)
       - p_train_tcn = TCN.predict_proba(train_data)
    
    5. Select thresholds (TRAINING data only):
       - τ_LR  = recall_first_threshold(p_train_lr, min_recall=0.75)
       - τ_TCN = far_first_threshold(p_train_tcn, max_far=0.25)
    
    6. Test on held-out subject:
       - p_test_lr  = LR.predict_proba(test_data)
       - p_test_tcn = TCN.predict_proba(test_data)
    
    7. Apply two-stage decision:
       - if p_test_lr < τ_LR: y_pred = 0
       - else if p_test_tcn >= τ_TCN: y_pred = 1
       - else: y_pred = 0
    
    8. Evaluate and store metrics

9. Aggregate results across all folds
```

## 📁 File Structure

```
experiments/09_two_stage_ensemble/
├── __init__.py                  # Module initialization
├── train.py                     # Main training script
├── threshold_selection.py       # Asymmetric threshold utilities
├── README.md                    # This file
└── results/
    ├── training.log
    ├── two_stage_ensemble_metrics.json
    ├── two_stage_ensemble_predictions.csv
    ├── two_stage_ensemble_fold_metrics.csv
    └── figures/
        ├── two_stage_ensemble_confusion_matrix.png
        ├── two_stage_ensemble_roc.png
        └── two_stage_ensemble_pr.png
```

## 🚀 Usage

### Quick Start (Strategy A - LR-Only)

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation
python experiments/two_stage_ensemble/train.py
```

This runs with the **default strategy A (LR-Only)** optimized for high recall.

### Strategy A: LR-Only (Maximize Recall)

```python
results = loso_cross_validation(
    windows_by_subject,
    config=DEFAULT_CONFIG,
    device=torch.device("mps"),
    logger=logger,
    min_recall_lr=0.90,         # LR must catch 90% of stress
    max_far_tcn=0.25,           # Not used in this mode
    n_epochs_tcn=100,
    decision_strategy="lr_only"  # ← LR decides, TCN provides confidence
)
```

### Strategy B: AND Cascade (Minimize False Alarms)

```python
results = loso_cross_validation(
    windows_by_subject,
    config=DEFAULT_CONFIG,
    device=torch.device("mps"),
    logger=logger,
    min_recall_lr=0.75,            # LR screening threshold
    max_far_tcn=0.25,              # TCN must keep FAR under 25%
    n_epochs_tcn=100,
    decision_strategy="and_cascade"  # ← Both must agree
)
```

### Strategy C: Stacked Ensemble (Maximum Performance)

#### Command Line

```bash
# Logistic Regression meta-model (recommended)
python experiments/two_stage_ensemble/train.py --stacking

# XGBoost meta-model (requires xgboost)
python experiments/two_stage_ensemble/train.py --stacking --meta-model xgboost

# Random Forest meta-model
python experiments/two_stage_ensemble/train.py --stacking --meta-model random_forest

# Or use the convenience script (with confirmation prompt)
python experiments/two_stage_ensemble/run_stacking.py
```

#### Python API

```python
results = stacked_loso_cross_validation(
    windows_by_subject,
    config=DEFAULT_CONFIG,
    device=torch.device("mps"),
    logger=logger,
    n_epochs_tcn=100,
    meta_model_type="logistic_regression"  # or "xgboost", "random_forest"
)
```

⚠️ **Warning**: Stacking takes ~24 hours due to nested LOSO (vs. 1.5 hours for simple ensemble)

## 🔧 Hyperparameters

### Logistic Regression
- **C**: 5.0 (regularization strength)
- **penalty**: 'l1' (L1 regularization)
- **solver**: 'saga'
- **class_weight**: 'balanced'
- **max_iter**: 1000

### TCN
- **Channels**: [16, 16, 16, 16, 16, 16]
- **Dilations**: [1, 2, 4, 8, 16, 32]
- **Kernel size**: 3
- **Dropout**: 0.3
- **FC hidden**: 128
- **Pooling**: Last timestep
- **Epochs**: 100 (with early stopping, patience=15)
- **Learning rate**: 1e-3
- **Class weights**: sklearn balanced formula

### Threshold Selection
- **Decision strategy**: "lr_only" (default) or "and_cascade"
- **LR min_recall**: 0.90 (90%) for Strategy A, 0.75 (75%) for Strategy B
- **TCN max_far**: 0.25 (25%) - used only in Strategy B

## 📊 Output Metrics

### Per-Fold Metrics (CSV)
- subject, lr_threshold, tcn_threshold
- recall, precision, specificity, f1, gmean
- auroc, pr_auc, balanced_accuracy
- lr_train_recall, tcn_train_far

### Aggregated Metrics (JSON)
- Mean ± std for all metrics
- Overall confusion matrix
- Threshold statistics

## 🔍 Key Advantages

1. **Asymmetric Optimization**:
   - Each model optimized for its strength
   - LR: Recall-first (screening)
   - TCN: FAR-first (confirmation)

2. **No Leakage**:
   - Thresholds selected on TRAINING data only
   - Applied to TEST data for fair evaluation

3. **Subject-Independent**:
   - LOSO ensures generalization to new subjects
   - Each fold trains on N-1 subjects

4. **Computational Efficiency**:
   - TCN only runs when LR detects stress
   - Expected TCN calls: ~30-40% of samples

5. **Interpretability**:
   - Clear decision logic
   - Can trace why prediction was made
   - Separate threshold for each stage

## 🎓 Theoretical Foundation

### Why This Works

1. **Serial Filtering**:
   - LR acts as high-recall filter (Stage 1)
   - TCN acts as high-specificity filter (Stage 2)
   - Final output has better precision without sacrificing recall

2. **Asymmetric Loss Functions**:
   - Different costs for different stages
   - Stage 1: Minimize false negatives
   - Stage 2: Minimize false positives

3. **Model Diversity**:
   - LR: Linear, feature-based, fast
   - TCN: Non-linear, temporal patterns, deep
   - Complementary modeling approaches

## 📈 Expected vs. Actual Results

After training, compare:

**Expected**:
- Recall: 55-65%
- Specificity: 85-90%
- Precision: 40-50%
- G-mean: 70-77%

**Analysis Questions**:
1. Did TCN's high specificity improve precision?
2. Was recall maintained better than TCN alone?
3. How often did TCN reject LR's detections?
4. Are thresholds stable across folds?

## 🐛 Troubleshooting

### Strategy A (LR-Only)

**Issue**: Recall too low (< 85%)
- **Solution**: Increase `min_recall_lr` (try 0.95 or even 0.98)

**Issue**: Too many false alarms
- **Solution**: Switch to **Strategy B (AND Cascade)** or post-filter by confidence

**Issue**: Confidence scores are too low
- **Solution**: TCN confidence might be poorly calibrated - this is informational only

### Strategy B (AND Cascade)

**Issue**: Recall too low (< 40%)
- **Solution**: Increase `min_recall_lr` (try 0.80) or increase `max_far_tcn` (try 0.30)

**Issue**: Too many false alarms
- **Solution**: Decrease `max_far_tcn` (try 0.20 or 0.15)

**Issue**: TCN rejects too many true positives
- **Solution**: Switch to **Strategy A (LR-Only)** to preserve LR's recall

### General

**Issue**: Training takes too long
- **Solution**: Reduce `n_epochs_tcn` or use GPU/MPS

**Issue**: Models not converging
- **Solution**: Check data quality, increase `n_epochs_tcn`, or adjust learning rate

## 📚 References

1. **Asymmetric Threshold Optimization**:
   - Provost, F. (2000). "Machine Learning from Imbalanced Data Sets"
   
2. **Cascaded Classifiers**:
   - Viola, P., & Jones, M. (2001). "Rapid object detection using a boosted cascade"

3. **Base Models**:
   - Logistic Regression: `experiments/01_classical_ml/`
   - TCN: `experiments/08_tcn/`

## 📞 Contact

For questions or issues, check:
- Training logs: `results/training.log`
- Fold metrics: `results/two_stage_ensemble_fold_metrics.csv`
- Individual model results: `experiments/01_classical_ml/` and `experiments/08_tcn/`

---

**Last Updated**: 2025-12-28  
**Version**: 1.0.0

