# Stacked Ensemble with Nested LOSO Cross-Validation

This guide explains how to use the **stacked ensemble** implementation, which uses a meta-model to optimally combine LR and TCN predictions.

---

## 🎯 What is Stacking?

**Stacking** (stacked generalization) trains a meta-model to learn the optimal way to combine base model predictions:

```
┌─────────┐                     ┌──────────────┐
│   LR    │──→ P_lr    ┐        │              │
└─────────┘            ├───→    │  Meta-Model  │──→ Final
                       │        │ (Learns when │   Prediction
┌─────────┐            │        │  to trust    │
│   TCN   │──→ P_tcn   ┘        │  each model) │
└─────────┘                     └──────────────┘
```

The meta-model learns:
- When to trust LR vs. TCN
- How to weight each model's predictions
- Non-linear combinations of probabilities

---

## 🔄 Nested LOSO Algorithm (No Data Leakage)

### For Each Outer Fold (e.g., Test = S21):

```
OUTER FOLD: Test = S21, Train = S1-S20

├── PHASE 1: Nested LOSO (Generate Meta-Features)
│   ├── Inner Fold 1: Train on S2-S20 → Predict S1 (out-of-fold)
│   ├── Inner Fold 2: Train on S1,S3-S20 → Predict S2 (out-of-fold)
│   ├── Inner Fold 3: Train on S1-S2,S4-S20 → Predict S3 (out-of-fold)
│   └── ... (20 inner folds total)
│   Result: Out-of-fold predictions for all training subjects (S1-S20)
│
├── PHASE 2: Train Meta-Model
│   └── Fit on out-of-fold predictions (no leakage!)
│
├── PHASE 3: Train Final Base Models
│   ├── LR trained on all S1-S20
│   └── TCN trained on all S1-S20
│
├── PHASE 4: Predict on Test Subject
│   ├── LR predicts S21
│   └── TCN predicts S21
│
└── PHASE 5: Meta-Model Final Prediction
    └── Combines LR and TCN predictions for S21
```

**Key Point**: Each outer fold trains its own meta-model and base models!

---

## 🚀 Usage

### Basic Usage (Logistic Regression Meta-Model)

```bash
python experiments/two_stage_ensemble/train.py --stacking
```

### With XGBoost Meta-Model

```bash
python experiments/two_stage_ensemble/train.py --stacking --meta-model xgboost
```

### With Random Forest Meta-Model

```bash
python experiments/two_stage_ensemble/train.py --stacking --meta-model random_forest
```

### Regular (Non-Stacked) Ensemble

```bash
python experiments/two_stage_ensemble/train.py
# OR explicitly:
python experiments/two_stage_ensemble/train.py --no-stacking
```

---

## ⚙️ Meta-Model Options

### 1. Logistic Regression (Default, Recommended)

```python
meta_model_type="logistic_regression"
```

**Pros:**
- ✅ Fast training
- ✅ Interpretable (see feature weights)
- ✅ No hyperparameter tuning needed
- ✅ Works well for 2 features (LR prob, TCN prob)

**Cons:**
- ⚠️ Only learns linear combinations

**Best for**: Understanding what the meta-model learns

### 2. XGBoost

```python
meta_model_type="xgboost"
```

**Pros:**
- ✅ Non-linear combinations
- ✅ Handles feature interactions automatically
- ✅ Usually best performance

**Cons:**
- ⚠️ Requires XGBoost installation (`pip install xgboost`)
- ⚠️ Less interpretable
- ⚠️ Might overfit with only 2 features

**Best for**: Maximum performance

### 3. Random Forest

```python
meta_model_type="random_forest"
```

**Pros:**
- ✅ Non-linear combinations
- ✅ Robust to outliers
- ✅ No hyperparameter tuning needed

**Cons:**
- ⚠️ Less interpretable
- ⚠️ Might be overkill for 2 features

**Best for**: Balanced approach between LR and XGBoost

---

## ⏱️ Computational Cost

### For 21 Subjects:

| Ensemble Type | Outer Folds | Inner Folds | Model Trainings | Time Estimate |
|--------------|-------------|-------------|-----------------|---------------|
| **Simple** | 21 | 0 | 42 | ~1.5 hours |
| **Stacked** | 21 | 420 (20×21) | 903 | **~24 hours** |

**Time per fold** (with TCN):
- Simple: ~4 minutes
- Stacked: ~1 hour (due to 20 inner folds)

**Breakdown for ONE outer fold**:
```
Inner loop:  20 folds × (1 LR + 1 TCN) = 40 trainings × ~3 min = ~2 hours
Meta-model:  1 training × <1 sec = negligible
Final models: 1 LR + 1 TCN = 2 trainings × ~3 min = ~6 min
Total per outer fold: ~2 hours
```

---

## 📊 Expected Performance

Based on typical stacking results:

| Metric | Simple Ensemble | Stacked Ensemble | Improvement |
|--------|----------------|------------------|-------------|
| **Recall** | 75-85% | 75-88% | +0-3% |
| **Precision** | 30-35% | 32-40% | **+2-5%** ✓ |
| **G-mean** | 74-80% | 76-82% | **+2-3%** ✓ |
| **AUROC** | 0.80-0.83 | 0.82-0.85 | **+2-3%** ✓ |

**Where Stacking Helps Most:**
- ✅ **Precision**: Meta-model learns when TCN's high specificity is reliable
- ✅ **G-mean**: Better balance between recall and specificity
- ✅ **AUROC**: Better probability calibration

**Where Stacking Doesn't Help Much:**
- ⚠️ **Recall**: Can't recover what base models missed
- ⚠️ **FAR**: Similar to simple ensemble

---

## 🔍 Understanding the Meta-Model

### Example: Logistic Regression Meta-Model

After training, you'll see:

```
Meta-model weights: LR=1.234, TCN=0.876
Meta-model intercept: -0.543
```

**Interpretation**:

```python
# Meta-model learned this decision function:
stress_score = 1.234 × LR_proba + 0.876 × TCN_proba - 0.543
prediction = 1 if stress_score > 0 else 0
```

**Example Decisions**:

| LR Proba | TCN Proba | Stress Score | Prediction | Reasoning |
|----------|-----------|--------------|------------|-----------|
| 0.9 | 0.2 | 1.234×0.9 + 0.876×0.2 - 0.543 = **0.80** | ✅ Stress | LR confident, weight LR more |
| 0.6 | 0.7 | 1.234×0.6 + 0.876×0.7 - 0.543 = **0.61** | ✅ Stress | Both agree |
| 0.8 | 0.1 | 1.234×0.8 + 0.876×0.1 - 0.543 = **0.53** | ✅ Stress | LR very confident |
| 0.3 | 0.8 | 1.234×0.3 + 0.876×0.8 - 0.543 = **0.53** | ✅ Stress | TCN confident |
| 0.4 | 0.3 | 1.234×0.4 + 0.876×0.3 - 0.543 = **0.19** | ❌ No stress | Neither confident |

---

## 🐛 Troubleshooting

### Issue: Training is too slow

**Solution 1**: Use fewer epochs
```bash
# Edit train.py, line ~985:
n_epochs_tcn=50  # Instead of 100
```

**Solution 2**: Use GPU/MPS
```bash
# Automatically detected, just ensure PyTorch has GPU support
```

**Solution 3**: Skip stacking, use simple ensemble
```bash
python experiments/two_stage_ensemble/train.py
```

### Issue: Meta-model weights seem wrong

**Example**: `LR=-2.5, TCN=3.1` (negative LR weight?)

**Explanation**: This can happen if:
- LR is overconfident on false positives
- Meta-model learned to downweight LR
- This is actually correct behavior!

**Check**: Look at fold-level base model performance to verify

### Issue: Stacking performs worse than simple ensemble

**Possible causes**:
1. **Overfitting**: Meta-model overfitted to training distribution
2. **Small dataset**: Not enough data for meta-learning
3. **Model diversity**: Base models too similar

**Solutions**:
- Try simpler meta-model (Logistic Regression)
- Check if base models are too correlated
- Use simple ensemble instead

### Issue: XGBoost not found

```bash
pip install xgboost
```

If installation fails, use Logistic Regression:
```bash
python experiments/two_stage_ensemble/train.py --stacking --meta-model logistic_regression
```

---

## 📈 When to Use Stacking vs. Simple Ensemble

### Use **Simple Ensemble** if:
- ✅ Time is limited (need results quickly)
- ✅ Base models are very different (LR vs. TCN) - simple average works well
- ✅ You want interpretability
- ✅ You're doing exploratory analysis

### Use **Stacked Ensemble** if:
- ✅ You have 24+ hours for training
- ✅ You want maximum performance (every % matters)
- ✅ You're doing final model evaluation for paper/thesis
- ✅ You want to see how much meta-learning helps

---

## 📊 Output Files

Stacking produces the same output structure as simple ensemble:

```
experiments/two_stage_ensemble/results/
├── training.log                         # Full training log
├── stacked_ensemble_metrics.json        # Overall metrics
├── stacked_ensemble_predictions.csv     # Per-sample predictions
├── stacked_ensemble_fold_metrics.csv    # Per-fold metrics
└── figures/
    ├── stacked_ensemble_confusion_matrix.png
    ├── stacked_ensemble_roc.png
    └── stacked_ensemble_pr.png
```

---

## 🔬 Comparison with Other Strategies

| Strategy | Recall | Precision | Training Time | Complexity |
|----------|--------|-----------|---------------|------------|
| **LR Only** | 85-90% | 30-35% | 30 min | Low |
| **TCN Only** | 40-50% | 30-35% | 1 hour | Medium |
| **LR-Only (Strategy A)** | 85-90% | 30-35% | 1.5 hours | Low |
| **AND Cascade (Strategy B)** | 40-50% | 55-65% | 1.5 hours | Low |
| **Soft Voting** | 75-85% | 32-38% | 1.5 hours | Low |
| **Stacked Ensemble** | 75-88% | 35-42% | **24 hours** | **High** |

**Recommendation**: 
- For your use case (catching all stress = priority), **LR-Only Strategy A** is best
- Stacking might improve precision by 2-5%, but won't improve recall much

---

## 💡 Key Takeaways

1. **Stacking uses nested LOSO** to avoid data leakage
2. **Each outer fold trains new models** (meta + base)
3. **Out-of-fold predictions** are crucial for valid meta-training
4. **Computational cost is ~16× higher** than simple ensemble
5. **Performance gain is typically +2-5%** in key metrics
6. **Best for precision/specificity**, not recall

---

## 📚 References

1. **Wolpert, D. H. (1992)** - "Stacked generalization" - Neural Networks
2. **Breiman, L. (1996)** - "Stacked regressions" - Machine Learning
3. **Nested CV**: Varma & Simon (2006) - "Bias in error estimation when using cross-validation"

---

## 📞 Support

For questions or issues:
- Check training log: `results/training.log`
- Compare with simple ensemble results
- Verify base model performance first (LR and TCN individually)

**Last Updated**: 2025-12-30  
**Version**: 1.0.0

