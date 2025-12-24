# MOMENT Unconstrained Training Update

**Date:** December 22, 2024  
**Status:** ✅ Implemented

## Changes Made

### 1. Freeze All Backbone Layers
**Before:**
```python
UNFREEZE_LAST_N_BLOCKS = 2  # Unfreeze last 2 transformer blocks
```

**After:**
```python
UNFREEZE_LAST_N_BLOCKS = 0  # Freeze ALL backbone, train head only
```

**Rationale:**
- With only 1717 windows (200 stress samples per fold), unfreezing risks overfitting
- Head-only training is safer with limited data
- Faster training (~2-3x speedup)
- Still leverages MOMENT's pre-trained representations

### 2. Remove Threshold Constraints
**Before:**
```python
THRESHOLD_METHOD = "constrained_gmean"
MIN_RECALL = 0.85  # Force ≥85% sensitivity
MAX_FPR = 0.20     # Force ≤20% false alarms
```

**After:**
```python
THRESHOLD_METHOD = "geometric_mean"
MIN_RECALL = 0.0   # No constraints
MAX_FPR = 1.0      # No constraints
```

**Rationale:**
- Let the model find its natural operating point
- Previous constraints were often unachievable (recall only reached ~0.65)
- Unconstrained G-Mean still balances recall and specificity
- More honest evaluation - no forced trade-offs

### 3. Use SimpleMOMENTClassifier
**Already set:**
```python
use_simple = True
```

**Benefit:**
- Lighter architecture
- Faster training with limited data
- Still uses MOMENT embeddings effectively

## Expected Performance Changes

### Training Speed
- **Before:** ~180-200 seconds/fold (with unfreezing)
- **After:** ~100-120 seconds/fold (head-only)
- **Speedup:** ~1.7-2x faster

### Model Performance
**Without constraints, expect:**
- **Recall:** May drop to 0.50-0.70 (natural balance point)
- **Precision:** May improve to 0.30-0.50 (fewer false alarms)
- **G-Mean:** Should stay ~0.70-0.80 (balanced)
- **AUROC:** Should remain 0.75-0.85 (discrimination unchanged)
- **F1-Score:** May improve (better precision-recall balance)

### Threshold Distribution
- **Before:** 0.28-0.35 (forced low to meet recall constraint)
- **After:** 0.45-0.60 (natural balance point, likely higher)

## Training Configuration Summary

```python
# Architecture
Model: SimpleMOMENTClassifier
Backbone: Frozen (all layers)
Trainable: Classification head only (~50K params)

# Training
Epochs: 100
Batch size: 16
Learning rate: 1e-3
Optimizer: Adam
Loss: CrossEntropyLoss with class weighting (9:1)

# Threshold
Method: geometric_mean (unconstrained)
Optimization: Maximize sqrt(recall × specificity)
No hard constraints on recall or FPR
```

## Why This Configuration?

### 1. Limited Data (1717 windows)
- Head-only training prevents overfitting
- Pre-trained MOMENT backbone provides strong features
- Don't need to fine-tune representations with so few samples

### 2. Severe Class Imbalance (12.4% stress)
- Weighted loss handles imbalance during training
- Let threshold optimization find natural balance
- Don't force constraints that may be impossible to meet

### 3. Realistic Evaluation
- Unconstrained thresholding shows what the model can actually achieve
- No artificially inflated recall from forced constraints
- More honest reporting for dissertation

## Comparison with Previous Runs

| Metric | Constrained (Target) | Unconstrained (Expected) |
|--------|---------------------|--------------------------|
| **Recall** | Target: ≥0.85<br>Actual: ~0.65 | Expected: 0.55-0.70 |
| **Precision** | ~0.25 | Expected: 0.30-0.45 |
| **G-Mean** | ~0.75 | Expected: 0.70-0.80 |
| **AUROC** | ~0.80 | Expected: 0.75-0.85 |
| **Threshold** | 0.28-0.35 (forced low) | 0.45-0.60 (natural) |
| **Training time** | ~180s/fold | ~100s/fold |

## Files Modified

- ✅ `experiments/02_moment/train.py`
  - Line 356: Changed UNFREEZE_LAST_N_BLOCKS to 0
  - Line 710: Changed THRESHOLD_METHOD to "geometric_mean"
  - Line 711-712: Set MIN_RECALL=0.0, MAX_FPR=1.0
  - Updated logging and comments throughout

## Next Steps

1. **Run training** and observe actual performance
2. **Compare with SSL** (also using unconstrained thresholding)
3. **Assess if results are competitive** with VitaStress paper baselines
4. **Document findings** in dissertation

## For Dissertation

**When reporting results:**
- ✅ State: "Head-only fine-tuning with unconstrained threshold optimization"
- ✅ Explain: "Given limited data (1717 windows), we freeze the pre-trained backbone to prevent overfitting"
- ✅ Justify: "Unconstrained G-Mean thresholding allows the model to find its natural balance point"
- ✅ Discuss: "This provides an honest evaluation of what the model can achieve without forced constraints"

**Limitations section:**
- Mention that constrained thresholding (recall ≥ 0.85) was attempted but proved infeasible with current data
- This is expected with severe class imbalance (12.4% stress)
- Future work with more data could explore stricter clinical requirements

---

**Last Updated:** December 22, 2024
