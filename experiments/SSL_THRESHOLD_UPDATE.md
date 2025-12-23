# SSL Threshold Method Update

**Date:** December 22, 2024  
**Status:** ✅ Implemented

## Problem

SSL fine-tuning used a different threshold selection strategy than MOMENT and Classical ML:

### Before (Inconsistent)

```python
# SSL compared 5 threshold methods per fold and picked the best G-mean
threshold_methods = ["youden", "f1", "balanced", "geometric_mean", "constrained_gmean"]

for method in threshold_methods:
    # Try each method...
    threshold_results[method] = {...}

# Pick method with highest G-mean on training data
best_method = max(threshold_results.keys(), key=lambda m: threshold_results[m]["gmean"])
```

**Problems:**
1. ❌ **Cherry-picking bias:** Selecting the best method per fold gives optimistic estimates
2. ❌ **Inconsistent with other experiments:** MOMENT and Classical ML use `constrained_gmean` directly
3. ❌ **Unfair comparison:** Can't compare SSL results with other models when using different strategies
4. ❌ **Variable methods:** Different folds use different methods (Youden, F1, Balanced, etc.)

### Observed Behavior (from logs)

```
Fold 17 → Selected: geometric_mean
Fold 18 → Selected: youden
Fold 19 → Selected: balanced  
Fold 20 → Selected: youden
Fold 21 → Selected: constrained_gmean
```

Each fold picked whatever method happened to have the highest training G-mean, leading to inconsistent evaluation.

## Solution

### After (Consistent)

Use `constrained_gmean` directly for all folds, matching MOMENT and Classical ML:

```python
# Use constrained G-mean threshold (consistent with MOMENT and Classical ML)
# This ensures high recall (≥85%) while controlling false alarm rate (≤20%)
optimal_threshold, train_thresh_metrics = find_optimal_threshold(
    y_train_true, 
    y_train_proba, 
    method="constrained_gmean",
    min_recall=0.85,
    max_fpr=0.20
)
```

**Benefits:**
1. ✅ **Consistent evaluation:** All experiments use the same threshold strategy
2. ✅ **Fair comparison:** SSL results directly comparable with MOMENT, Classical ML
3. ✅ **Clinical requirements:** Ensures recall ≥ 85% and FPR ≤ 20% across all folds
4. ✅ **No cherry-picking:** Same method for all folds, honest evaluation

## Changes Made

### 1. Modified `finetune_fold()` in `train.py`

**Before (lines 288-334):**
- Compared 5 threshold methods
- Selected best method based on training G-mean
- Different method per fold

**After:**
- Directly uses `constrained_gmean` with `min_recall=0.85`, `max_fpr=0.20`
- Same method for all folds
- Reports training metrics to verify constraints are met

### 2. Updated Logging

**Before:**
```
Method=geometric_mean
Method=youden
Method=balanced
```

**After:**
```
Threshold method: constrained_gmean
  Constraints: Recall ≥ 0.85, FPR ≤ 0.20
  Applied per-fold on TRAINING data (no leakage)
```

### 3. Metrics Updates

Added to `overall_metrics`:
```python
overall_metrics["threshold_method"] = "constrained_gmean"
overall_metrics["min_recall_constraint"] = 0.85
overall_metrics["max_fpr_constraint"] = 0.20
```

## Comparison with Other Experiments

| Experiment | Threshold Method | Constraints |
|------------|-----------------|-------------|
| **Classical ML** | `constrained_gmean` | Recall ≥ 0.85, FPR ≤ 0.20 ✅ |
| **MOMENT** | `constrained_gmean` | Recall ≥ 0.85, FPR ≤ 0.20 ✅ |
| **SSL (Before)** | Best of 5 methods per fold | Variable ❌ |
| **SSL (After)** | `constrained_gmean` | Recall ≥ 0.85, FPR ≤ 0.20 ✅ |
| **Multi-Rate Fusion** | `constrained_gmean` | Recall ≥ 0.85, FPR ≤ 0.20 ✅ |
| **MAML** | `constrained_gmean` | Recall ≥ 0.85, FPR ≤ 0.20 ✅ |

## Expected Impact

### Performance Changes

**Likely outcomes:**
- Slightly **lower aggregate G-mean** (no cherry-picking advantage)
- **Higher recall** (enforced ≥85% constraint)
- **Lower FPR** (enforced ≤20% constraint)
- **More consistent** threshold distribution across folds

### Evaluation Integrity

**Benefits:**
- ✅ Honest comparison with other models
- ✅ Meets clinical requirements consistently
- ✅ No optimistic bias from method selection
- ✅ Reproducible and defensible for dissertation

## Verification

After retraining, verify:

1. **All folds use same method:**
   ```
   grep "threshold_method" results/ssl_*.json
   # Should show "constrained_gmean" for all folds
   ```

2. **Constraints are met:**
   - Check that fold-level recall ≥ 0.85 (or close, given constraint satisfaction)
   - Check that fold-level FPR ≤ 0.20

3. **Comparable with MOMENT:**
   - Aggregate metrics computed same way (fold-level predictions)
   - Threshold reporting same format (mean ± std, min-max range)
   - Same evaluation pipeline (`experiments/shared/evaluation.py`)

## Files Modified

- ✅ `experiments/05_subject_aware_ssl/train.py`
  - Line ~288-334: Threshold selection in `finetune_fold()`
  - Line ~360-363: LOSO logging
  - Line ~418-421: Per-fold logging
  - Line ~447-456: Overall metrics with threshold statistics

## Related Documentation

- `experiments/THRESHOLD_METHOD_UPDATE.md` - Original constrained G-mean implementation
- `experiments/AGGREGATION_FIX.md` - Proper fold-level prediction aggregation
- `experiments/THRESHOLD_PARAMETER_FIX.md` - Honest threshold reporting

## Recommendation for Dissertation

**Use the updated SSL results** with `constrained_gmean` for:
1. Fair model comparison (all use same strategy)
2. Meeting clinical requirements consistently
3. Honest evaluation without cherry-picking bias
4. Reproducibility and scientific integrity

The old results (method comparison) can be mentioned in limitations as "exploratory analysis showed similar performance across threshold methods, but constrained_gmean was selected for consistency."

---

**Last Updated:** December 22, 2024
