# Aggregation Strategy Fix

## The Problem

### Original Flawed Approach (MOMENT Only)

```python
# WRONG: Re-compute predictions with averaged threshold
mean_threshold = np.mean([fold1_thresh, fold2_thresh, ...])
all_y_pred = (all_y_proba >= mean_threshold).astype(int)

aggregate_metrics = evaluate_predictions(all_y_true, all_y_pred, all_y_proba)
```

### Why This Was Problematic

**1. Statistical Meaninglessness**
- Averaging thresholds from different distributions is invalid
- Like averaging optimal cutoffs from different diseases
- Each fold has different class balance, subject physiology, optimal operating point

**2. Violates Constraints**
```python
Fold 1: threshold=0.30 → recall=0.87, FPR=0.18 ✅ (meets constraints)
Fold 2: threshold=0.25 → recall=0.86, FPR=0.19 ✅ (meets constraints)

Mean threshold=0.275 applied to Fold 1 data:
  → recall=0.75, FPR=0.22 ❌ (violates constraints!)
```

**3. Inconsistency**
- **Fold-level metrics**: Computed with fold-specific thresholds
- **Aggregate metrics**: Computed with mean threshold
- **Result**: `aggregate_metrics ≠ average(fold_metrics)` ❌

**4. Not What Actually Happened**
- Each fold used its own threshold during testing
- Aggregate should reflect **actual predictions made**, not re-thresholded versions

---

## The Correct Approach

### Store Fold-Level Predictions

```python
# Collect predictions made with fold-specific thresholds
all_y_pred = []

for each fold:
    # Find optimal threshold on TRAINING data
    fold_threshold = find_optimal_threshold(train_y, train_proba)
    
    # Apply to TEST data
    y_pred = (test_y_proba >= fold_threshold).astype(int)
    
    # Store actual predictions
    all_y_pred.extend(y_pred)

# Aggregate using actual predictions (NO threshold parameter!)
aggregate_metrics = evaluate_predictions(
    all_y_true,
    all_y_pred,  # Predictions made with fold-specific thresholds
    all_y_proba
    # NO threshold - we're using pre-computed predictions!
)

# Report threshold distribution (NOT a single value)
fold_thresholds = [f["threshold"] for f in fold_metrics]
aggregate_metrics["threshold_mean"] = np.mean(fold_thresholds)
aggregate_metrics["threshold_std"] = np.std(fold_thresholds)
aggregate_metrics["threshold_min"] = np.min(fold_thresholds)
aggregate_metrics["threshold_max"] = np.max(fold_thresholds)
```

### Why This Is Correct ✅

1. **Mathematically sound**: Uses actual predictions, no re-thresholding
2. **Consistent**: `aggregate_metrics ≈ average(fold_metrics)`
3. **Honest**: Reflects what actually happened during evaluation
4. **Preserves constraints**: If each fold met constraints, aggregate reflects that
5. **Informative**: Reports threshold distribution, not misleading "mean"

---

## Changes Made

### 1. MOMENT Training (`02_moment/train.py`) - **FIXED** ✅

**Before:**
```python
all_y_true = []
all_y_proba = []
all_subjects = []

# ... fold loop ...

# Re-threshold with mean
mean_threshold = np.mean([...])
all_y_pred = (all_y_proba >= mean_threshold).astype(int)
```

**After:**
```python
all_y_true = []
all_y_pred = []  # Store fold-level predictions
all_y_proba = []
all_subjects = []

# In fold loop:
y_pred = (y_proba >= fold_threshold).astype(int)
all_y_pred.extend(y_pred)  # Store actual predictions

# Aggregate with actual predictions (NO threshold!)
aggregate_metrics = evaluate_predictions(
    all_y_true,
    all_y_pred,  # No re-thresholding!
    all_y_proba
)

# Report threshold distribution
fold_thresholds = [f["threshold"] for f in fold_metrics]
aggregate_metrics["threshold_mean"] = np.mean(fold_thresholds)
aggregate_metrics["threshold_std"] = np.std(fold_thresholds)
aggregate_metrics["threshold_min"] = np.min(fold_thresholds)
aggregate_metrics["threshold_max"] = np.max(fold_thresholds)
```

### 2. Classical ML (`01_classical_ml/train.py`) - **ALREADY CORRECT** ✅

- Already stored fold-level predictions
- Added clarifying comments and mean_threshold reporting

### 3. SSL (`05_subject_aware_ssl/train.py`) - **ALREADY CORRECT** ✅

- Already used fold-level predictions
- Added clarifying comments

### 4. Multi-Rate Fusion (`06_multirate_fusion/train.py`) - **ALREADY CORRECT** ✅

- Already used fold-level predictions  
- Added clarifying comments

---

## Impact

### Before (MOMENT):
```python
# Fold 1: recall=0.87, threshold=0.30
# Fold 2: recall=0.85, threshold=0.32
# Fold 3: recall=0.90, threshold=0.28
# Average: recall≈0.87

# Aggregate with mean_threshold=0.30:
# → recall=0.79 ❌ (inconsistent!)
```

### After (All Experiments):
```python
# Fold 1: recall=0.87, threshold=0.30
# Fold 2: recall=0.85, threshold=0.32
# Fold 3: recall=0.90, threshold=0.28
# Average: recall≈0.87

# Aggregate using actual predictions:
# → recall≈0.87 ✅ (consistent!)
```

---

## Verification

To verify the fix works:

```python
# These should now be approximately equal:
avg_fold_recall = np.mean([f['recall'] for f in fold_metrics])
aggregate_recall = aggregate_metrics['recall']

assert abs(avg_fold_recall - aggregate_recall) < 0.01  # Should pass now!
```

---

## Key Takeaways

### What We Fixed:
1. ✅ **Store predictions** made with fold-specific thresholds
2. ✅ **No re-thresholding** when aggregating
3. ✅ **Consistent** fold and aggregate metrics
4. ✅ **Honest** reporting of actual performance

### What We Report:
- **Fold metrics**: Per-subject performance with optimal threshold
- **Aggregate metrics**: Overall performance using actual fold predictions
- **Threshold statistics**: Range (min-max), mean ± std - NOT a single "mean threshold"
  - Example: "Thresholds ranged from 0.28 to 0.35 (mean: 0.31 ± 0.02)"
  - This shows each fold adapted its threshold, not that we used one value

### Why It Matters:
- **Scientific validity**: Accurate reporting of model performance
- **Reproducibility**: Others can verify results
- **Clinical relevance**: Reflects real deployment where each subject gets optimal threshold
- **Constraint satisfaction**: Preserves guarantees (recall≥85%, FPR≤20%)

---

## Related Documents

- `THRESHOLD_METHOD_UPDATE.md` - Details on constrained G-Mean thresholding
- `experiments/shared/evaluation.py` - Core evaluation functions

---

## Date

Fixed: 2025-12-21
