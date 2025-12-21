# Threshold Parameter Fix in `evaluate_predictions()`

## The Problem

### Original Behavior (Misleading)

```python
def evaluate_predictions(y_true, y_pred, y_proba, model_name, threshold=None):
    # ALWAYS set and store threshold
    if threshold is None:
        threshold = 0.5
    metrics["threshold"] = float(threshold)  # ← Always stored!
    
    # Only use threshold if y_pred not provided
    if y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)
    
    # Compute metrics using y_pred
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    # ...
```

### Why This Was Misleading

**Case 1: Fold-level evaluation (y_pred provided)**
```python
# We provide y_pred computed with fold-specific threshold
evaluate_predictions(
    y_true, 
    y_pred=y_pred,  # Already thresholded at 0.31
    y_proba=y_proba,
    threshold=0.31  # Explicitly passed
)

# Result: ✅ Correct
# - Uses provided y_pred (computed with 0.31)
# - Stores threshold: 0.31
# - Metrics reflect 0.31 threshold
```

**Case 2: Aggregate evaluation (y_pred provided, threshold=None)**
```python
# We provide y_pred from multiple fold-specific thresholds
evaluate_predictions(
    all_y_true, 
    y_pred=all_y_pred,  # Mix of predictions from [0.28, 0.31, 0.35, ...]
    y_proba=all_y_proba
    # No threshold parameter
)

# OLD BEHAVIOR: ❌ Misleading
# - Uses provided y_pred (correct)
# - Defaults threshold to 0.5 (wrong!)
# - Stores threshold: 0.5 (misleading!)
# - Implies metrics were computed with 0.5 (false!)

# Result metrics:
{
    "threshold": 0.5,  # ← LIE! Not used
    "accuracy": 0.78,  # ← Computed from y_pred with various thresholds
    "recall": 0.87     # ← Computed from y_pred with various thresholds
}
```

**Problem**: Readers would think threshold=0.5 was used, when actually predictions came from fold-specific thresholds (0.28-0.35).

---

## The Fix

### New Behavior (Honest)

```python
def evaluate_predictions(y_true, y_pred, y_proba, model_name, threshold=None):
    # Only compute and store threshold when actually used
    if y_pred is None:
        # Need to compute y_pred from y_proba
        if y_proba is not None:
            if threshold is None:
                threshold = 0.5
            y_pred = (y_proba >= threshold).astype(int)
            metrics["threshold"] = float(threshold)  # ✅ Store because used
        else:
            return {"error": "..."}
    else:
        # y_pred already provided - threshold NOT used for classification
        if threshold is not None:
            metrics["threshold"] = float(threshold)  # ✅ Store if explicitly provided
        # else: Don't store threshold (not used, not provided)
    
    # Compute metrics using y_pred (provided or computed)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    # ...
```

### New Behavior Examples

**Case 1: Fold-level (threshold provided)**
```python
evaluate_predictions(
    y_true, 
    y_pred=y_pred,
    y_proba=y_proba,
    threshold=0.31  # ← Explicitly provided
)

# Result: ✅ Stores threshold: 0.31
# Honest: Shows which threshold was used for this fold
```

**Case 2: Aggregate (no threshold)**
```python
evaluate_predictions(
    all_y_true, 
    y_pred=all_y_pred,
    y_proba=all_y_proba
    # No threshold
)

# Result: ✅ No "threshold" key in metrics
# Honest: Doesn't claim a single threshold was used

# Threshold stats added separately:
metrics["threshold_mean"] = 0.31
metrics["threshold_std"] = 0.02
metrics["threshold_min"] = 0.28
metrics["threshold_max"] = 0.35
```

**Case 3: Compute predictions (no y_pred provided)**
```python
evaluate_predictions(
    y_true,
    y_pred=None,  # ← Need to compute
    y_proba=y_proba,
    threshold=0.4
)

# Result: ✅ Stores threshold: 0.4
# Correct: threshold was actually used to compute y_pred
```

---

## Impact

### Before Fix:

```python
# Aggregate metrics (MISLEADING)
{
    "threshold": 0.5,        # ← WRONG! Not used
    "accuracy": 0.78,
    "recall": 0.87,
    # ...
}
```

### After Fix:

```python
# Aggregate metrics (HONEST)
{
    # No "threshold" key (wasn't used)
    "accuracy": 0.78,
    "recall": 0.87,
    "threshold_mean": 0.31,  # ← Added by calling code
    "threshold_std": 0.02,   # ← Shows distribution
    "threshold_min": 0.28,
    "threshold_max": 0.35,
    # ...
}
```

---

## Why This Matters

### Scientific Integrity ✅

- **Transparent**: Only stores threshold when it's actually used
- **Honest**: Doesn't claim a single threshold when multiple were used
- **Reproducible**: Readers know exactly what happened

### Clinical Relevance ✅

- **Accurate reporting**: Threshold stats show adaptation to different subjects
- **No confusion**: Clear that LOSO uses per-fold thresholds
- **Proper interpretation**: Metrics reflect actual deployment scenario

### Prevents Misinterpretation ❌→✅

**Before**:
> "The model achieved 87% recall with threshold=0.5"
> ❌ False! Used different thresholds per subject

**After**:
> "The model achieved 87% recall using subject-specific thresholds (range: 0.28-0.35, mean: 0.31±0.02)"
> ✅ True! Accurately describes LOSO strategy

---

## Related Changes

This fix complements:
1. **AGGREGATION_FIX.md** - Store fold-level predictions (no re-thresholding)
2. **THRESHOLD_METHOD_UPDATE.md** - Constrained G-Mean thresholding
3. **Threshold statistics** - Report distribution instead of single value

Together, these ensure:
- Honest, accurate, scientifically valid evaluation
- Clear reporting of LOSO cross-validation strategy
- Transparent about threshold selection and usage

---

## Date

Fixed: 2025-12-21
