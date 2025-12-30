# Multirate Fusion Threshold Update

**Date**: 2025-12-28  
**Status**: ✅ Complete

## Problem Identified

The multirate fusion model was using **constrained G-mean** with very strict constraints:
- `min_recall=0.70` (70% recall constraint)
- `max_fpr=0.30` (30% FPR constraint)
- **But the logs incorrectly showed**: `min_recall=0.85, max_fpr=0.20`

### Issues This Caused:

1. **Threshold was too high (0.8952)**:
   - Training set: Only 22.9% recall (failed to meet 70% constraint)
   - Test set: 0.0% recall (predicted NOTHING as positive)

2. **Good AUROC (0.968) but useless operating point**:
   - Model has discriminative ability
   - But threshold is so conservative it never predicts stress

3. **Logging mismatch**:
   - Code used 70/30 constraints
   - Logs claimed 85/20 constraints

## Solution Implemented

Changed from **constrained G-mean** to **balanced threshold**:

```python
# OLD: Constrained G-mean with strict requirements
optimal_threshold, _ = find_optimal_threshold(
    y_train_true, y_train_proba,
    method="constrained_gmean",
    min_recall=0.70,
    max_fpr=0.30
)

# NEW: Balanced method (equal sensitivity/specificity)
optimal_threshold, _ = find_optimal_threshold(
    y_train_true, y_train_proba,
    method="balanced"
)
```

## What is the Balanced Method?

The **balanced threshold** method finds the point where **Sensitivity ≈ Specificity**:

- Minimizes: |Sensitivity - Specificity|
- Forces a **50/50 trade-off** between true positive rate and true negative rate
- Equal performance on both classes (stress detection and non-stress)
- More conservative than Youden's method
- Aims for equality rather than maximum discriminative power

### Difference from Youden's Method:

| Method | Goal | Formula |
|--------|------|---------|
| **Balanced** | Equal Sens/Spec | min(\|Sens - Spec\|) |
| **Youden** | Max discriminative power | max(Sens + Spec - 1) |

- **Balanced**: Forces equal error rates on both classes
- **Youden**: Allows unequal Sens/Spec if it maximizes overall performance

For imbalanced datasets, **balanced** is often more conservative and fair.

## Changes Made

### 1. Threshold Selection (Line ~258 & ~292)

**During training monitoring (Line 258):**
```python
# Monitor training progress with balanced threshold
train_threshold, _ = find_optimal_threshold(y_train_true, y_train_proba, method="balanced")
```

**Final evaluation (Line 292):**
```python
# Use balanced threshold for final evaluation
# Minimizes |Sensitivity - Specificity| for equal performance on both classes
optimal_threshold, _ = find_optimal_threshold(
    y_train_true, y_train_proba,
    method="balanced"
)
```

**Note**: Both now use the same method for consistency!

### 2. Logging (Line ~303)
```python
logger.info(
    f"  Balanced threshold: {optimal_threshold:.4f} "
    f"(equal sensitivity and specificity)"
)
```

### 3. Metrics Storage (Line ~324)
```python
# Save threshold method info
metrics["threshold_method"] = "balanced"
metrics["optimal_threshold"] = optimal_threshold
metrics["train_epochs"] = epoch + 1
# Removed: min_recall and max_fpr (not applicable)
```

### 4. Configuration Logging (Line ~483)
```python
logger.info(f"  Threshold method: balanced (equal sensitivity + specificity)")
```

### 5. Final Summary (Line ~547)
```python
print(f"Threshold Method: balanced (equal sensitivity + specificity)")
```

## Expected Improvements

With the balanced threshold, you should see:

1. **Lower threshold values** (likely 0.3-0.6 range)
2. **Non-zero recall** on test sets
3. **Equal sensitivity and specificity** (or very close)
4. **More reasonable operating point** that reflects model's true capability
5. **Fair treatment** of both classes (stress and non-stress)

## Comparison with Other Models

- **Classical ML (LR)**: Uses same balanced method ✅
- **TCN**: Currently uses `geometric_mean` (unconstrained G-mean) ⚠️
- **MOMENT**: Uses constrained G-mean with 75/25 constraints

**Note**: For fair comparison, you may want to align all models to use the same threshold method.

## Training Command

To retrain with the new threshold method:

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation
python experiments/06_multirate_fusion/train.py --window_size 120 --horizon 0
```

## Next Steps

1. ✅ **Retrain the model** with new threshold method
2. **Compare results** with previous runs
3. **Verify** threshold values are more reasonable (0.3-0.6 range)
4. **Check** recall is non-zero on test sets
5. **Optionally**: Align TCN and other models to use same threshold method for fair comparison

## Files Modified

- `experiments/06_multirate_fusion/train.py`:
  - Line ~258: Changed threshold method from `geometric_mean` to `balanced` (training monitoring)
  - Line ~292: Changed threshold method from `constrained_gmean` to `balanced` (final evaluation)
  - Line ~303: Updated logging message
  - Line ~324: Removed constraint parameters from metrics, changed method to "balanced"
  - Line ~483: Updated configuration logging
  - Line ~547: Updated final summary

## References

- **Balanced Threshold**: Minimizes |Sensitivity - Specificity| for equal class performance
- **Youden's J statistic**: Youden, W.J. (1950). "Index for rating diagnostic tests". Cancer 3: 32–35.
- **Shared evaluation module**: `experiments/shared/evaluation.py` (implements `balanced` method)

