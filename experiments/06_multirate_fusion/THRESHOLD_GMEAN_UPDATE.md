# Multirate Fusion - Threshold Method Update

**Date**: 2025-12-28  
**Status**: ✅ Updated to Geometric Mean

## Changes Made

Changed threshold method from **"balanced"** to **"geometric_mean"** (unconstrained) throughout the multirate fusion training script.

### What Changed:

**From: Balanced Threshold**
- Method: `"balanced"`
- Objective: Minimize |Sensitivity - Specificity|
- Result: Forces equal sensitivity and specificity

**To: Unconstrained Geometric Mean**
- Method: `"geometric_mean"`
- Objective: Maximize √(Sensitivity × Specificity)
- Result: Optimal G-mean without constraints

## Modified Locations:

### 1. Training Progress Monitoring (Line ~268)
```python
train_threshold, _ = find_optimal_threshold(y_train_true, y_train_proba, method="geometric_mean")
```

### 2. Final Threshold Selection (Line ~304)
```python
# Use unconstrained geometric mean threshold
# Maximizes sqrt(Sensitivity × Specificity) for optimal G-mean
optimal_threshold, _ = find_optimal_threshold(
    y_train_true, y_train_proba,
    method="geometric_mean"
)
```

### 3. Logging (Line ~307)
```python
logger.info(
    f"  Geometric mean threshold: {optimal_threshold:.4f} "
    f"(maximizes sqrt(Sensitivity × Specificity))"
)
```

### 4. Metrics Storage (Line ~329)
```python
metrics["threshold_method"] = "geometric_mean"
```

### 5. Configuration Logging (Line ~493)
```python
logger.info(f"  Threshold method: geometric_mean (unconstrained, maximizes G-mean)")
```

### 6. Final Summary (Line ~557)
```python
print(f"Threshold Method: geometric_mean (unconstrained, maximizes G-mean)")
```

## What to Expect:

### Geometric Mean Characteristics:
- **Optimizes**: √(Sensitivity × Specificity)
- **Balance**: Naturally seeks balance but doesn't force equality
- **Flexibility**: Can favor one metric slightly if it improves overall G-mean
- **No constraints**: No minimum recall or maximum FPR limits

### Comparison with Other Methods:

| Method | Objective | Constraints | Best For |
|--------|-----------|-------------|----------|
| **geometric_mean** | max(√(Sens × Spec)) | None | Overall discriminative power |
| **balanced** | min(\|Sens - Spec\|) | Equal Sens/Spec | Fair treatment of both classes |
| **constrained_gmean** | max(√(Sens × Spec)) | min_recall, max_fpr | Clinical safety requirements |
| **youden** | max(Sens + Spec - 1) | None | ROC-based optimization |

### Expected Results:
- **More flexible thresholds**: Not forced to equality
- **Better G-mean scores**: Optimized directly for this metric
- **Natural balance**: Usually close to balanced but can deviate if beneficial
- **Threshold range**: Typically 0.3-0.6 for imbalanced datasets

## Current Configuration Summary:

| Component | Setting |
|-----------|---------|
| **Threshold Method** | `geometric_mean` (unconstrained) |
| **Class Weights** | sklearn balanced: `[0.57, 4.07]` (1:7 ratio) |
| **Loss Function** | Weighted CrossEntropyLoss |
| **Optimizer** | AdamW with ReduceLROnPlateau |
| **Early Stopping** | Patience = 10 epochs |

## Comparison with Other Models:

| Model | Threshold Method | Notes |
|-------|------------------|-------|
| **Multirate Fusion** | `geometric_mean` | ✅ Now unconstrained |
| **TCN** | `geometric_mean` | ✅ Same method |
| **Classical ML (LR)** | `youden` | Different, but similar objective |
| **MOMENT** | `constrained_gmean` | With 75/25 constraints |

## Files Modified:

- `experiments/06_multirate_fusion/train.py`:
  - Line ~268: Training monitoring threshold
  - Line ~304: Final threshold selection
  - Line ~307: Threshold logging
  - Line ~329: Metrics storage
  - Line ~493: Configuration logging
  - Line ~557: Final summary

## Training Command:

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation
python experiments/06_multirate_fusion/train.py --window_size 120 --horizon 0
```

## Expected Log Output:

```
Threshold method: geometric_mean (unconstrained, maximizes G-mean)
...
Geometric mean threshold: 0.4523 (maximizes sqrt(Sensitivity × Specificity))
```

## Why Geometric Mean?

1. **Direct optimization**: Optimizes the exact metric you care about (G-mean)
2. **Natural balance**: Seeks balance because √(Sens × Spec) is highest when both are similar
3. **Flexibility**: Can slightly favor one metric if it improves overall score
4. **No arbitrary constraints**: No need to set min_recall or max_fpr
5. **Standard approach**: Widely used in imbalanced classification

## Note on Consistency:

Both **training monitoring** (line 268) and **final evaluation** (line 304) now use the same `"geometric_mean"` method, ensuring consistency between training logs and final metrics.

