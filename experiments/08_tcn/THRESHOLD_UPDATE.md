# TCN Threshold Method Update

## Changes Made

Updated the threshold selection in `train.py` to use **UNCONSTRAINED geometric mean** optimization.

## Previous Configuration

```python
fold_threshold, train_thresh_metrics = find_optimal_threshold(
    train_y_true, train_y_proba, 
    method="geometric_mean"
    # Uses defaults: min_recall=0.85, max_fpr=0.20
)
```

**Constraints:**
- Minimum recall: 85% (forces high sensitivity)
- Maximum FPR: 20% (forces low false positive rate)

These constraints **restrict the threshold search space** and may not find the true optimal point for geometric mean.

## New Configuration

```python
fold_threshold, train_thresh_metrics = find_optimal_threshold(
    train_y_true, train_y_proba, 
    method="geometric_mean",
    min_recall=0.0,  # No minimum recall constraint
    max_fpr=1.0      # No maximum FPR constraint
)
```

**No Constraints:**
- Minimum recall: 0% (fully unconstrained)
- Maximum FPR: 100% (fully unconstrained)

This allows the optimizer to find the **true optimal threshold** that maximizes:

```
Geometric Mean = √(Sensitivity × Specificity)
```

without any artificial constraints.

## Why This Matters

### Geometric Mean (G-mean)

The geometric mean is designed to find the **balance point** between sensitivity and specificity. It naturally penalizes extreme values:

- If sensitivity = 100% but specificity = 10% → G-mean = 31.6% (bad)
- If sensitivity = 50% but specificity = 50% → G-mean = 50.0% (better)
- If sensitivity = 80% but specificity = 80% → G-mean = 80.0% (best)

### Impact of Constraints

With constraints `min_recall=0.85, max_fpr=0.20`:
- Forces sensitivity ≥ 85%
- Forces specificity ≥ 80% (since FPR ≤ 20% means specificity ≥ 80%)
- **Problem**: The true optimal G-mean might be at sensitivity=75%, specificity=85%
- The constrained version would **miss this optimal point**

### Unconstrained Optimization

With `min_recall=0.0, max_fpr=1.0`:
- Searches **entire ROC curve**
- Finds the point that **maximizes G-mean** without bias
- More appropriate for research/comparison purposes
- Reflects the **true capability** of the model

## When to Use Constraints

Constraints are useful for:
- **Clinical deployment**: "We need at least 90% sensitivity to catch most cases"
- **Operational requirements**: "FPR must be below 10% due to alert fatigue"
- **Regulatory compliance**: "Minimum recall mandated by policy"

For **research and model comparison**, unconstrained optimization is preferred.

## Expected Changes

After this update, you may see:
- **Different threshold values** per fold
- Potentially **lower sensitivity** but **higher specificity** (or vice versa)
- **More balanced** sensitivity/specificity trade-off
- G-mean values that better reflect the **true optimal point**

The overall G-mean might be **higher or lower** depending on whether the previous constraints were helping or hurting performance.

## Training

To train with the new unconstrained threshold method:

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments/08_tcn
python train.py
```

The model will now find optimal thresholds without constraints, allowing for a fair comparison with other methods.

## Comparison with MOMENT

This change brings TCN in line with MOMENT's threshold selection approach, enabling **apples-to-apples comparison** of model performance.

