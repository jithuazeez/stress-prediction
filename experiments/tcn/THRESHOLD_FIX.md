# Critical Bug Fix: Constrained Threshold Parameters

## Date: 2025-12-28

## Bug Description

**Symptom:** Even when using `threshold_method="constrained_gmean"` with `min_recall=0.70` and `max_fpr=0.30`, model was still getting low recall (40%) on test folds.

**Root Cause:** Hardcoded constraint values in `loso_cross_validation` function were overriding the parameters passed from `main()`.

---

## The Bug

### Location: `train.py` lines 508-509

**Before (Buggy Code):**
```python
fold_threshold, train_thresh_metrics = find_optimal_threshold(
    train_y_true, train_y_proba, 
    method=threshold_method,
    min_recall=0.75,  # ❌ HARDCODED!
    max_fpr=0.25      # ❌ HARDCODED!
)
```

**Problem:**
- Function signature accepts `min_recall` and `max_fpr` parameters
- But these were IGNORED inside the function
- Hardcoded values (0.75, 0.25) were used instead
- When user passed (0.70, 0.30), they were silently ignored

---

## The Fix

**After (Fixed Code):**
```python
fold_threshold, train_thresh_metrics = find_optimal_threshold(
    train_y_true, train_y_proba, 
    method=threshold_method,
    min_recall=min_recall,  # ✅ Use parameter from function signature
    max_fpr=max_fpr         # ✅ Use parameter from function signature
)
```

**Additional Improvement:**
```python
# Added logging to show training performance
train_sens = train_thresh_metrics.get("sensitivity", float("nan"))
train_spec = train_thresh_metrics.get("specificity", float("nan"))
train_gmean_val = train_thresh_metrics.get("gmean", float("nan"))
logger.info(f"  Training @ thr={fold_threshold:.3f}: Sens={train_sens:.3f}, Spec={train_spec:.3f}, Gmean={train_gmean_val:.3f}")
```

Now you can **verify** the constraint is being met on training data!

---

## Impact

### Before Fix:
```
User sets: min_recall=0.70, max_fpr=0.30
Code uses: min_recall=0.75, max_fpr=0.25  ← Different!
Result: Constraints might not be satisfiable → fallback to unconstrained
```

### After Fix:
```
User sets: min_recall=0.70, max_fpr=0.30
Code uses: min_recall=0.70, max_fpr=0.30  ✅ Correct!
Result: Constraints are actually applied as intended
```

---

## Why This Matters

### Constraint Satisfiability

With 1:7 class imbalance:

**Constraint: min_recall=0.75, max_fpr=0.25 (Strict)**
- Requires: Catch 75% stress AND keep false alarms ≤ 25%
- For poorly-performing folds: **Might not be achievable**
- Result: Falls back to unconstrained G-mean

**Constraint: min_recall=0.70, max_fpr=0.30 (Relaxed)**
- Requires: Catch 70% stress AND keep false alarms ≤ 30%
- Slightly easier to achieve
- Result: **More likely to satisfy constraint**

---

## How to Verify Fix

### In Your Training Logs

**Look for this new line:**
```
Training @ thr=0.XXX: Sens=0.YYY, Spec=0.ZZZ, Gmean=0.WWW
```

**Check if:**
1. **Sens ≥ min_recall** (should be ≥ 0.70 in your case)
2. **FPR ≤ max_fpr** (FPR = 1 - Spec, should be ≤ 0.30)

**Example (Good):**
```
Training @ thr=0.550: Sens=0.720, Spec=0.750, Gmean=0.735
```
- Sens = 72% ≥ 70% ✅
- FPR = 1 - 0.75 = 25% ≤ 30% ✅
- Constraint satisfied!

**Example (Fallback):**
```
UserWarning: No thresholds satisfy constraints (recall≥0.70, FPR≤0.30)
Training @ thr=0.650: Sens=0.550, Spec=0.850, Gmean=0.684
```
- Sens = 55% < 70% ❌
- This means model couldn't meet constraint
- Fell back to unconstrained G-mean

---

## Expected Behavior After Fix

### On Training Data:
Every fold should show:
```
Training @ thr=X: Sens≥0.70, FPR≤0.30
```

### On Test Data:
- **Most folds:** Test recall should be close to training recall (60-80%)
- **Some folds:** Test recall might be lower (poor generalization)
- **No folds:** Should collapse to 10-20% like before (that was due to high threshold)

---

## Related Issues

### If Constraint Still Can't Be Met:

If you STILL see warnings about unsatisfiable constraints, try:

**Option 1: Relax Constraints More**
```python
min_recall=0.65,  # Even more relaxed
max_fpr=0.35
```

**Option 2: Check Model Performance**
- If model can't achieve 70% recall on TRAINING data, model is too weak
- Consider: More epochs, better architecture, more data

**Option 3: Accept Unconstrained for Some Folds**
- Some subjects might be inherently hard
- Unconstrained G-mean is still better than random

---

## Testing

### Quick Test:
```bash
cd experiments/08_tcn
python train.py
```

### Watch for:
1. ✅ New log lines showing training performance
2. ✅ Training sensitivity ≥ min_recall
3. ✅ No warnings about unsatisfiable constraints (ideally)
4. ✅ Test recall improved from previous runs

---

## Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Parameter passing** | ❌ Ignored | ✅ Respected |
| **Constraint values** | Always 0.75/0.25 | User-specified |
| **Training visibility** | ❌ Not logged | ✅ Logged |
| **Debugging** | Hard to diagnose | Easy to verify |

**Bottom line:** This was a parameter shadowing bug. The function had the right logic, but wasn't using the right values. Now fixed!

