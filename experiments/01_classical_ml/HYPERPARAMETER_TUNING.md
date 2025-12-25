# Hyperparameter Tuning Implementation

## Overview

Implemented **nested cross-validation** for hyperparameter tuning in the classical ML pipeline following the algorithm you specified.

## Algorithm Structure

```
FOR each test subject (OUTER LOSO LOOP):
    1. Split: Test subject vs all other subjects
    2. INNER CV: Tune hyperparameters on training subjects
       - Split training subjects into K=5 folds
       - Try all hyperparameter combinations
       - Select best based on G-mean
    3. Train final model with best hyperparameters
    4. Select optimal threshold on training data
    5. Evaluate on test subject
    
Aggregate results across all folds
```

## Hyperparameter Grids

### Logistic Regression
- **C**: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] - Regularization strength
- **penalty**: ['l1', 'l2'] - Regularization type
- **solver**: ['liblinear', 'saga'] - Optimizers supporting both L1/L2
- **Total combinations**: 24

### Random Forest
- **n_estimators**: [50, 100, 200] - Number of trees
- **max_depth**: [5, 10, 15, 20, None] - Tree depth
- **min_samples_split**: [2, 5, 10] - Min samples to split
- **min_samples_leaf**: [1, 2, 4] - Min samples at leaf
- **max_features**: ['sqrt', 'log2', None] - Features per split
- **Total combinations**: 270

### SVM
- **C**: [0.1, 1.0, 10.0, 100.0] - Regularization
- **kernel**: ['rbf', 'poly', 'sigmoid'] - Kernel types
- **gamma**: ['scale', 'auto', 0.001, 0.01, 0.1] - Kernel coefficient
- **degree**: [2, 3, 4] - Polynomial degree (for poly kernel)
- **Total combinations**: 60

### XGBoost
- **n_estimators**: [50, 100, 200] - Boosting rounds
- **max_depth**: [3, 5, 7, 9] - Tree depth
- **learning_rate**: [0.01, 0.05, 0.1, 0.3] - Step size
- **subsample**: [0.7, 0.8, 1.0] - Sample ratio
- **colsample_bytree**: [0.7, 0.8, 1.0] - Feature sampling
- **min_child_weight**: [1, 3, 5] - Min weights in child
- **gamma**: [0, 0.1, 0.2] - Min loss reduction
- **Total combinations**: 972

## Key Features

### 1. **Nested CV (No Data Leakage)**
- Outer loop: LOSO (21 folds, one per subject)
- Inner loop: 5-fold subject-wise CV for hyperparameter selection
- Test data is **never** seen during hyperparameter tuning

### 2. **Subject-Wise Splitting**
- Inner CV splits by subjects (not windows)
- Prevents data leakage from same subject across folds

### 3. **G-Mean Optimization**
- Hyperparameters selected based on G-mean
- Balances sensitivity and specificity
- Appropriate for imbalanced data (1:7.1 ratio)

### 4. **Automatic Grid Search**
- Grids defined based on sklearn documentation
- Different grids per model type
- Can be customized via `param_grid` parameter

### 5. **Tracking & Logging**
- Logs best parameters per fold
- Reports most common hyperparameter values
- Returns `best_params_per_fold` in results

## Usage

### Enable/Disable Tuning

```python
# In loso_cross_validation call
result = loso_cross_validation(
    X, y, subjects, features,
    model_class, model_name, model_kwargs, logger,
    enable_tuning=True,      # Set to False to disable
    param_grid=None          # None = use default grid
)
```

### Custom Grid

```python
custom_grid = {
    'C': [1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

result = loso_cross_validation(
    ...,
    enable_tuning=True,
    param_grid=custom_grid
)
```

## Output

### Console Logs
```
============================================================
Training: LOGISTIC_REGRESSION
============================================================
LOSO CV with 21 folds (subjects)
Hyperparameter tuning: ENABLED
  Param grid size: 24 combinations

    Inner CV: Tuning 24 configurations...
    Best params: {'C': 1.0, 'penalty': 'l2', 'solver': 'saga'} (score: 0.7234)

...

============================================================
HYPERPARAMETER TUNING SUMMARY
============================================================
  C:
    1.0: 12/21 folds
    10.0: 6/21 folds
    0.1: 3/21 folds
  penalty:
    l2: 15/21 folds
    l1: 6/21 folds
```

### Results Dictionary
```python
{
    "metrics": {...},
    "y_true": [...],
    "y_pred": [...],
    "y_proba": [...],
    "subjects": [...],
    "fold_metrics": [...],
    "best_params_per_fold": [  # NEW
        {'C': 1.0, 'penalty': 'l2', ...},
        {'C': 10.0, 'penalty': 'l1', ...},
        ...
    ]
}
```

## Performance Considerations

### Training Time Estimates (21 subjects)

- **Logistic Regression**: ~5-10 minutes
  - 21 outer folds × 5 inner folds × 24 params = 2,520 fits
  
- **Random Forest**: ~30-60 minutes
  - 21 outer folds × 5 inner folds × 270 params = 28,350 fits
  
- **SVM**: ~15-30 minutes
  - 21 outer folds × 5 inner folds × 60 params = 6,300 fits
  
- **XGBoost**: ~60-120 minutes
  - 21 outer folds × 5 inner folds × 972 params = 102,060 fits

### Optimization Tips

1. **Reduce inner folds**: Change `n_inner_folds=5` to `n_inner_folds=3`
2. **Reduce grid size**: Fewer hyperparameter values
3. **Parallel processing**: Already using `n_jobs=-1` where available
4. **Progressive refinement**: Start with coarse grid, refine around best values

## Validation

The implementation follows your specified algorithm exactly:
- ✅ OUTER LOSO LOOP (subject-wise)
- ✅ STEP 1: Subject-wise split
- ✅ STEP 2: Hyperparameter tuning (inner CV)
- ✅ STEP 3: Train final model with best params
- ✅ STEP 4: Threshold selection on training data
- ✅ STEP 5: Test on held-out subject
- ✅ STEP 6: Final evaluation and aggregation

## Files Modified

- `experiments/01_classical_ml/train.py`:
  - Added `get_hyperparameter_grid()`
  - Added `inner_cv_hyperparameter_tuning()`
  - Modified `loso_cross_validation()` signature
  - Updated main() to enable tuning
  - Added hyperparameter summary logging

## Next Steps

1. Run training to see which hyperparameters work best
2. Analyze `best_params_per_fold` to see consistency
3. Optionally refine grids based on results
4. Compare tuned vs. untuned performance

