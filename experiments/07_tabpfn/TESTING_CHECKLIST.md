# Testing Checklist for TabPFN Experiment

## Pre-Testing Requirements

- [ ] Python 3.9+ installed
- [ ] Classical ML experiment (01_classical_ml) completed (optional but recommended)
- [ ] HuggingFace account created
- [ ] Internet connection available (for first-time model download)

## Installation Testing

### Step 1: Install TabPFN
```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments
pip install tabpfn
```

**Expected output:**
```
Successfully installed tabpfn-6.x.x
```

- [ ] TabPFN installed without errors
- [ ] Version 6.0.0 or higher

### Step 2: HuggingFace Authentication
```bash
huggingface-cli login
```

**Expected:**
- [ ] Login prompt appears
- [ ] Token accepted
- [ ] Credentials saved

### Step 3: Accept Model License
Visit: https://huggingface.co/Prior-Labs/tabpfn_2_5

- [ ] License page loads
- [ ] "Agree and access repository" button clicked
- [ ] Access granted

## Import Testing

Test if TabPFN can be imported:

```bash
cd 07_tabpfn
python -c "from tabpfn import TabPFNClassifier; print('✓ TabPFN imported successfully')"
```

**Expected output:**
```
✓ TabPFN imported successfully
```

- [ ] Import successful
- [ ] No error messages

## Feature Extraction Testing

Test if feature extraction works:

```bash
cd ..
python -c "from experiments.classical_ml.feature_extraction import BasicFeatureExtractor; print('✓ Feature extractor imported')"
```

- [ ] Feature extractor imported
- [ ] No import errors

## Dry Run Test

Run with a single subject to test the pipeline:

```bash
cd 07_tabpfn
python train.py 2>&1 | head -100
```

**Check for:**
- [ ] "TabPFN model accessible" message
- [ ] Device detected (cuda or cpu)
- [ ] "Found 21 subject folders" (or your number of subjects)
- [ ] Loading subjects progress bar starts
- [ ] No immediate errors

## Full Execution Test

Run the complete experiment:

```bash
python train.py
```

**Monitor for:**

### Phase 1: Data Loading (Expected: ~30-60 seconds)
- [ ] Progress bar shows "Loading subjects"
- [ ] No "Failed to load" errors
- [ ] "Loaded: 21/21 subjects" (or your number)
- [ ] HRV extraction statistics shown
- [ ] Missing data analysis shown
- [ ] Class distribution shown

### Phase 2: Training (Expected: ~5-20 minutes)
- [ ] Progress bar shows "TabPFN"
- [ ] Fold metrics update (Gmean, Recall, Test count)
- [ ] No "Training failed" warnings
- [ ] All 21 folds complete
- [ ] "Training completed in X.Xs" message

### Phase 3: Results Saving
- [ ] Metrics saved to `results/tabpfn_metrics.json`
- [ ] Predictions saved to `results/tabpfn_predictions.csv`
- [ ] Fold metrics saved to `results/tabpfn_fold_metrics.csv`
- [ ] Plots generated in `results/figures/`

## Output Validation

### Check Files Created

```bash
ls -lh results/
```

**Expected files:**
- [ ] `training.log` - Detailed log file
- [ ] `tabpfn_metrics.json` - Overall metrics
- [ ] `tabpfn_fold_metrics.csv` - Per-fold results
- [ ] `tabpfn_predictions.csv` - All predictions
- [ ] `features_dataset.csv` - Extracted features
- [ ] `missing_analysis.json` - Data quality
- [ ] `class_distribution.json` - Class balance
- [ ] `figures/tabpfn_roc.png` - ROC curve
- [ ] `figures/tabpfn_pr.png` - PR curve
- [ ] `figures/tabpfn_confusion_matrix.png` - Confusion matrix

### Check Metrics

```bash
python -c "import json; metrics = json.load(open('results/tabpfn_metrics.json')); print(f\"AUROC: {metrics['auroc']:.4f}\")"
```

**Expected:**
- [ ] AUROC between 0.50-1.00 (hopefully >0.70)
- [ ] PR-AUC between 0.50-1.00
- [ ] F1 score between 0.50-1.00

### Check Predictions

```bash
wc -l results/tabpfn_predictions.csv
```

**Expected:**
- [ ] Number of rows matches your dataset size (~1400-1800 windows)
- [ ] Header row present

### Check Fold Metrics

```bash
wc -l results/tabpfn_fold_metrics.csv
```

**Expected:**
- [ ] 22 lines (21 folds + 1 header)
- [ ] Each fold has metrics

## Comparison Testing

Test integration with comparison script:

```bash
cd ..
python compare_all.py
```

**Check for:**
- [ ] TabPFN results loaded
- [ ] "✓ TabPFN: Loaded" message
- [ ] Comparison table includes TabPFN
- [ ] Bar chart shows TabPFN
- [ ] Radar chart shows TabPFN

## Performance Validation

### Expected Metrics Range

Based on VitaStress dataset characteristics:

- [ ] **AUROC**: 0.70-0.90 (competitive with XGBoost)
- [ ] **PR-AUC**: 0.55-0.80
- [ ] **F1**: 0.60-0.80
- [ ] **Gmean**: 0.65-0.85
- [ ] **Recall**: 0.60-0.90
- [ ] **Precision**: 0.60-0.80

### Speed Validation

- [ ] **With GPU**: ~10-20 seconds per fold
- [ ] **With CPU**: ~20-60 seconds per fold
- [ ] **Total time**: 5-20 minutes

## Error Handling Tests

### Test 1: Missing TabPFN
```bash
# Temporarily rename tabpfn
pip uninstall tabpfn -y
python train.py 2>&1 | head -20
```

**Expected:**
- [ ] Clear error message: "TabPFN not installed!"
- [ ] Installation instructions shown
- [ ] Script exits gracefully

**Cleanup:**
```bash
pip install tabpfn
```

### Test 2: No HuggingFace Auth
```bash
# Logout (if applicable)
# Run without authentication
```

**Expected:**
- [ ] Clear error about authentication
- [ ] Instructions for login shown
- [ ] Script exits gracefully

### Test 3: Interrupted Execution
```bash
# Start training and press Ctrl+C after 2-3 folds
python train.py
# Ctrl+C
```

**Expected:**
- [ ] Partial results saved
- [ ] Can resume or restart
- [ ] No corruption of files

## Log File Validation

Check the training log for completeness:

```bash
grep "ERROR" results/training.log
grep "WARNING" results/training.log
tail -50 results/training.log
```

**Verify:**
- [ ] No unexpected ERRORs
- [ ] WARNINGs are expected (e.g., HeartPy availability)
- [ ] Log ends with "EXPERIMENT END"
- [ ] Final summary shown

## Comparison with Classical ML

If you have classical ML results:

```bash
# Compare AUROC
python -c "
import json
tabpfn = json.load(open('07_tabpfn/results/tabpfn_metrics.json'))
xgb = json.load(open('01_classical_ml/results/xgboost_metrics.json'))
print(f'TabPFN AUROC: {tabpfn[\"auroc\"]:.4f}')
print(f'XGBoost AUROC: {xgb[\"auroc\"]:.4f}')
print(f'Difference: {(tabpfn[\"auroc\"] - xgb[\"auroc\"]):.4f}')
"
```

**Expected:**
- [ ] TabPFN within ±0.05 of XGBoost
- [ ] TabPFN potentially better due to pre-training

## Documentation Validation

- [ ] README.md is clear and comprehensive
- [ ] QUICKSTART.md has correct paths
- [ ] IMPLEMENTATION_SUMMARY.md matches actual implementation
- [ ] All referenced files exist

## Final Checklist

- [ ] All tests passed
- [ ] Results are reasonable
- [ ] Logs are complete
- [ ] Comparison works
- [ ] Documentation is accurate
- [ ] No unexpected errors

## Troubleshooting Notes

**If tests fail, check:**

1. Python version (must be 3.9+)
2. HuggingFace authentication
3. Internet connection (first run)
4. Disk space (~500MB for model)
5. Data path in `shared/config.py`
6. Log files for detailed errors

**Common issues:**

- **"Access denied"**: Accept license at HuggingFace
- **"CUDA out of memory"**: Use CPU mode (automatic fallback)
- **"Import error"**: Check Python path and package installation
- **"No data loaded"**: Check data path in config

## Success Criteria

✅ All checkboxes marked
✅ AUROC > 0.70
✅ All output files generated
✅ Comparison script includes TabPFN
✅ No unexpected errors in logs

---

**Test Date**: __________
**Tested By**: __________
**Result**: ☐ Pass  ☐ Fail
**Notes**: ___________________________________________

