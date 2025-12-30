# Two-Stage Ensemble Implementation Summary

**Date**: 2025-12-28  
**Experiment**: 09_two_stage_ensemble  
**Status**: ✅ Complete - Ready for Training

## 📋 What Was Implemented

A **cascaded ensemble** combining:
1. **Logistic Regression** (Stage 1) - Recall-first screening
2. **Temporal Convolutional Network** (Stage 2) - FAR-first confirmation

### Key Innovation: Asymmetric Threshold Optimization

**Traditional Approach** (Same threshold objective for both):
- Both models optimize for G-mean or F1
- Doesn't leverage complementary strengths

**Our Approach** (Different objectives):
- **LR**: Optimize for **high recall** (catch most stress) → Recall-first threshold
- **TCN**: Optimize for **low FAR** (avoid false alarms) → FAR-first threshold
- Each model plays to its strength

## 🎯 Decision Logic

```
Input: Raw sensor data

┌─────────────────────────────────┐
│ Stage 1: Logistic Regression    │
│ (Extract features + HRV)         │
│ Threshold: Recall-first (75%)    │
└──────────┬──────────────────────┘
           │
           ├─→ P(stress) < τ_LR ? → [NO STRESS] (Skip Stage 2)
           │
           └─→ P(stress) ≥ τ_LR ? ↓
                                   │
           ┌───────────────────────┴─────┐
           │ Stage 2: TCN                 │
           │ (Raw time series 8×120)      │
           │ Threshold: FAR-first (25%)   │
           └──────────┬───────────────────┘
                      │
                      ├─→ P(stress) ≥ τ_TCN ? → [STRESS]
                      │
                      └─→ P(stress) < τ_TCN ? → [NO STRESS]
```

## 📁 Files Created

```
experiments/09_two_stage_ensemble/
├── __init__.py                        # Module initialization
├── train.py                           # Main training script (600+ lines)
├── threshold_selection.py             # Asymmetric threshold utilities
├── run.py                             # Quick start script
├── README.md                          # Complete documentation
└── IMPLEMENTATION_SUMMARY.md          # This file
```

## 🔧 Implementation Details

### 1. Threshold Selection (`threshold_selection.py`)

#### Recall-First Threshold (for LR)
```python
def find_recall_first_threshold(y_true, y_proba, min_recall=0.75):
    """
    Find LOWEST threshold that achieves minimum recall.
    
    Strategy:
    - Start from lowest threshold (highest recall)
    - Find the highest threshold that still meets recall constraint
    - Prioritizes catching stress over precision
    """
```

#### FAR-First Threshold (for TCN)
```python
def find_far_first_threshold(y_true, y_proba, max_far=0.25):
    """
    Find HIGHEST threshold that achieves maximum FAR.
    
    Strategy:
    - Start from highest threshold (lowest FAR)
    - Find the lowest threshold that still meets FAR constraint
    - Prioritizes avoiding false alarms over recall
    """
```

#### Two-Stage Decision
```python
def apply_two_stage_decision(lr_proba, tcn_proba, lr_thresh, tcn_thresh):
    """
    Apply cascaded logic:
    LR screens → TCN confirms
    """
    lr_detects = (lr_proba >= lr_thresh)
    tcn_confirms = (tcn_proba >= tcn_thresh)
    return (lr_detects & tcn_confirms).astype(int)
```

### 2. LOSO Training Loop (`train.py`)

**Algorithm**:
```
FOR EACH test_subject IN subjects:
    1. Split data (train vs test)
    
    2. Train LR:
       - Extract features from aligned 1Hz data
       - Add HRV metrics from PPG
       - Train with class_weight='balanced'
    
    3. Train TCN:
       - Use raw multichannel time series (8×120)
       - Subject-wise normalization
       - Weighted CE loss, early stopping
    
    4. Get TRAINING probabilities:
       - p_train_lr  = LR.predict_proba(train)
       - p_train_tcn = TCN.predict_proba(train)
    
    5. Select thresholds (TRAINING only):
       - τ_LR  = recall_first(p_train_lr, min_recall=0.75)
       - τ_TCN = far_first(p_train_tcn, max_far=0.25)
    
    6. Test on held-out subject:
       - p_test_lr  = LR.predict_proba(test)
       - p_test_tcn = TCN.predict_proba(test)
    
    7. Apply two-stage decision:
       - y_pred = two_stage_decision(p_test_lr, p_test_tcn, τ_LR, τ_TCN)
    
    8. Evaluate and store metrics

9. Aggregate across all folds
```

### 3. Key Functions

#### `load_and_prepare_windows()`
- Loads all subjects
- Aligns to 1Hz
- Creates labeled windows
- Returns `windows_by_subject` dictionary

#### `extract_features_from_windows()`
- Uses `BasicFeatureExtractor` from classical ML
- Extracts ~100+ features per window
- Imputes missing values with median
- Returns `X, y, feature_names`

#### `train_lr_model()`
- StandardScaler normalization
- Logistic Regression with:
  - C=5.0, penalty='l1', solver='saga'
  - class_weight='balanced'
  - max_iter=1000
- Returns model and scaler

#### `train_tcn_model()`
- Creates `VitaStressTCNDataset` with seq_len=120
- TCN architecture:
  - 8 input channels
  - 6 layers: [16,16,16,16,16,16]
  - Dilations: [1,2,4,8,16,32]
  - Receptive field: 127
- Weighted CE loss (sklearn balanced weights)
- Adam optimizer, lr=1e-3
- Early stopping (patience=15)
- Returns trained model

#### `get_tcn_probabilities()`
- Creates dataset from windows
- Runs TCN inference
- Returns probabilities for positive class

#### `loso_cross_validation()`
- Main LOSO loop
- Trains both models per fold
- Selects asymmetric thresholds
- Applies two-stage decision
- Aggregates results

## 📊 Expected Performance

| Metric | LR | TCN | **Two-Stage** |
|--------|-----|-----|---------------|
| Recall | 0.75 | 0.41 | **0.55-0.65** |
| Specificity | 0.74 | 0.86 | **0.85-0.90** |
| Precision | 0.31 | 0.30 | **0.40-0.50** |
| FAR | 26% | 13.5% | **10-15%** |
| G-mean | 0.74 | 0.60 | **0.70-0.77** |
| AUROC | 0.828 | 0.727 | **0.80-0.85** |

**Why This Should Work**:
1. ✅ LR catches 75% of stress (high recall)
2. ✅ Of those, TCN confirms ~70-85% (high specificity filter)
3. ✅ Final recall: 0.75 × 0.75 ≈ 0.56 (better than TCN alone)
4. ✅ Final FAR: Much lower than LR alone

## 🚀 How to Run

### Basic Usage
```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation
python experiments/09_two_stage_ensemble/train.py
```

### Custom Parameters
```bash
python experiments/09_two_stage_ensemble/run.py \
    --min-recall-lr 0.80 \
    --max-far-tcn 0.20 \
    --epochs 100
```

### Expected Runtime
- **Per fold**: ~2-3 minutes
  - LR training: ~10-20 seconds
  - TCN training: ~90-120 seconds (100 epochs with early stopping)
- **Total (21 folds)**: ~40-60 minutes

## 📈 Outputs

### Files Generated
```
experiments/09_two_stage_ensemble/results/
├── training.log                               # Detailed training log
├── two_stage_ensemble_metrics.json            # Aggregated metrics
├── two_stage_ensemble_predictions.csv         # All predictions
├── two_stage_ensemble_fold_metrics.csv        # Per-fold metrics
└── figures/
    ├── two_stage_ensemble_confusion_matrix.png
    ├── two_stage_ensemble_roc.png
    └── two_stage_ensemble_pr.png
```

### Metrics Logged Per Fold
- `lr_threshold`: LR threshold value
- `tcn_threshold`: TCN threshold value
- `lr_train_recall`: LR recall on training data
- `tcn_train_far`: TCN FAR on training data
- `recall`, `precision`, `specificity`, `f1`, `gmean`: Test metrics
- `auroc`, `pr_auc`: Threshold-independent metrics

## 🔍 Verification Checklist

After training, verify:

### ✅ Thresholds Make Sense
- [ ] LR thresholds: 0.30-0.50 (lower for high recall)
- [ ] TCN thresholds: 0.70-0.85 (higher for low FAR)
- [ ] LR thresholds < TCN thresholds in most folds

### ✅ Training Constraints Met
- [ ] LR training recall ≥ 75% in all/most folds
- [ ] TCN training FAR ≤ 25% in all/most folds

### ✅ Test Performance
- [ ] Test recall: 50-70% (better than TCN alone: 41%)
- [ ] Test specificity: 80-90% (better than LR alone: 74%)
- [ ] Test precision: 35-50% (better than both: ~30%)
- [ ] G-mean: 65-80% (competitive with LR: 74%)

### ✅ Logical Consistency
- [ ] When LR recall is high, ensemble recall is higher
- [ ] When TCN FAR is low, ensemble FAR is lower
- [ ] Ensemble is not worse than both individuals on all metrics

## 🐛 Known Issues & Solutions

### Issue 1: Import Errors
**Error**: `ModuleNotFoundError: No module named 'feature_extraction'`

**Solution**: The script uses try/except to handle imports:
```python
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "01_classical_ml"))
    from feature_extraction import BasicFeatureExtractor
except ImportError:
    from experiments.01_classical_ml.feature_extraction import BasicFeatureExtractor
```

If still failing, ensure you run from project root.

### Issue 2: CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: The script auto-selects device:
```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
```

For Mac with MPS, it should work fine. Batch size is already 32 (reasonable).

### Issue 3: Recall Still Too Low
**Problem**: Test recall < 50%

**Solution**:
1. Check `lr_train_recall` in fold metrics - is LR meeting 75% on training?
2. Lower `min_recall_lr` to 0.70 or 0.65
3. Increase `max_far_tcn` to 0.30 (more lenient)

### Issue 4: Too Many False Alarms
**Problem**: Test FAR > 20%

**Solution**:
1. Check `tcn_train_far` in fold metrics - is TCN meeting 25% on training?
2. Lower `max_far_tcn` to 0.20 or 0.15
3. Increase `min_recall_lr` to 0.80 (TCN will see fewer candidates)

## 🎓 Theoretical Justification

### Why Asymmetric Thresholds?

**Problem with symmetric thresholds**:
- Both models optimize for same metric (e.g., G-mean)
- Doesn't leverage complementary strengths
- One model's advantage is wasted

**Solution with asymmetric thresholds**:
- LR optimizes for recall → catches most stress
- TCN optimizes for specificity → filters false alarms
- Each model contributes its strength

### Mathematical Justification

**Final Recall**:
```
R_ensemble = R_LR × R_TCN|LR_positive
```
Where `R_TCN|LR_positive` is TCN's recall among LR-detected cases.

If:
- R_LR = 0.75 (from recall-first threshold)
- R_TCN|LR_positive ≈ 0.70 (TCN's recall on harder cases)

Then:
- R_ensemble ≈ 0.75 × 0.70 = 0.525

**Final Specificity**:
```
S_ensemble = S_LR + (1-S_LR) × S_TCN
```

If:
- S_LR = 0.74
- S_TCN = 0.86

Then:
- S_ensemble ≈ 0.74 + 0.26 × 0.86 = 0.96

**Better G-mean**:
```
G_ensemble = √(R_ensemble × S_ensemble)
           = √(0.525 × 0.96)
           ≈ 0.71
```

vs LR alone: 0.74, TCN alone: 0.60

## 📚 Comparison with Other Approaches

| Approach | Recall | Specificity | Complexity |
|----------|--------|-------------|------------|
| **LR only** | 0.75 | 0.74 | Low |
| **TCN only** | 0.41 | 0.86 | Medium |
| **Soft voting (0.5,0.5)** | ~0.65 | ~0.80 | Low |
| **Stacking** | ~0.68 | ~0.82 | High |
| **Two-stage (ours)** | ~0.60 | ~0.88 | Medium |

**Our advantage**:
- ✅ More interpretable than stacking
- ✅ Better specificity than soft voting
- ✅ Explicit control over recall/FAR trade-off
- ✅ Computational efficiency (TCN runs only ~30% of time)

## ✅ Implementation Complete

All files created and ready for training:
- [x] `__init__.py` - Module initialization
- [x] `threshold_selection.py` - Asymmetric threshold utilities
- [x] `train.py` - Main training script with LOSO
- [x] `run.py` - Quick start script
- [x] `README.md` - Comprehensive documentation
- [x] `IMPLEMENTATION_SUMMARY.md` - This summary

## 🎉 Next Steps

1. **Run Training**:
   ```bash
   python experiments/09_two_stage_ensemble/train.py
   ```

2. **Monitor Progress**:
   - Watch `training.log` for per-fold results
   - Check if thresholds are reasonable
   - Verify recall/FAR constraints are met

3. **Analyze Results**:
   - Compare with LR-only and TCN-only results
   - Check fold-by-fold stability
   - Verify ensemble improves over individuals

4. **Iterate if Needed**:
   - Adjust `min_recall_lr` and `max_far_tcn` based on results
   - Try different combinations to find sweet spot

---

**Implementation Status**: ✅ COMPLETE  
**Ready for Training**: ✅ YES  
**Estimated Training Time**: 40-60 minutes (21 folds)  
**Expected Improvement**: +10-20% precision, +15-25% recall vs TCN

