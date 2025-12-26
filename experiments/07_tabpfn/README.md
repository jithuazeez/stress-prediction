# TabPFN Foundation Model for Stress Prediction

This experiment uses **TabPFN** (Tabular Prior-Data Fitted Network), a pre-trained foundation model for tabular classification, to predict emotional stress from wearable sensor data.

## What is TabPFN?

TabPFN is a **foundation model** specifically designed for small-to-medium tabular datasets:
- **Pre-trained** on synthetic tabular data
- **No hyperparameter tuning** needed
- **Fast inference** (seconds, not minutes)
- **Handles missing values** automatically
- **Works well on small datasets** (<10K samples, <100 features)

Unlike classical ML models (Logistic Regression, Random Forest, SVM) that require training from scratch, TabPFN leverages transfer learning from its pre-training.

**Paper**: [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)

## 🔧 Setup Instructions

### 1. Install TabPFN

```bash
pip install tabpfn
```

### 2. HuggingFace Authentication

TabPFN models are hosted on HuggingFace and may require authentication:

**Option A: Interactive Login**
```bash
huggingface-cli login
```

**Option B: Environment Variable (Recommended for Scripts)**
```bash
export HF_TOKEN=<your_huggingface_read_token>
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Accept Model License

Visit: https://huggingface.co/Prior-Labs/tabpfn_2_5

Click "Agree and access repository" (required for first-time use)

### 4. Run Training

```bash
cd experiments/07_tabpfn
python train.py
```

## 📊 Methodology

### Pipeline (Same as Classical ML)

1. **Data Loading**: Load raw sensor data from all subjects
2. **Alignment**: Align to 1Hz sampling rate
3. **Windowing**: Create 120s sliding windows with 30s stride
4. **Feature Extraction**: 
   - Accelerometer features (41)
   - Temperature features (7)
   - Heat flux features (9)
   - HR/HRV features from raw PPG at 64Hz (8)
   - **Total: ~65 features**
5. **Quality Filtering**: 
   - Stage 1: Keep only windows with valid `hr_bpm`
   - Stage 2: Impute remaining missing values with median
6. **LOSO Cross-Validation**: Leave-One-Subject-Out for robust evaluation
7. **Threshold Selection**: Optimize on training data (geometric mean of sensitivity/specificity)
8. **Evaluation**: Comprehensive metrics on test subject

### Key Differences from Classical ML

| Aspect | Classical ML | TabPFN |
|--------|-------------|--------|
| **Hyperparameter Tuning** | Nested CV (extensive) | None (pre-trained) |
| **Scaling** | Manual StandardScaler | Automatic |
| **Class Weights** | Manual (balanced) | Automatic |
| **Training Time** | Minutes | Seconds |
| **Model Complexity** | Simple (LR) to Complex (XGB) | Black-box Transformer |

### Evaluation Metrics (Same as Classical ML)

**Clinical Metrics (Primary):**
- **G-Mean**: Geometric mean of sensitivity and specificity (balanced metric)
- **Sensitivity (Recall)**: Stress detection rate
- **Specificity**: False alarm control
- **Precision**: Positive predictive value

**Standard Metrics:**
- AUROC, PR-AUC, F1-score, Accuracy

**Per-Fold Analysis:**
- Subject-wise metrics saved to `tabpfn_fold_metrics.csv`

## 📁 Output Files

All results saved to `experiments/07_tabpfn/results/`:

- `features_dataset.csv`: Full feature dataset (all windows)
- `hrv_features_dataset.csv`: HRV-only features with metadata
- `tabpfn_metrics.json`: Overall performance metrics
- `tabpfn_fold_metrics.csv`: Per-subject (fold) metrics
- `tabpfn_predictions.csv`: Predictions (subject_id, y_true, y_pred, y_proba)
- `tabpfn_roc.png`: ROC curve
- `tabpfn_pr.png`: Precision-Recall curve
- `tabpfn_confusion_matrix.png`: Confusion matrix
- `training.log`: Full training log

## 🎯 Expected Performance

Based on the classical ML baseline (with same features):

| Model | AUROC | PR-AUC | F1 | Recall | Precision |
|-------|-------|--------|----|----|-----------|
| Logistic Regression | ~0.75 | ~0.40 | ~0.35 | ~0.45 | ~0.30 |
| Random Forest | ~0.78 | ~0.42 | ~0.38 | ~0.48 | ~0.32 |
| **TabPFN** (expected) | **~0.76-0.80** | **~0.41-0.45** | **~0.36-0.40** | **~0.46-0.52** | **~0.31-0.35** |

TabPFN should perform **comparably or better** than classical ML, especially given:
- Small dataset (~1,400 windows, 21 subjects)
- High class imbalance (1:7.1 ratio)
- Complex feature interactions

## 🚀 Advantages of TabPFN

### 1. **No Hyperparameter Tuning**
Classical ML required extensive nested CV to find optimal hyperparameters (C, max_depth, gamma, etc.). TabPFN is **pre-trained** and ready to use.

### 2. **Fast Training**
- Classical ML: ~2-5 minutes per model (with tuning)
- TabPFN: ~10-30 seconds (no tuning needed)

### 3. **Automatic Preprocessing**
- No manual scaling (StandardScaler)
- No class weight tuning
- Handles missing values internally

### 4. **Transfer Learning**
TabPFN was pre-trained on millions of synthetic tabular datasets, so it has "seen" many classification patterns before.

## ⚠️ Limitations

### 1. **Dataset Size Constraints**
- Works best with **<10,000 training samples**
- Performance degrades on very large datasets
- Our dataset (~1,400 windows) is ideal

### 2. **Feature Limit**
- Works best with **<100 features**
- We have ~65 features (within limit)

### 3. **Black Box**
- Cannot inspect feature importance like LR coefficients
- No attention mechanisms exposed (yet)

### 4. **GPU Recommended**
- Runs on CPU but **10x faster on GPU**
- MPS (Apple Silicon) supported

## 🔍 Troubleshooting

### Error: "Model not accessible"
```
Solution:
1. Run: huggingface-cli login
2. Accept license at: https://huggingface.co/Prior-Labs/tabpfn_2_5
3. Ensure internet connection for first download (~500MB)
```

### Error: "ImportError: No module named 'tabpfn'"
```
Solution: pip install tabpfn
```

### Warning: "Running on CPU - this will be slower!"
```
This is OK but slow. To use GPU:
- NVIDIA: Install CUDA + PyTorch with CUDA
- Apple Silicon: PyTorch 2.0+ has MPS support automatically
```

### TabPFN is very slow
```
Possible causes:
1. Running on CPU (use GPU)
2. Training set too large (>50K samples)
3. Too many features (>100 features)

Our dataset is small (~1.4K samples, 65 features) so should be fast.
```

## 📚 References

1. **TabPFN Paper**: Hollmann et al. (2022). "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second". NeurIPS 2022.
   - https://arxiv.org/abs/2207.01848

2. **Official Repository**: 
   - https://github.com/PriorLabs/TabPFN

3. **Model Card**:
   - https://huggingface.co/Prior-Labs/tabpfn_2_5

## 🆚 Comparison with Classical ML

Run both experiments to compare:

```bash
# Classical ML (Logistic Regression, RF, SVM, XGBoost)
cd experiments/01_classical_ml
python train.py

# TabPFN (Foundation Model)
cd ../07_tabpfn
python train.py
```

Results will be automatically compared if classical ML results exist.

## 💡 Tips

1. **First run is slow** (~30s) due to model download (~500MB). Subsequent runs are fast.
2. **Use GPU if available** for 10x speedup
3. **TabPFN is deterministic** (unlike RF/XGBoost with random seeds)
4. **No need to tune anything** - just run and evaluate
5. **Check logs carefully** - HF authentication issues are common first time

## ✅ Success Checklist

- [ ] TabPFN installed (`pip install tabpfn`)
- [ ] HuggingFace authentication setup (login or HF_TOKEN)
- [ ] Model license accepted on HuggingFace
- [ ] First model download complete (~500MB)
- [ ] Training runs without errors
- [ ] Results saved to `results/` directory
- [ ] Metrics comparable to classical ML baseline

---

**Questions?** Check the training log in `results/training.log` for detailed diagnostics.
