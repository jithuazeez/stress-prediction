# Evaluation Protocol

## Overview

This document details the evaluation protocol for emotional stress prediction models in the VitaStress dataset. Our evaluation strategy ensures **subject-independent assessment** with **rigorous prevention of data leakage** through Leave-One-Subject-Out (LOSO) cross-validation and nested LOSO for meta-learning. This chapter explains:

1. Subject-independent LOSO cross-validation setup
2. Nested LOSO for stacked meta-learning
3. Data leakage prevention mechanisms
4. Comprehensive metrics reported
5. Rationale for design choices

---

## 1. Subject-Independent Evaluation: LOSO Cross-Validation

### 1.1 Why Subject-Independent?

**Problem with subject-dependent evaluation**:
- Models trained and tested on the same subjects (e.g., random train-test split)
- Can memorize subject-specific patterns (baseline physiology, sensor positioning, activity patterns)
- **Overestimates real-world performance**: New users will have unseen physiological signatures

**Our approach: Leave-One-Subject-Out (LOSO)**:
- Each subject becomes the test set exactly once
- Model never sees test subject's data during training
- **Simulates deployment**: New user with no historical data

This is the **gold standard** for wearable stress prediction evaluation.

---

### 1.2 LOSO Cross-Validation Setup

#### LOSO Procedure

For a dataset with **N = 21 subjects**:

1. **N folds** (one per subject)
2. **For fold k = 1 to N**:
   - **Test set**: Subject k (held out completely)
   - **Training set**: All other N-1 subjects
   - Train model on training set
   - Evaluate on test set
3. **Aggregate** results across all N folds

**Key property**: Each window in the dataset is tested exactly **once**, ensuring every prediction is subject-independent.

---

#### Implementation (Classical ML)

```python
# From train_classical_ml.py lines 221-243
def loso_with_ablation_b(X, y, subjects, model_class, model_name, model_params):
    """
    LOSO CV with Ablation B: 3 threshold strategies per fold.
    """
    unique_subjects = np.unique(subjects)  # N = 21 subjects
    n_subjects = len(unique_subjects)
    
    for test_subject in unique_subjects:
        # 1. Split: train vs test (SUBJECT-BASED)
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 2. Per-fold imputation (fit on train only)
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)  # Fit on train
        X_test = imputer.transform(X_test)        # Apply to test
        
        # 3. Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # 4. Select thresholds on TRAIN data
        y_train_proba = model.predict_proba(X_train)[:, 1]
        optimal_threshold = find_optimal_threshold(y_train, y_train_proba)
        
        # 5. Apply threshold to TEST data
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # 6. Evaluate
        metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
```

**Critical steps**:
- **Line 1**: Split by `subject_id`, not random indices
- **Line 2**: Imputation fitted only on training subjects
- **Line 4**: Threshold selected on **training probabilities** (not test)
- **Line 5**: Threshold applied to test probabilities

---

#### Implementation (TCN)

```python
# From train_tcn.py lines 182-248
def loso_with_ablation_b(windows_by_subject, config, device):
    """
    LOSO CV for TCN.
    """
    subjects = list(windows_by_subject.keys())  # N = 21 subjects
    
    for test_subject in subjects:
        # 1. Split windows by subject
        train_windows = []
        test_windows = []
        
        for subject_id, windows in windows_by_subject.items():
            if subject_id == test_subject:
                test_windows.extend(windows)
            else:
                train_windows.extend(windows)
        
        # 2. Create datasets with per-subject normalization
        train_dataset = VitaStressTCNDataset(
            train_windows,
            seq_len=480,
            normalize=True,
            normalization_mode="subject"  # Per-subject z-score
        )
        test_dataset = VitaStressTCNDataset(
            test_windows,
            seq_len=480,
            normalize=True,
            normalization_mode="subject"
        )
        
        # 3. Train model
        model = create_tcn_model(...)
        criterion = FocalLoss(alpha=0.88, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 4. Select thresholds on TRAIN data
        y_train, y_train_proba = evaluate_model(model, train_loader, device)
        optimal_threshold = find_optimal_threshold(y_train, y_train_proba)
        
        # 5. Apply threshold to TEST data
        y_test, y_test_proba = evaluate_model(model, test_loader, device)
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # 6. Evaluate
        metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
```

**Critical steps**:
- **Line 1**: Split entire window collection by subject
- **Line 2**: Per-subject normalization computed within each fold
- **Line 4**: Threshold selected on training set predictions
- **Line 5**: Model applied to held-out test subject

---

### 1.3 Per-Fold Statistics

**Number of folds**: 21 (one per subject)

**Typical fold sizes** (approximate):
- **Training set**: ~1,650 windows (20 subjects × ~83 windows/subject)
- **Test set**: ~83 windows (1 subject)
- **Class distribution per fold**: Varies (10-30% stress windows depending on subject)

**Imbalance variation across folds**:
- Each subject has different stress event frequency
- LOSO ensures model generalizes across varying imbalance ratios
- More challenging than fixed train-test split with uniform imbalance

---

## 2. Nested LOSO for Meta-Learning (Stacked Generalization)

### 2.1 The Data Leakage Problem in Meta-Learning

**Naive approach (WRONG)**:
1. Train base models (LR, TCN) using LOSO
2. Collect all base model predictions
3. Train meta-model on all predictions
4. Evaluate meta-model on same data

**Problem**: Meta-model sees test subject predictions during training → **data leakage!**

---

### 2.2 Nested LOSO Solution

We use **nested LOSO** to ensure meta-model never sees test subject's data:

```
For each test subject k:
    1. Hold out subject k completely (test set)
    2. Train meta-model on base predictions from OTHER N-1 subjects (train set)
    3. Apply meta-model to subject k's base predictions
    4. Evaluate on subject k
```

**Key property**: Meta-model trained on subject k = 1..20, tested on subject 21 → subject-independent at **both levels**.

---

### 2.3 Implementation (Stacked Generalization)

```python
# From train_fusion.py lines 82-136
def train_stacked_meta_model(lr_proba, tcn_proba, y_true, subjects):
    """
    Train stacked ensemble using nested LOSO.
    
    For each test subject:
    1. Hold out test subject
    2. Train meta-LR on out-of-fold base predictions
    3. Predict on test subject
    4. Select 3 thresholds on train, apply to test
    """
    unique_subjects = np.unique(subjects)  # N = 21
    
    for test_subject in unique_subjects:
        # 1. Split by subject (NESTED LOSO)
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        
        # 2. Training data for meta-model
        # Use base model probabilities from OTHER subjects
        X_train = np.column_stack([
            lr_proba[train_mask],   # LR probabilities (train subjects)
            tcn_proba[train_mask]   # TCN probabilities (train subjects)
        ])
        y_train = y_true[train_mask]
        
        # 3. Test data
        X_test = np.column_stack([
            lr_proba[test_mask],    # LR probabilities (test subject)
            tcn_proba[test_mask]    # TCN probabilities (test subject)
        ])
        y_test = y_true[test_mask]
        
        # 4. Train meta-model (Logistic Regression)
        meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_model.fit(X_train, y_train)  # Fit on train subjects only
        
        # 5. Select thresholds on TRAIN data
        y_train_proba = meta_model.predict_proba(X_train)[:, 1]
        optimal_threshold = find_optimal_threshold(y_train, y_train_proba)
        
        # 6. Apply to TEST data
        y_test_proba = meta_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # 7. Evaluate
        metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
```

**Why this prevents leakage**:
- **Line 4**: Meta-model fitted on train subjects only (never sees test subject)
- **Line 5**: Threshold selected on train subjects (not test)
- **Line 6**: Meta-model predicts on test subject (subject-independent)

---

### 2.4 Base Model Predictions for Meta-Learning

**Where do base model predictions come from?**

Base models (LR, TCN) are trained using **standard LOSO** (Section 1.2):
- For each subject k, base models trained on other N-1 subjects
- Predictions saved to CSV files:
  - `results/lr/lr_b1_predictions.csv`: LR probabilities per window
  - `results/tcn/tcn_b1_predictions.csv`: TCN probabilities per window

**Critical property**: These base predictions are **already subject-independent** because they were generated using LOSO.

**Meta-learning input**:
```python
# Load base predictions
lr_pred = pd.read_csv("results/lr/lr_b1_predictions.csv")
tcn_pred = pd.read_csv("results/tcn/tcn_b1_predictions.csv")

# Each row is a window with:
# - subject_id: Which subject this window belongs to
# - window_start, window_end: Window timestamps
# - y_true: Ground truth label
# - y_proba: Base model's stress probability (from LOSO)

# Stack probabilities as meta-features
X_meta = np.column_stack([lr_pred["y_proba"], tcn_pred["y_proba"]])
y_meta = lr_pred["y_true"]
subjects_meta = lr_pred["subject_id"]

# Train meta-model using nested LOSO
train_stacked_meta_model(X_meta[:, 0], X_meta[:, 1], y_meta, subjects_meta)
```

**Result**: Meta-model predictions are subject-independent at **two levels**:
1. Base predictions are subject-independent (from LOSO)
2. Meta-model is subject-independent (from nested LOSO)

---

## 3. Data Leakage Prevention Mechanisms

### 3.1 What is Data Leakage?

**Data leakage** occurs when information from the test set influences model training, leading to **overly optimistic** performance estimates that don't generalize to new data.

**Common sources of leakage in wearable stress prediction**:
1. **Subject leakage**: Test subject's windows in training set
2. **Temporal leakage**: Future labels influence past predictions
3. **Preprocessing leakage**: Statistics computed on test data
4. **Threshold leakage**: Threshold optimized on test labels
5. **Meta-learning leakage**: Meta-model trained on test predictions

**Our approach prevents all five sources.**

---

### 3.2 Mechanism 1: Subject-Based Splitting

**Problem**: Random window-level split allows test subject's windows in training set.

**Solution**: Split by `subject_id`, not by window index.

```python
# WRONG (data leakage)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Problem: Test subject's windows can be in training set!

# CORRECT (no leakage)
test_mask = subjects == test_subject
train_mask = ~test_mask
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
# Guarantee: Test subject's windows NEVER in training set
```

**Applied to**:
- Classical ML: `train_classical_ml.py` lines 221-227
- TCN: `train_tcn.py` lines 182-194
- Fusion: `train_fusion.py` lines 82-93

---

### 3.3 Mechanism 2: Temporal Causality in Labeling

**Problem**: Using stress event labels that occur **after** the window end time.

**Solution**: 5-minute prediction horizon ensures labels reflect **future** stress.

```python
# From shared/windowing.py
def create_labeled_windows(aligned, event_info, window_size_sec=120, 
                          horizons_minutes=[5]):
    """
    Label windows based on future stress events.
    
    Window labeled as stress if:
        stress_event_start - window_end <= 5 minutes
    
    This ensures:
    - Window at time t uses data from [t-120s, t]
    - Label predicts stress at t+5min
    - No future information in features
    """
    for window in windows:
        window_end = window["window_end"]
        
        # Find stress events starting within 5 minutes after window
        for event_start, event_end in stress_events:
            if 0 <= (event_start - window_end) <= 5*60:  # 5 min in seconds
                window["label_5min"] = 1
                break
```

**Guarantees**:
- Features: Computed from `[window_start, window_end]` (past data only)
- Label: Based on events starting **after** `window_end + 5min` (future state)
- **No temporal leakage**: Features never contain future information

---

### 3.4 Mechanism 3: Per-Fold Preprocessing

**Problem**: Computing normalization statistics or imputation values on entire dataset (including test set).

**Solution**: Fit preprocessing on **training set only**, then apply to test set.

#### Classical ML: Per-Fold Imputation

```python
# From train_classical_ml.py lines 232-235
for test_subject in unique_subjects:
    # Split first
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Fit imputer on train set only
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)  # Compute median from train
    X_test = imputer.transform(X_test)        # Apply train median to test
    
    # NOT: imputer.fit_transform(X)  ← Would use test data to compute median!
```

**Why this matters**:
- Median computed from training subjects only
- Test subject's values don't influence imputation
- Simulates deployment: impute new user's missing data using historical statistics

---

#### TCN: Per-Subject Normalization

```python
# From experiments/tcn/dataset.py lines 89-110
class VitaStressTCNDataset(Dataset):
    def __init__(self, windows, normalize=True, normalization_mode="subject"):
        """
        TCN dataset with per-subject z-score normalization.
        
        normalization_mode="subject":
            - For each window, normalize using that SUBJECT's own statistics
            - Statistics computed from ALL windows of that subject
            - NO cross-subject information leakage
        """
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        
        if normalize and normalization_mode == "subject":
            # Group windows by subject
            windows_by_subject = defaultdict(list)
            for w in windows:
                subject_id = w.get("subject_id")
                windows_by_subject[subject_id].append(w)
            
            # Compute per-subject statistics
            for subject_id, subject_windows in windows_by_subject.items():
                # Stack all windows for this subject
                all_sequences = np.stack([w["window_data"] for w in subject_windows])
                
                # Compute mean/std across ALL timesteps of ALL windows
                # Shape: (n_windows, seq_len=480, n_channels=8)
                subject_mean = all_sequences.mean(axis=(0, 1))  # Shape: (8,)
                subject_std = all_sequences.std(axis=(0, 1))    # Shape: (8,)
                
                # Store for this subject
                for w in subject_windows:
                    w["subject_mean"] = subject_mean
                    w["subject_std"] = subject_std
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        X = window["window_data"]  # Shape: (seq_len, n_channels)
        
        if self.normalize and self.normalization_mode == "subject":
            # Normalize using this subject's statistics
            subject_mean = window["subject_mean"]
            subject_std = window["subject_std"]
            X = (X - subject_mean) / (subject_std + 1e-8)
        
        return X, y
```

**Key property**: Subject normalization uses only that subject's own data → **no cross-subject leakage**.

**Why this is safe in LOSO**:
- Training subjects: Each subject normalized by their own statistics → no leakage
- Test subject: Normalized by their own statistics (computed from test windows)
- **Crucial**: Test subject's normalization stats don't influence training in any way

**Important note**: This is an **exception** to the "no test data in preprocessing" rule because:
1. Normalization is **per-subject**, not cross-subject
2. Test subject's stats are **self-contained** (don't affect training)
3. Simulates deployment: new user's baseline computed from their own data

---

### 3.5 Mechanism 4: Threshold Selection on Training Data

**Problem**: Optimizing decision threshold on test labels.

**Solution**: Select threshold on **training probabilities**, apply to test.

```python
# From train_classical_ml.py lines 264-280
for test_subject in unique_subjects:
    # Train model
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]  # Train probabilities
    y_test_proba = model.predict_proba(X_test)[:, 1]    # Test probabilities
    
    # Select threshold on TRAIN data (no peeking at test labels!)
    threshold_b1, _ = find_optimal_threshold(
        y_train,        # Train labels (known during training)
        y_train_proba,  # Train probabilities
        method="geometric_mean"
    )
    
    # Apply threshold to TEST data
    y_test_pred = (y_test_proba >= threshold_b1).astype(int)
    
    # Evaluate (now we use test labels for evaluation only)
    metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
```

**Why this prevents leakage**:
- Threshold optimized using `y_train` and `y_train_proba` (training set only)
- Test labels `y_test` **never** used until final evaluation step
- Simulates deployment: threshold chosen using historical data, applied to new users

**Common mistake**:
```python
# WRONG (data leakage)
threshold = find_optimal_threshold(y_test, y_test_proba)
# Problem: Optimizing threshold using test labels!
```

---

### 3.6 Mechanism 5: Nested LOSO for Meta-Learning

**Problem**: Training meta-model on test subject's base predictions.

**Solution**: Nested LOSO (already explained in Section 2.3).

**Quick recap**:
- Base models trained using LOSO (subject-independent)
- Meta-model trained using nested LOSO on base predictions
- For each test subject k, meta-model fitted on subjects 1..k-1, k+1..N
- **No leakage**: Meta-model never sees test subject's data

---

### 3.7 Summary: Leakage Prevention Checklist

| **Leakage Source** | **Prevention Mechanism** | **Implementation** |
|-------------------|-------------------------|-------------------|
| **Subject leakage** | Subject-based splitting (not random) | `test_mask = subjects == test_subject` |
| **Temporal leakage** | 5-minute prediction horizon (future labels) | `label_5min` computed from future events |
| **Preprocessing leakage** | Per-fold imputation (fit on train only) | `imputer.fit_transform(X_train)` then `transform(X_test)` |
| **Preprocessing (normalization)** | Per-subject z-score (self-contained) | TCN: `normalization_mode="subject"` |
| **Threshold leakage** | Threshold selected on train probabilities | `find_optimal_threshold(y_train, y_train_proba)` |
| **Meta-learning leakage** | Nested LOSO (meta-model trained per fold) | Meta-model fitted on N-1 subjects, tested on held-out subject |

**All mechanisms are implemented and verified.**

---

## 4. Metrics Reported

### 4.1 Primary Metrics (Threshold-Dependent)

These metrics depend on the decision threshold and reflect operational performance:

#### Recall (Sensitivity, True Positive Rate)

**Definition**:
```
Recall = TP / (TP + FN)
       = True Positives / Total Positive Samples
```

**Interpretation**:
- **What % of stress events were detected?**
- Clinical relevance: High recall ensures most stress onsets trigger warnings
- Trade-off: Higher recall often increases false alarms

**Why we report it**: Primary clinical goal is to **detect stress early**.

---

#### False Alarm Rate (FAR, False Positive Rate)

**Definition**:
```
FAR = FP / (FP + TN)
    = False Positives / Total Negative Samples
```

**Interpretation**:
- **What % of non-stress periods triggered false alarms?**
- User experience: High FAR reduces trust in system
- Trade-off: Lower FAR often reduces recall

**Why we report it**: Deployment feasibility depends on **tolerable false alarm rate**.

---

#### Specificity (True Negative Rate)

**Definition**:
```
Specificity = TN / (TN + FP)
            = 1 - FAR
```

**Interpretation**:
- **What % of non-stress periods were correctly identified?**
- Complementary to FAR (higher specificity = lower FAR)

**Why we report it**: Balances recall in G-mean optimization.

---

#### G-mean (Geometric Mean)

**Definition**:
```
G-mean = √(Recall × Specificity)
```

**Interpretation**:
- **Balanced performance metric for imbalanced data**
- Penalizes models with high recall but low specificity (or vice versa)
- Used for threshold optimization

**Why we report it**: Primary metric for threshold selection (Section 1.2).

**Example**:
```
Model A: Recall = 90%, Specificity = 50% → G-mean = 67%
Model B: Recall = 70%, Specificity = 70% → G-mean = 70%

Model B preferred: Better balance despite lower recall
```

---

#### Precision (Positive Predictive Value)

**Definition**:
```
Precision = TP / (TP + FP)
```

**Interpretation**:
- **What % of stress predictions were correct?**
- User perspective: How often is the alert accurate?
- Influenced by class imbalance (low precision common with imbalanced data)

**Why we report it**: Reflects user trust in positive predictions.

---

#### F1-Score

**Definition**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation**:
- Harmonic mean of precision and recall
- Balances both metrics

**Why we report it**: Standard metric for imbalanced classification (for completeness).

**Note**: We prioritize G-mean over F1 because G-mean includes specificity (F1 does not).

---

### 4.2 Secondary Metrics (Threshold-Independent)

These metrics characterize model performance across all possible thresholds:

#### AUROC (Area Under ROC Curve)

**Definition**:
- ROC curve: Plots Recall (TPR) vs FAR (FPR) at all thresholds
- AUROC: Area under this curve

**Interpretation**:
- **Probability that model ranks a random stress window higher than a random non-stress window**
- Range: [0, 1], where 0.5 = random chance, 1.0 = perfect ranking
- Threshold-independent: Evaluates ranking quality, not classification

**Why we report it**: Standard metric for evaluating probabilistic predictions.

**Robustness**: Unaffected by class imbalance (unlike accuracy).

---

#### PR-AUC (Precision-Recall Area Under Curve)

**Definition**:
- PR curve: Plots Precision vs Recall at all thresholds
- PR-AUC: Area under this curve

**Interpretation**:
- **Average precision across all recall levels**
- Range: [baseline (% positive), 1.0], where baseline = class imbalance ratio
- More sensitive to class imbalance than AUROC

**Why we report it**: Better than AUROC for **highly imbalanced data**.

**Example**:
```
Dataset: 20% stress (1:4 imbalance)
Random baseline: PR-AUC = 0.20, AUROC = 0.50

Model: PR-AUC = 0.45, AUROC = 0.75
→ Model significantly better than random on both metrics
```

---

### 4.3 Metrics NOT Reported (and Why)

#### Accuracy

**Definition**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Why we DON'T report it**:
- **Misleading for imbalanced data**
- Example: 80% non-stress → always predicting non-stress gives 80% accuracy
- Dominated by majority class (true negatives)
- Ignores clinical cost asymmetry (FN ≠ FP in importance)

**See**: Class imbalance handling document (Section 2.2) for full explanation.

---

#### Balanced Accuracy

**Definition**:
```
Balanced Accuracy = (Recall + Specificity) / 2
```

**Why we DON'T report it**:
- Similar to G-mean but uses arithmetic mean (less sensitive to imbalance)
- G-mean (geometric mean) penalizes imbalance more strongly → preferred for severe imbalance

**We use G-mean instead.**

---

### 4.4 Confusion Matrix Components

For each fold, we report the full confusion matrix:

```
                  Predicted
                 No Stress | Stress
Actual  -------------------------
No Stress (0)  |    TN    |   FP    
Stress (1)     |    FN    |   TP    
```

**Reported values**:
- **True Negative (TN)**: Correctly identified non-stress windows
- **False Positive (FP)**: Non-stress windows incorrectly predicted as stress (false alarms)
- **False Negative (FN)**: Stress windows incorrectly predicted as non-stress (missed detections)
- **True Positive (TP)**: Correctly identified stress windows

**Derived metrics**:
- Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- FAR = FP / (FP + TN)
- Precision = TP / (TP + FP)

---

### 4.5 Aggregation: Fold-Level vs Dataset-Level

We report metrics at **two levels**:

#### Fold-Level Metrics

**Per-fold statistics** (one value per test subject):
- Recall, FAR, Specificity, Precision, G-mean, F1
- Reported as: mean ± std across 21 folds

**Example**:
```json
{
  "recall_mean": 0.72,
  "recall_std": 0.14,
  "far_mean": 0.28,
  "far_std": 0.11
}
```

**Interpretation**:
- Mean: Average performance across subjects
- Std: Variability in performance (subject-dependent differences)
- High std indicates some subjects harder to predict

---

#### Dataset-Level Metrics

**Aggregate across all windows** (pool all 21 folds):
- Concatenate predictions from all folds: `y_true_all`, `y_pred_all`, `y_proba_all`
- Compute metrics on pooled data

**Example**:
```json
{
  "recall": 0.71,
  "far": 0.29,
  "auroc": 0.75,
  "pr_auc": 0.42
}
```

**Interpretation**:
- Overall performance across entire dataset
- Accounts for varying fold sizes (subjects have different window counts)
- **AUROC and PR-AUC**: Only meaningful at dataset level (threshold-independent)

---

### 4.6 Metrics Summary Table

| **Metric** | **Type** | **Range** | **Interpretation** | **Primary Use** |
|-----------|----------|-----------|-------------------|----------------|
| **Recall** | Threshold-dependent | [0, 1] | % of stress detected | Clinical efficacy |
| **FAR** | Threshold-dependent | [0, 1] | % of false alarms | Deployment feasibility |
| **Specificity** | Threshold-dependent | [0, 1] | % of non-stress correct | Balancing FAR |
| **Precision** | Threshold-dependent | [0, 1] | % of alerts correct | User trust |
| **G-mean** | Threshold-dependent | [0, 1] | Balanced performance | Threshold selection |
| **F1** | Threshold-dependent | [0, 1] | Harmonic mean of precision/recall | Standard comparison |
| **AUROC** | Threshold-independent | [0.5, 1] | Ranking quality | Overall model quality |
| **PR-AUC** | Threshold-independent | [baseline, 1] | Average precision | Imbalance-robust quality |

---

## 5. Evaluation Workflow

### 5.1 Classical ML Evaluation

```
1. Load data (1733 windows, 21 subjects, 39 features)
   ↓
2. LOSO Cross-Validation (21 folds)
   For each test subject:
     a. Split: Train (20 subjects) vs Test (1 subject)
     b. Per-fold imputation (fit on train, transform test)
     c. Train model (LR/RF/SVM with class_weight="balanced")
     d. Get probabilities (train and test)
     e. Select 3 thresholds on train:
        - B1: Unconstrained G-mean
        - B2: Recall ≥ 70% constrained G-mean
        - B3: FAR ≤ 30% constrained G-mean
     f. Apply thresholds to test
     g. Evaluate (recall, FAR, specificity, precision, G-mean, F1)
     h. Save predictions to CSV
   ↓
3. Aggregate results
   - Fold-level: mean ± std across 21 folds
   - Dataset-level: pool all predictions, compute AUROC/PR-AUC
   ↓
4. Save results
   - Metrics: results/{model}/_{model}_b{1,2,3}_metrics.json
   - Predictions: results/{model}/{model}_b{1,2,3}_predictions.csv
   - Fold metrics: results/{model}/{model}_b{1,2,3}_fold_metrics.csv
```

---

### 5.2 TCN Evaluation

```
1. Load data (1733 windows, 21 subjects, 8 channels)
   ↓
2. LOSO Cross-Validation (21 folds)
   For each test subject:
     a. Split windows by subject
     b. Create datasets with per-subject normalization
     c. Train TCN with Focal Loss, early stopping
     d. Get probabilities (train and test)
     e. Select 3 thresholds on train (B1, B2, B3)
     f. Apply thresholds to test
     g. Evaluate
     h. Save predictions
   ↓
3. Aggregate results (same as classical ML)
   ↓
4. Save results (same structure)
```

---

### 5.3 Fusion Evaluation

```
1. Load base predictions
   - LR predictions: results/lr/lr_b1_predictions.csv
   - TCN predictions: results/tcn/tcn_b1_predictions.csv
   ↓
2. Decision-Level Fusion (3 strategies)
   
   Strategy 1: Logical OR
     - For each window: stress if (LR=stress OR TCN=stress)
     - Uses base thresholds (no additional training)
   
   Strategy 2: Cascade
     - LR screens (high recall), TCN confirms (high precision)
     - Uses base thresholds (no additional training)
   
   Strategy 3: Stacked Generalization
     - Nested LOSO (21 folds):
       For each test subject:
         a. Hold out test subject
         b. Train meta-LR on base probabilities (train subjects)
         c. Select 3 thresholds on train
         d. Apply to test subject
         e. Evaluate
   ↓
3. Aggregate results (same structure)
   ↓
4. Save results
```

---

## 6. Result Files Structure

### 6.1 Directory Layout

```
experiments/final_models/results/
├── lr/
│   ├── lr_b1_metrics.json           # Aggregate metrics (B1: unconstrained G-mean)
│   ├── lr_b1_predictions.csv        # Per-window predictions
│   ├── lr_b1_fold_metrics.csv       # Per-fold metrics
│   ├── lr_b2_metrics.json           # B2: Recall ≥ 70%
│   ├── lr_b2_predictions.csv
│   ├── lr_b2_fold_metrics.csv
│   ├── lr_b3_metrics.json           # B3: FAR ≤ 30%
│   ├── lr_b3_predictions.csv
│   └── lr_b3_fold_metrics.csv
├── rf/                              # Same structure
├── svm/                             # Same structure
├── tcn/                             # Same structure
├── fusion_or/                       # Logical OR fusion
├── fusion_cascade/                  # Cascade fusion
└── fusion_stacked/                  # Stacked generalization
```

---

### 6.2 Metrics JSON Format

**File**: `results/{model}/{model}_b1_metrics.json`

**Content** (example):
```json
{
  "model_name": "LR",
  "threshold_strategy": "b1_gmean",
  
  // Dataset-level metrics (pooled across all folds)
  "recall": 0.714,
  "false_alarm_rate": 0.287,
  "specificity": 0.713,
  "precision": 0.423,
  "gmean": 0.713,
  "f1": 0.532,
  "auroc": 0.753,
  "pr_auc": 0.418,
  
  // Fold-level statistics (mean ± std across 21 folds)
  "recall_mean": 0.721,
  "recall_std": 0.142,
  "far_mean": 0.285,
  "far_std": 0.108,
  "specificity_mean": 0.715,
  "specificity_std": 0.108,
  "precision_mean": 0.441,
  "precision_std": 0.176,
  "gmean_mean": 0.712,
  "gmean_std": 0.094,
  "f1_mean": 0.531,
  "f1_std": 0.143,
  
  // Sample counts
  "n_samples": 1733,
  "n_positive": 421,
  "n_negative": 1312,
  
  // Confusion matrix
  "true_positive": 301,
  "false_positive": 411,
  "false_negative": 120,
  "true_negative": 901
}
```

---

### 6.3 Predictions CSV Format

**File**: `results/{model}/{model}_b1_predictions.csv`

**Columns**:
- `subject_id`: Subject identifier
- `window_start`: Window start timestamp (Unix seconds)
- `window_end`: Window end timestamp (Unix seconds)
- `y_true`: Ground truth label (0=non-stress, 1=stress)
- `y_pred`: Predicted label (0=non-stress, 1=stress)
- `y_proba`: Predicted probability of stress (0-1)
- `threshold`: Decision threshold used for this fold

**Example**:
```csv
subject_id,window_start,window_end,y_true,y_pred,y_proba,threshold
id_0a73ef1b-da67-43ff-b61a-f98c151be799,1234567890,1234568010,0,0,0.234,0.42
id_0a73ef1b-da67-43ff-b61a-f98c151be799,1234567950,1234568070,1,1,0.678,0.42
...
```

**Usage**:
- Error analysis: identify false positives, false negatives
- Subject-wise performance: group by `subject_id`
- Temporal analysis: plot predictions over time using timestamps
- Meta-learning input: load base probabilities for fusion

---

### 6.4 Fold Metrics CSV Format

**File**: `results/{model}/{model}_b1_fold_metrics.csv`

**Columns**:
- `fold`: Fold index (0-20)
- `subject`: Test subject ID for this fold
- `n_samples`: Number of windows in test set
- `n_positive`: Number of stress windows in test set
- `recall`, `far`, `specificity`, `precision`, `gmean`, `f1`: Metrics for this fold

**Example**:
```csv
fold,subject,n_samples,n_positive,recall,far,specificity,precision,gmean,f1
0,id_0a73ef1b-da67-43ff-b61a-f98c151be799,83,21,0.714,0.290,0.710,0.441,0.712,0.545
1,id_1b2c3d4e-5f6a-7b8c-9d0e-1f2a3b4c5d6e,91,18,0.778,0.247,0.753,0.467,0.765,0.583
...
```

**Usage**:
- Identify difficult subjects (low recall, high FAR)
- Analyze subject-level variability (std across folds)
- Stratified analysis: correlate performance with subject characteristics

---

## 7. Rationale for Design Choices

### 7.1 Why LOSO over K-Fold?

**K-Fold Cross-Validation** (alternative):
- Split data into K random folds
- Each fold is test set once

**Problem**:
- Same subject's windows can appear in both train and test
- Model memorizes subject-specific patterns
- **Overestimates** real-world performance

**LOSO (our choice)**:
- Each **subject** is test set once
- Guarantees subject-independence
- **Simulates deployment**: new user with no historical data

**Empirical evidence**: LOSO typically reduces performance by 5-15% compared to random K-fold, reflecting realistic generalization.

---

### 7.2 Why Not Hold-Out Test Set?

**Hold-out test set** (alternative):
- Split: 70% train, 30% test (subject-level)
- Train once, evaluate once

**Problems**:
1. **Small dataset (N=21 subjects)**: 30% = only 6 subjects for testing → high variance
2. **Single estimate**: No insight into subject-level variability
3. **Inefficient**: Most subjects never used for testing

**LOSO (our choice)**:
- Uses all subjects for testing (100% utilization)
- 21 estimates → robust statistics (mean ± std)
- Reveals subject-level performance variability

**When to use hold-out**: Large datasets (N>100 subjects) where single test estimate is reliable.

---

### 7.3 Why Geometric Mean for Threshold Selection?

**Alternatives considered**:

1. **F1-score maximization**:
   - F1 = harmonic mean of precision and recall
   - **Problem**: Ignores true negatives (specificity not considered)
   - Can achieve high F1 with high FAR (poor user experience)

2. **Youden's Index** (TPR - FPR):
   - Equivalent to maximizing (Recall - FAR)
   - **Problem**: Arithmetic difference → less sensitive to imbalance

3. **Fixed threshold (0.5)**:
   - Default for many classifiers
   - **Problem**: Biased toward majority class in imbalanced data
   - Often results in low recall

4. **Constrained optimization** (e.g., Recall ≥ 70%):
   - Imposes minimum recall requirement
   - **Problem**: Arbitrary constraint (why 70%? deployment-specific)
   - We use this for **Ablation B2/B3**, but not as primary method

**Geometric mean (our choice)**:
```
G-mean = √(Recall × Specificity)
```

**Advantages**:
- Balances both classes (includes true negatives via specificity)
- Geometric mean penalizes imbalance strongly (if either metric low, G-mean is low)
- No arbitrary constraints (finds natural optimal point)
- Standard practice for imbalanced classification

**Example**:
```
Model A: Recall=90%, Specificity=50% → G-mean=67%, F1=75%
Model B: Recall=70%, Specificity=70% → G-mean=70%, F1=70%

G-mean prefers Model B (better balance)
F1 prefers Model A (higher recall dominates)

For stress prediction: Model B likely better (acceptable recall + low FAR)
```

---

### 7.4 Why Report Both Fold-Level and Dataset-Level Metrics?

**Fold-level (mean ± std)**:
- Reveals subject-level **variability**
- High std indicates some subjects are harder to predict
- Important for understanding model **robustness**

**Dataset-level (pooled)**:
- Overall **aggregate performance**
- Accounts for varying window counts per subject
- Threshold-independent metrics (AUROC, PR-AUC) only meaningful here

**Example interpretation**:
```
Recall: 0.72 ± 0.14 (fold-level)
Recall: 0.71 (dataset-level)

Interpretation:
- Average subject-level recall: 72%
- Overall recall: 71% (slightly lower due to fold size differences)
- Std = 14% → significant subject variability (some subjects 58%, others 86%)
```

**Both are necessary** for complete evaluation.

---

## 8. Comparison with Literature

### 8.1 Common Pitfalls in Stress Prediction Evaluation

| **Pitfall** | **Our Approach** | **Why It Matters** |
|-------------|-----------------|-------------------|
| **Subject-dependent split** (random train-test) | LOSO (subject-independent) | Prevents memorizing subject patterns |
| **Reporting only accuracy** | Recall, FAR, AUROC, PR-AUC | Avoids misleading metrics for imbalanced data |
| **Threshold optimization on test set** | Threshold selected on train set | Prevents data leakage |
| **Global normalization** (fit on entire dataset) | Per-fold imputation (fit on train only) | Prevents preprocessing leakage |
| **No subject-level statistics** | Report mean ± std across folds | Shows model robustness and variability |
| **Single threshold evaluation** | 3 threshold strategies (Ablation B) | Explores recall-FAR trade-off space |
| **Meta-learning without nested CV** | Nested LOSO for stacking | Prevents meta-model leakage |

**Our evaluation is aligned with best practices.**

---

### 8.2 Reproducibility

**Our evaluation protocol ensures reproducibility** through:

1. **Fixed hyperparameters**: All model parameters in `config.py` (no manual tuning)
2. **Deterministic splits**: LOSO splits are deterministic (subject IDs sorted)
3. **Random seeds**: `random_state=42` for all stochastic operations
4. **Saved predictions**: Full predictions saved to CSV (can recompute metrics)
5. **Comprehensive logging**: Training logs include all hyperparameters and data statistics
6. **Documented pipeline**: All preprocessing, windowing, and evaluation steps documented

**Reproducibility checklist**:
- ✅ Subject-level split (not random)
- ✅ Fixed random seeds
- ✅ Saved predictions and metrics
- ✅ Documented hyperparameters
- ✅ Version-controlled code

---

## 9. Conclusion

Our evaluation protocol ensures **subject-independent assessment** with **rigorous data leakage prevention** through:

1. **LOSO cross-validation**: Every subject tested independently
2. **Nested LOSO for meta-learning**: Meta-model never sees test subject
3. **Per-fold preprocessing**: Imputation and threshold selection on training data only
4. **Comprehensive metrics**: Recall, FAR, Specificity, G-mean, AUROC, PR-AUC
5. **Two-level aggregation**: Fold-level variability + dataset-level overall performance

**Key takeaway**: Our evaluation **simulates realistic deployment** where models are applied to **unseen subjects** with **no historical data**. This ensures reported performance reflects **true generalization capability**, not memorization of training subjects.

Our protocol aligns with best practices in wearable stress prediction and prevents common pitfalls that lead to overly optimistic results.

---

## References

**Subject-Independent Evaluation**:
- Schmidt, P., et al. (2018). "Introducing WESAD, a multimodal dataset for wearable stress and affect detection." *ICMI*.
- Gjoreski, M., et al. (2020). "Datasets for cognitive load inference using wearable sensors and psychological traits." *ACM TIST*.

**LOSO Cross-Validation**:
- Arlot, S., & Celisse, A. (2010). "A survey of cross-validation procedures for model selection." *Statistics Surveys*.
- Cawley, G. C., & Talbot, N. L. (2010). "On over-fitting in model selection and subsequent selection bias in performance evaluation." *JMLR*.

**Nested Cross-Validation**:
- Varma, S., & Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection." *BMC Bioinformatics*.
- Krstajic, D., et al. (2014). "Cross-validation pitfalls when selecting and assessing regression and classification models." *J. Cheminformatics*.

**Data Leakage Prevention**:
- Kaufman, S., et al. (2012). "Leakage in data mining: Formulation, detection, and avoidance." *ACM TKDD*.
- Kapoor, S., & Narayanan, A. (2022). "Leakage and the reproducibility crisis in ML-based science." *arXiv*.

**Metrics for Imbalanced Data**:
- Kubat, M., & Matwin, S. (1997). "Addressing the curse of imbalanced training sets: one-sided selection." *ICML*.
- Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLoS ONE*.
