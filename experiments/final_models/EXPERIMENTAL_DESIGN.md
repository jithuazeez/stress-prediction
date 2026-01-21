# Chapter 3: Experimental Design

This chapter presents the experimental methodology for evaluating wearable-based emotional stress prediction. The design addresses three research questions:

1. **RQ1:** Can short-term historical wearable sensor data be used to predict the onset of emotional stress in unseen subjects?
2. **RQ2:** How does a Temporal Convolutional Network compare to classical machine learning models in terms of recall–specificity trade-offs for emotional stress prediction in class-imbalanced, subject-independent settings?
3. **RQ3:** Can decision-level fusion strategies, such as logical operators and stacked generalisation, improve the balance between stress prediction recall and false alarm rate in subject-independent settings?

---

## 3.1 Problem Formulation

### Task Definition

This study frames stress prediction as a **binary near-future stress prediction** task:

> *Given physiological signals from time [t − 120s, t], predict whether emotional stress will onset within the interval [t, t + 5 min].*

| Component | Specification |
|-----------|---------------|
| **Input** | 8-channel multimodal physiological time series (120 seconds) |
| **Output** | Binary label: 1 = stress onset predicted, 0 = no stress |
| **Prediction Horizon** | 5 minutes into the future |
| **Task Type** | Binary classification (forecasting) |

### Why Forecasting, Not Detection

This study explicitly distinguishes **prediction** from **detection**:

| Aspect | Detection | Prediction (This Study) |
|--------|-----------|-------------------------|
| Question | "Is the subject currently stressed?" | "Will stress **begin** in the next 5 minutes?" |
| Temporal focus | Present state | Future onset |
| Signal characteristics | Obvious stress signatures (elevated HR, sweating) | Subtle precursor patterns |
| Clinical utility | Reactive intervention | **Proactive intervention** |

**Labelling rule:**
```
label = 1  if  emotional_stress_onset ∈ [window_end, window_end + 5 min]
label = 0  otherwise (including during ongoing stress)
```

This formulation forces the model to learn subtle **anticipatory patterns** rather than detecting obvious stress manifestations. It enables preventive interventions—a clinically more valuable outcome than reactive detection.

---

## 3.2 Dataset Description

### VitaStress Dataset

This study uses the **VitaStress** dataset, a multimodal physiological dataset collected from wearable sensors during controlled stress protocols.

| Property | Value |
|----------|-------|
| **Participants** | 21 subjects |
| **Collection Context** | Laboratory-based protocol with controlled stress induction |
| **Duration** | ~60 minutes per subject |
| **Stress Protocol** | Cognitive tasks (mental arithmetic, Stroop test), public speaking |
| **Baseline Protocol** | Confirmed rest periods (5 minutes) |
| **Sensors** | Accelerometer, PPG (photoplethysmography), heat flux sensor |

### Sensor Modalities

| Sensor | Signal | Native Sampling Rate | Physiological Basis |
|--------|--------|---------------------|---------------------|
| **Accelerometer** | acc_x, acc_y, acc_z | ~32 Hz | Motor activity, fidgeting, tremors |
| **PPG** | Photoplethysmogram | ~64 Hz | Cardiac activity (for HR/HRV extraction) |
| **Heat Flux Sensor** | skin_temp, heatflux, cbt | 1 Hz | Thermoregulation, peripheral vasoconstriction |

### Stress Event Types

The dataset distinguishes between **emotional** and **physical** stress:

| Event Type | Label | Examples |
|------------|-------|----------|
| **Emotional Stress** | 1 (Positive) | Cognitive tasks, public speaking |
| **Physical Stress** | 0 (Negative) | Treadmill running, exercise |
| **Baseline** | 0 (Negative) | Rest periods |

Physical stress (exercise) produces similar physiological responses (elevated HR, increased temperature) but is **not** emotional stress. This labelling helps the model learn: *"High HR + sitting = stress"* vs *"High HR + running = exercise"*.

### Dataset Suitability

VitaStress is suitable for this study because:
- **Multimodal signals:** Captures complementary stress markers (cardiac, thermal, motor)
- **Controlled protocol:** Known stress onset times enable precise labelling
- **Subject diversity:** 21 subjects enables subject-independent evaluation
- **Temporal density:** Sufficient samples for 5-minute prediction horizon

### Known Limitations

- **Laboratory setting:** May not fully represent naturalistic stress
- **Short duration:** ~60 minutes per subject limits long-term patterns
- **Class imbalance:** Stress events are sparse (~12% of windows)

---

## 3.3 Signal Preprocessing

Preprocessing is organised by signal modality to ensure physiological validity.

### 3.3.1 Cardiovascular Signals (PPG → HR/HRV)

Raw PPG signals are processed at **native 64 Hz** before alignment to preserve cardiac waveform structure:

| Step | Method | Rationale |
|------|--------|-----------|
| **Quality filtering** | Remove samples with quality flag < 3 | Exclude motion-corrupted beats |
| **Zero removal** | Replace zeros with NaN | Device artifact handling |
| **Outlier removal** | 1st–99th percentile clipping | Remove physiologically implausible values |
| **Gap interpolation** | Linear interpolation (max 10 samples) | Handle small dropouts |
| **Detrending** | DC offset removal + scipy.signal.detrend | Remove baseline wander |
| **Bandpass filtering** | 0.7–3.5 Hz (HeartPy) | Isolate cardiac frequencies (42–210 BPM) |
| **Peak detection** | HeartPy with quotient-filter RR cleaning | Robust R-peak identification |

**Critical design decision:** HR and HRV features are extracted at **native 64 Hz** using HeartPy, then aligned to 4 Hz. Downsampling raw PPG to 1 Hz or 4 Hz destroys the cardiac waveform structure.

### 3.3.2 Accelerometer Signals

| Step | Method | Rationale |
|------|--------|-----------|
| **Resampling** | 32 Hz → 4 Hz (decimation) | Align to common time grid |
| **Normalisation** | Subject-wise z-score | Handle inter-subject baseline differences |

Separate axes (x, y, z) are preserved rather than computing magnitude alone, as directional information aids stress-related fidgeting detection.

### 3.3.3 Temperature / Heat Flux Signals

| Step | Method | Rationale |
|------|--------|-----------|
| **Resampling** | 1 Hz → 4 Hz (interpolation) | Align to common time grid |
| **Normalisation** | Subject-wise z-score | Handle thermoregulatory baseline differences |

### 3.3.4 Signal Alignment

All modalities are aligned to a **common 4 Hz grid** for the final models:

```
ACC (32 Hz)  ──┬──→ Decimate ──────────────────┐
PPG (64 Hz)  ──┼──→ HeartPy → HR/HRV (64 Hz) ──┼──→ 4 Hz Grid ──→ Windowing
Temp (1 Hz)  ──┴──→ Interpolate ───────────────┘
```

### 3.3.5 Normalisation Strategy

**Subject-wise z-score normalisation** is applied to handle inter-subject variability:

```python
For each channel c and subject s:
    μ_s,c = mean(all values of channel c for subject s)
    σ_s,c = std(all values of channel c for subject s)
    
    normalised_value = (raw_value - μ_s,c) / σ_s,c
```

This ensures that models learn **relative deviations from personal baselines** rather than absolute physiological values, which vary substantially between individuals.

---

## 3.4 Windowing Strategy and Label Construction

### Window Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Window length** | 120 seconds | Captures HRV dynamics (RMSSD requires ~2 min for reliable estimation); literature supports ≥90s for stress detection |
| **Overlap** | 50% | Increases sample count while maintaining temporal resolution |
| **Prediction horizon** | 5 minutes | Clinically actionable (sufficient time for intervention) |
| **Skip first** | 1 minute | Sensor settling period |

### Justification for 120-Second Windows

The 120-second window duration balances competing requirements:

1. **Physiological validity:** Heart rate variability metrics (RMSSD, SDNN) require ~2 minutes of data for reliable estimation
2. **Feature stability:** Short windows (<60s) produce noisy statistical features
3. **Temporal resolution:** Longer windows (>180s) reduce the number of samples and delay predictions
4. **Literature support:** Prior work demonstrates improved performance with ≥90-second windows for stress-related physiological pattern recognition

### Label Construction (Prediction, Not Detection)

Labels are constructed to enforce **pure prediction**:

```python
For each window ending at time t:
    stress_events = get_emotional_stress_onsets()  # Cognitive, Public Speaking
    
    label = 1 if any(t ≤ event_onset ≤ t + 5 min for event in stress_events)
    label = 0 otherwise
```

**Key constraints:**
- Only **emotional stress** onsets are labelled as positive
- Physical stress (exercise) is labelled as **negative**
- Ongoing stress periods are labelled as **negative** (prediction of onset only)
- Windows during stress are excluded to prevent detection shortcuts

### Overlap Safety with LOSO

50% overlap is safe because **all windows from the same subject remain in the same split**:
- No data leakage between subjects
- Overlap increases samples **within** each subject's data
- Autocorrelation is handled by subject-level cross-validation

---

## 3.5 Feature Extraction (Classical Models)

Classical machine learning models use **39 hand-crafted statistical features** extracted from each 120-second window.

### Feature Groups by Modality

| Modality | Feature Count | Features |
|----------|---------------|----------|
| **Accelerometer** | 15 | Top features selected by LR coefficient magnitude |
| **Temperature** | 7 | Statistical summaries + temporal dynamics |
| **Heat Flux / CBT** | 9 | Enhanced thermal features |
| **HR / HRV** | 8 | Time-domain HRV metrics from HeartPy |
| **Total** | **39** | |

### Accelerometer Features (15 Features)

Features were reduced from 41 to **15 top-ranked features** based on Logistic Regression coefficient analysis:

| Rank | Feature | |coef| | Physiological Basis |
|------|---------|-------|---------------------|
| 1 | `acc_dominant_freq_power` | 0.67 | Rhythmicity of movement (distinguishes exercise from fidgeting) |
| 2 | `acc_magnitude_max` | 0.67 | Peak movement intensity |
| 3 | `acc_sma` | 0.64 | Signal magnitude area (overall activity) |
| 4 | `acc_zcr` | 0.62 | Zero-crossing rate (oscillatory motion) |
| 5 | `acc_spectral_entropy` | 0.52 | Frequency randomness (irregular = stress-related) |
| 6 | `acc_magnitude_std` | 0.50 | Movement variability |
| 7 | `acc_jerk_max` | 0.50 | Maximum sudden movement (tremors) |
| 8 | `acc_magnitude_skewness` | 0.45 | Distribution asymmetry |
| 9 | `acc_jerk_mean` | 0.39 | Average jerkiness (fidgeting) |
| 10 | `acc_y_std` | 0.39 | Y-axis variability |
| 11 | `acc_x_std` | 0.33 | X-axis variability |
| 12 | `acc_magnitude_mean` | 0.32 | Average movement |
| 13 | `acc_ima` | 0.32 | Integral of magnitude |
| 14 | `acc_magnitude_kurtosis` | 0.26 | Distribution peakedness |
| 15 | `acc_jerk_energy` | 0.24 | Energy in sudden movements |

**Features removed (26 features):**
- **Posture features** (tilt_x/y/z, roll/pitch): Captured experimental setup (sitting vs cycling), not true stress
- **Redundant statistics**: min, median, range, IQR, z_std
- **Low-importance frequency features**: dominant_freq, freq_ratios, PSD bands
- **Activity flags**: is_stationary, is_walking, motion_flag

### Temperature Features (7 Features)

| Feature | Description |
|---------|-------------|
| `temp_mean` | Mean skin temperature |
| `temp_std` | Temperature variability |
| `temp_min`, `temp_max`, `temp_range` | Range statistics |
| `temp_slope` | Linear trend (vasoconstriction indicator) |
| `temp_change` | End - start difference |

### Heat Flux / CBT Features (9 Features)

| Feature | Description |
|---------|-------------|
| `heatflux_mean`, `heatflux_std` | Heat flux statistics |
| `heatflux_min`, `heatflux_max`, `heatflux_range` | Range statistics |
| `heatflux_change` | Temporal change |
| `cbt_mean`, `cbt_std`, `cbt_change` | Core body temperature |

### HR / HRV Features (8 Features)

Extracted using HeartPy at native 64 Hz PPG sampling rate:

| Feature | Description | Stress Response |
|---------|-------------|-----------------|
| `hr_bpm` | Mean heart rate | ↑ During stress |
| `hr_std` | Heart rate variability (beat-to-beat) | ↑ During stress |
| `hrv_mean_rr` | Mean RR interval | ↓ During stress |
| `hrv_sdnn` | Standard deviation of RR intervals | ↓ During stress |
| `hrv_rmssd` | Root mean square of successive differences | ↓ During stress (parasympathetic withdrawal) |
| `hrv_pnn50` | % of successive RR differences > 50ms | ↓ During stress |
| `hrv_pnn20` | % of successive RR differences > 20ms | ↓ During stress |
| `hrv_sdsd` | Standard deviation of successive differences | ↓ During stress |

**Excluded features:**
- Frequency-domain HRV (LF, HF, LF/HF ratio): Unreliable for 120-second windows (require 5+ minutes)
- Breathing rate: Spectral analysis unreliable for short windows

### Feature Standardisation

Per-fold standardisation is applied during LOSO cross-validation:

```python
# Inside each LOSO fold
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)   # Fit on training data
X_test = imputer.transform(X_test)         # Transform test data (no leakage)
```

### Relevance of Hand-Crafted Features

Despite advances in deep learning, hand-crafted features remain relevant because:
1. **Interpretability:** Clinicians can understand which physiological markers drive predictions
2. **Sample efficiency:** Work well with limited training data (21 subjects)
3. **Computational efficiency:** Fast extraction and inference
4. **Competitive performance:** Literature shows classical ML remains competitive for wearable stress detection

---

## 3.6 Models Evaluated

### 3.6.1 Classical Machine Learning Models

Three classical models are evaluated, representing different inductive biases:

#### Logistic Regression (LR)

| Property | Value |
|----------|-------|
| **Regularisation** | L2 (C=0.1) |
| **Solver** | SAGA |
| **Class weighting** | Balanced |

**Rationale for inclusion:**
- Strong baseline with well-calibrated probability outputs
- Interpretable coefficients reveal feature importance
- Linear decision boundary provides upper bound for linear separability

**Strengths:** Interpretability, calibration, computational efficiency  
**Weaknesses:** Cannot capture nonlinear feature interactions

#### Random Forest (RF)

| Property | Value |
|----------|-------|
| **Trees** | 100 |
| **Max depth** | 5 |
| **Max features** | log2 |
| **Class weighting** | Balanced |

**Rationale for inclusion:**
- Captures nonlinear feature interactions
- Robust to outliers and missing data
- Feature importance via permutation

**Strengths:** Nonlinearity, robustness, ensemble benefits  
**Weaknesses:** Less interpretable, may overfit with limited data

#### Support Vector Machine (SVM)

| Property | Value |
|----------|-------|
| **Kernel** | RBF |
| **C** | 10.0 |
| **Gamma** | 0.1 |
| **Class weighting** | Balanced |

**Rationale for inclusion:**
- Effective in high-dimensional spaces
- Kernel trick captures complex boundaries
- Different inductive bias from tree-based methods

**Strengths:** High-dimensional effectiveness, kernel flexibility  
**Weaknesses:** Poor probability calibration, sensitive to scaling

### 3.6.2 Deep Learning Model: Temporal Convolutional Network (TCN)

#### Architecture

```
Input: [batch, 8 channels, 480 timesteps (4 Hz × 120s)]
    ↓
TCN Encoder:
    - 8 dilated convolutional blocks
    - Channels: [16, 16, 16, 16, 16, 16, 16, 16]
    - Dilations: [1, 2, 4, 8, 16, 32, 64, 128]
    - Kernel size: 3
    - Dropout: 0.3
    - Receptive field: 256 timesteps (64 seconds)
    ↓
Global pooling (last timestep)
    ↓
Fully connected (128 → 2)
    ↓
Output: Binary classification
```

| Property | Value |
|----------|-------|
| **Batch size** | 16 |
| **Learning rate** | 1e-3 |
| **Epochs** | 100 (early stopping) |
| **Patience** | 15 epochs |
| **Loss** | Focal Loss (α=0.88, γ=2.0) |
| **Optimiser** | Adam |

#### Why TCN Over Alternative Architectures

**TCN vs Standard CNN:**
- **Causal convolutions:** Ensure no future information leakage
- **Dilated convolutions:** Exponentially increasing receptive field captures both short-term and long-term patterns
- **Parameter efficiency:** Fewer parameters than equivalent receptive field CNN

**TCN vs RNN/LSTM:**
- **Parallelisation:** Convolutions are fully parallelisable (faster training)
- **Stable gradients:** No vanishing/exploding gradient problem
- **Consistent receptive field:** Fixed, known receptive field (256 timesteps)
- **Simpler architecture:** No hidden state management

**TCN vs Transformer:**
- **Computational efficiency:** O(n) vs O(n²) complexity
- **Inductive bias:** Built-in temporal locality appropriate for physiological signals
- **Data efficiency:** Works well with limited samples (21 subjects)

---

## 3.7 Decision-Level Fusion Strategies

Decision-level fusion combines predictions from complementary models (LR and TCN) to potentially improve the recall–FAR balance.

### 3.7.1 Fusion Inputs

Fusion operates on **probability outputs** from base models:
- `P_LR`: Logistic Regression probability of stress
- `P_TCN`: TCN probability of stress

### 3.7.2 Logical OR Fusion

```
Prediction = 1 if (P_LR ≥ thr_LR) OR (P_TCN ≥ thr_TCN)
           = 0 otherwise
```

| Property | Expected Behaviour |
|----------|-------------------|
| **Recall** | ↑ Increases (catches events either model detects) |
| **FAR** | ↑ Increases (union of false alarms) |
| **Use case** | When missing stress is costly (safety-critical) |

### 3.7.3 Cascade Fusion

```
Prediction = 0 if P_LR < thr_LR         # LR rejects → final answer
           = (P_TCN ≥ thr_TCN) otherwise # LR accepts → defer to TCN
```

| Property | Expected Behaviour |
|----------|-------------------|
| **Recall** | ↓ Decreases (requires agreement) |
| **FAR** | ↓↓ Significantly decreases (conservative) |
| **Use case** | When false alarms are costly (user trust, intervention fatigue) |

### 3.7.4 Stacked Generalisation (Meta-Learner)

```
Meta-features: X_meta = [P_LR, P_TCN]
Meta-model: Logistic Regression (C=1.0)
Prediction: Apply threshold to meta-model probability
```

| Property | Expected Behaviour |
|----------|-------------------|
| **Recall/FAR** | Learned optimal weighting |
| **Flexibility** | Adapts to which base model is more reliable |
| **Use case** | General-purpose improvement |

**Nested LOSO for Leakage Prevention:**

Stacked generalisation requires careful cross-validation to avoid meta-learner leakage:

```python
For each test_subject s:
    # Outer loop: Test on subject s
    Base predictions for subject s: [P_LR(s), P_TCN(s)]
    
    # Use OUT-OF-FOLD base predictions for meta-training
    # (predictions from when each training subject was held out)
    Meta-train data: [P_LR(t), P_TCN(t)] for t ∈ train_subjects
    
    Meta-model.fit(meta_train_features, meta_train_labels)
    Final prediction = Meta-model.predict([P_LR(s), P_TCN(s)])
```

### 3.7.5 Why Decision-Level Fusion

Fusion operates at the **decision level** (probabilities) rather than feature level because:
1. **Model complementarity:** LR captures linear relationships; TCN captures temporal patterns
2. **Calibration:** Probability outputs enable threshold-based fusion rules
3. **Flexibility:** Different fusion rules can optimise for different objectives (recall vs FAR)
4. **Interpretability:** Rule-based fusion (OR, Cascade) is clinically interpretable

---

## 3.8 Handling Class Imbalance

### Nature of Imbalance

The VitaStress dataset exhibits significant class imbalance:

| Class | Description | Proportion |
|-------|-------------|------------|
| Negative (0) | No stress / physical activity | ~88% |
| Positive (1) | Emotional stress onset predicted | ~12% |

**Imbalance ratio:** Approximately 1:7.3 (negative:positive)

### Why Accuracy is Misleading

With 88% negative samples, a trivial classifier predicting "no stress" achieves 88% accuracy while detecting **zero** stress events. Accuracy is therefore an inappropriate metric for this task.

### Techniques Used

#### Class Weighting (Training)

Both classical ML and TCN use **balanced class weights**:

```python
# Classical ML
class_weight = "balanced"  # Sklearn computes n_samples / (n_classes × n_samples_per_class)

# TCN: Focal Loss
alpha = 0.88  # Weight for positive class (approximately 1 - class proportion)
gamma = 2.0   # Focus on hard examples
```

#### Threshold Optimisation (Inference)

Three threshold selection strategies are evaluated (**Ablation B**):

| Strategy | Method | Clinical Rationale |
|----------|--------|-------------------|
| **B1** | Maximise G-mean = √(Recall × Specificity) | Balanced baseline |
| **B2** | Maximise G-mean s.t. Recall ≥ 70% | Prioritise catching stress events |
| **B3** | Maximise G-mean s.t. FAR ≤ 30% | Limit false alarms for user trust |

Thresholds are selected **per-fold on training data only** to prevent data leakage.

### What Was NOT Used

| Technique | Reason for Exclusion |
|-----------|---------------------|
| **SMOTE / Oversampling** | Creates synthetic time-series samples that may not preserve physiological realism; interpolation in feature space is problematic for temporal data |
| **GAN-based augmentation** | Requires substantial data for training; risk of generating unrealistic samples |
| **Undersampling** | Discards potentially valuable negative samples in an already limited dataset |

---

## 3.9 Evaluation Protocol

### 3.9.1 Subject-Independent Validation: Leave-One-Subject-Out (LOSO)

**LOSO cross-validation** is the primary evaluation protocol:

```python
For each subject s ∈ {1, 2, ..., 21}:
    Train set: All windows from subjects {1..21} \ {s}  # 20 subjects
    Test set:  All windows from subject s               # 1 subject
    
    1. Train model on 20 subjects
    2. Select threshold on TRAINING data (B1/B2/B3)
    3. Apply threshold to test subject predictions
    4. Record per-subject (fold) metrics

Aggregate: Concatenate fold-level predictions (NO re-thresholding)
```

### Why LOSO Reflects Real Deployment

LOSO answers: *"How well does the model predict stress in a person it has never seen?"*

This is the realistic deployment scenario:
- A new user downloads a stress prediction app
- The model has never seen this user's physiological data
- Performance must generalise to this unseen individual

Standard k-fold cross-validation would allow **within-subject leakage**, where the model sees windows from the same subject in both training and testing—unrealistic and overly optimistic.

### 3.9.2 No Train-Test Window Overlap

Within each LOSO fold:
- Training windows come from 20 subjects
- Test windows come from 1 held-out subject
- **No windows from the test subject appear in training**

The 50% overlap between windows is safe because it occurs **within** each subject's data, not across the train-test boundary.

### 3.9.3 Aggregation Protocol

**Critical:** Aggregation uses **actual fold-level predictions**, not re-thresholded probabilities:

```python
# ✅ CORRECT: Use predictions made with fold-specific thresholds
all_predictions = []
for fold in folds:
    y_pred = (y_proba >= fold_threshold).astype(int)  # Fold-specific threshold
    all_predictions.extend(y_pred)

aggregate_metrics = evaluate(all_predictions, all_labels)

# ❌ WRONG: Re-threshold with mean (statistically invalid)
# mean_threshold = np.mean([fold_thresholds])
# all_predictions = (all_proba >= mean_threshold)  # INVALID
```

Averaging thresholds from different class distributions is statistically meaningless. Each fold's threshold is optimised for its specific training set.

### 3.9.4 Evaluation Metrics

#### Primary Metrics (Threshold-Dependent)

| Metric | Definition | Clinical Relevance |
|--------|------------|-------------------|
| **Recall (Sensitivity)** | TP / (TP + FN) | What % of stress events are predicted? |
| **Specificity** | TN / (TN + FP) | What % of non-stress correctly identified? |
| **False Alarm Rate (FAR)** | FP / (FP + TN) = 1 - Specificity | What % of predictions are false alarms? |
| **G-mean** | √(Recall × Specificity) | Balanced performance at operating threshold |
| **Precision** | TP / (TP + FP) | What % of positive predictions are correct? |

#### Secondary Metrics (Threshold-Independent)

| Metric | Definition | Relevance |
|--------|------------|-----------|
| **AUROC** | Area under ROC curve | Overall discrimination ability (rank-based) |
| **PR-AUC** | Area under Precision-Recall curve | More informative than AUROC under class imbalance |

### 3.9.5 Clinical Relevance of Metrics

**Recall is primary** because missed stress events (false negatives) have higher cost than false alarms:
- A missed stress event = lost opportunity for intervention
- A false alarm = minor inconvenience (user dismisses notification)

However, **excessive false alarms erode user trust**, making FAR an important constraint.

The **G-mean** metric balances these competing concerns, while **constrained thresholds (B2, B3)** allow explicit control over recall-FAR trade-offs.

### 3.9.6 Statistical Analysis

To compare models rigorously:

1. **Paired Wilcoxon signed-rank test** across 21 LOSO folds
   - Non-parametric (no normality assumption)
   - Accounts for paired nature of comparisons

2. **Effect size: Cliff's delta**
   - Small: |d| < 0.147
   - Medium: 0.147 ≤ |d| < 0.33
   - Large: |d| ≥ 0.33

3. **Per-subject analysis**
   - Identify subjects where models struggle
   - Understand sources of inter-subject variance

---

## Summary: Experimental Design Overview

| Aspect | Specification |
|--------|---------------|
| **Task** | Binary near-future stress prediction (5-min horizon) |
| **Input** | 8-channel × 120s windows at 4 Hz (39 features for classical ML) |
| **Dataset** | VitaStress (21 subjects, lab-based protocol) |
| **Preprocessing** | Subject-wise z-score, HeartPy HR/HRV extraction at 64 Hz |
| **Validation** | Leave-One-Subject-Out (LOSO) |
| **Models** | LR, RF, SVM (classical); TCN (deep learning) |
| **Fusion** | Logical OR, Cascade, Stacked Generalisation |
| **Imbalance Handling** | Class weighting + threshold optimisation |
| **Threshold Strategies** | B1 (G-mean), B2 (Recall≥70%), B3 (FAR≤30%) |
| **Primary Metrics** | Recall, FAR, G-mean |
| **Secondary Metrics** | AUROC, PR-AUC |

This design enables systematic evaluation of:
- **RQ1:** Feasibility via LOSO validation metrics
- **RQ2:** TCN vs classical ML across threshold strategies
- **RQ3:** Fusion improvement over single models

---

## Appendix: Experimental Configuration

### Fixed Hyperparameters (No Tuning)

```python
# Classical ML
LR_PARAMS = {"C": 0.1, "penalty": "l2", "solver": "saga", "class_weight": "balanced"}
RF_PARAMS = {"n_estimators": 100, "max_depth": 5, "max_features": "log2", "class_weight": "balanced"}
SVM_PARAMS = {"C": 10.0, "kernel": "rbf", "gamma": 0.1, "class_weight": "balanced"}

# TCN
TCN_PARAMS = {
    "batch_size": 16, "lr": 1e-3, "epochs": 100, "dropout": 0.3,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128], "channels": [16] * 8, "patience": 15
}

# Pipeline
TARGET_HZ = 4.0
WINDOW_SIZE_SEC = 120
OVERLAP_RATIO = 0.5
LABEL_COL = "label_5min"
RANDOM_SEED = 42
```

### Threshold Strategies

```python
THRESHOLD_CONFIGS = {
    "B1": {"method": "gmean", "name": "Unconstrained G-Mean"},
    "B2": {"method": "recall", "min_recall": 0.70, "name": "Recall-Constrained (≥70%)"},
    "B3": {"method": "fpr", "max_fpr": 0.30, "name": "FAR-Constrained (≤30%)"}
}
```
