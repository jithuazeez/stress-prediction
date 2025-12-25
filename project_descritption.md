# Stress Prediction from VitaStress Wearable Data — Project Description

**Goal:** Build a system that predicts whether **emotional stress onset** will occur in the **next 5 minutes** from multimodal wearable physiological signals (accelerometer, temperature, heart rate, HRV) collected via the VitaStress dataset.

**Key Task:** Pure **prediction** (not detection) - predict future stress before it happens, enabling proactive intervention.

**Implementation Status:** ✅ Complete pipeline with 4 model architectures, LOSO validation, and constrained threshold optimization.

---

## 0. Quick Summary

* **Dataset:** VitaStress — 21 subjects with multimodal physiological signals from wearable sensors (accelerometer, temperature, PPG, heat flux)
* **Task:** Predict emotional stress onset 5 minutes in advance (pure prediction, not detection)
* **Input:** 8-channel time series (acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd) × 120 seconds
* **Models Implemented:** ✅ Classical ML (61 features), ✅ MOMENT (foundation model), ✅ SSL (self-supervised), ✅ Multi-Rate Fusion
* **Validation:** Leave-One-Subject-Out (LOSO) cross-validation with constrained G-mean threshold optimization
* **Status:** Complete pipeline, models trained, awaiting 8-channel retraining with RMSSD

---

## 1. Dataset: VitaStress

**VitaStress** is a real-world wearable dataset designed for stress monitoring research. The dataset contains multimodal physiological signals collected from participants in their daily lives using consumer-grade wearables.

### 1.1 VitaStress Overview

* **Signals used in final pipeline:**
  * **Accelerometer:** 3-axis motion (acc_x, acc_y, acc_z) at ~32Hz
  * **PPG:** Photoplethysmography at ~64Hz (for HR/HRV extraction)
  * **Heat Flux Sensor:** Skin temperature, heat flux, core body temperature at 1Hz
  * **Heart Rate (HR):** Extracted from PPG via HeartPy at native 64Hz
  * **Heart Rate Variability (RMSSD):** Parasympathetic activity indicator
  
* **Signals excluded (with rationale):**
  * **Raw PPG at 1Hz/8Hz:** Downsampling destroys cardiac waveform (use HR/HRV instead)
  * **EDA:** Too low sampling rate (~0.017Hz, 1 sample/minute, unreliable)
  * **Activity sensor HR:** High missingness (65%+), use HeartPy-extracted HR instead
  
* **Labels:** Button-press annotations marking stress event onsets
  * Cognitive stress (mental arithmetic, Stroop test)
  * Public speaking stress (social anxiety)
  * Physical stress (exercise) - NOT labeled as emotional stress
  * Baseline periods (confirmed rest)
  
* **Participants:** 21 subjects, ~60 minutes per subject
* **Collection context:** Lab-based protocol with controlled stress induction
* **Data format:** CSV files per subject per modality
* **Sampling rates:** ACC (~32Hz), PPG (~64Hz), Heatflux (1Hz), Annotations (event-based)

### 1.2 Dataset Strengths & Challenges

**Strengths:**
* Real-world data (high ecological validity)
* Multiple physiological modalities
* Extended collection periods per participant
* Suitable for predictive modeling (temporal density adequate for 5–10 min ahead prediction)

**Challenges:**
* Class imbalance (stress events may be rare)
* Missing data and sensor dropouts
* Inter-subject variability
* Possible motion artifacts

Deliverable: `vitastress_eda.ipynb` (completed), `vitastress_signal_processing.ipynb` (in progress).

---

## 2. Research Questions

This project investigates three core research questions, informed by the VitaStress EDA analysis:

### RQ1: Window Size Optimization
**How does changing the input window affect stress detection performance?**

* **Motivation:** The trade-off between detection accuracy and prediction latency is critical for early warning systems. Longer windows capture more contextual information but increase latency; shorter windows enable faster alerts but may lack sufficient signal.
* **Approach:** Systematically compare models trained with varying input window sizes: 30s, 60s, 90s, 120s, 150s, and 180s of prior physiological data.
* **Expected outcomes:** Studies suggest ≈90+ second windows often yield better accuracy, but this may vary by modality and individual. We will quantify the accuracy vs. latency trade-off specifically for VitaStress data.
* **Metrics:** AUROC, PR-AUC, recall@precision, and time-to-warning for each window configuration.

### RQ2: Feature Importance Analysis
**Which physiological signals and features best predict an imminent stress event?**

* **Motivation:** Understanding which signals are most predictive enables focused sensor selection, reduces computational cost, and improves model interpretability. EDA features are often the strongest stress indicators, while HR/HRV can be confounded by physical activity.
* **Approach:** 
  * Extract comprehensive features from each modality (HR/HRV, EDA, respiration, activity)
  * Use permutation importance, SHAP values, or attention weights to identify most predictive features
  * Conduct ablation studies (remove modality groups systematically)
* **Expected outcomes:** Identify the minimal set of features/sensors required for acceptable performance; understand activity-related confounds.
* **Deliverable:** Feature importance rankings and modality ablation results.

### RQ3: Model Architecture Comparison
**How do different model architectures perform on stress prediction from VitaStress data?**

* **Motivation:** Stress prediction from multivariate time-series can be approached with various architectures. Understanding which performs best on VitaStress data—and why—informs practical deployment decisions.
* **Approach:** Compare performance across:
  * **Baselines:** Logistic Regression, Random Forest, XGBoost (on aggregated features)
  * **Deep learning:** 1D CNN (local patterns), LSTM/GRU (temporal dependencies)
  * **Transformers:** Self-attention over multivariate sequences (long-range dependencies)
* **Architectures tested:** Binary classification (stress vs. no-stress) using encoder + classification head.
* **Expected outcomes:** Identify the architecture that best balances accuracy, computational cost, and interpretability for this task.
* **Deliverable:** Comparative performance table and computational cost analysis.

---

## 3. Labels & Ground Truth Strategy (IMPLEMENTED ✅)

### 3.1 Task Definition: Pure Prediction

**Primary Task:** Predict if emotional stress will START in the next 5 minutes

**Labeling Rule:**
```python
label = 1  if  window_end ≤ emotional_stress_onset ≤ window_end + horizon
label = 0  otherwise
```

**Key Design Choices:**
1. ✅ **Predict ONSET only** (not ongoing stress or detection)
   - Forces model to learn subtle pre-stress patterns
   - Prevents easy shortcuts (detecting obvious stress)
   - Clear research contribution (prediction > detection)
   
2. ✅ **Emotional stress only** (cognitive, public speaking)
   - Physical stress (exercise) is NOT labeled as stress
   - Prevents false positives from physical activity
   - Model learns: "High HR + sitting = stress" vs "High HR + running = exercise"
   
3. ✅ **Multiple horizons** (3, 5, 10 minutes)
   - Default: 5 minutes (optimal for intervention)
   - 3 min: Immediate warning
   - 10 min: Early detection

### 3.2 Event Types

**Emotional Stress Events (label = 1):**
- "Cognitive: Start" → Mental arithmetic, Stroop test, cognitive tasks
- "Public Speaking Start" → Social anxiety, performance stress

**Physical Stress Events (label = 0):**
- "Physical: Start" → Running on treadmill (NOT emotional stress)

**Baseline Events:**
- "Baseline: Start" → Confirmed rest periods (5 minutes)

### 3.3 Label Distribution

**Typical per subject:**
- Total windows: ~200-300 (2-minute windows over ~60-minute experiment)
- Positive windows (label=1): ~10-20 (3-5 minutes before each stress event)
- **Class imbalance: ~90% negative, ~10% positive**

**Handled via:**
- Weighted cross-entropy loss (9:1 weight ratio)
- Constrained G-mean threshold optimization
- NOT via SMOTE (inappropriate for time series)

### 3.4 Data Leakage Prevention

**No overlap:** overlap_ratio = 0.5
**Skip settling:** First 1 minutes excluded
**Future-only labeling:** Labels based on events AFTER window ends
**LOSO validation:** Complete subject separation (no subject in both train and test)

---

## 4. Feature Engineering & Windows (IMPLEMENTED ✅)

### 4.1 Multi-Channel Time Series (Deep Learning Models)

**8 Channels** (MOMENT, SSL, Multi-Rate):
1. **acc_x, acc_y, acc_z** (32Hz → aligned) - Directional movement information
2. **skin_temp** (1Hz) - Peripheral temperature
3. **heatflux** (1Hz) - Thermal energy transfer rate
4. **cbt** (1Hz) - Core body temperature estimate
5. **hr_bpm** (extracted at 64Hz via HeartPy → aligned to 1Hz/8Hz) - Heart rate
6. **rmssd** (extracted at 64Hz via HeartPy → aligned to 1Hz/8Hz) - Heart rate variability

**Window Configuration:**
- **MOMENT:** 120 seconds at 1Hz → resampled to 512 timesteps → Shape: [batch, 8, 512]
- **SSL:** 120 seconds at 8Hz → 960 timesteps → Shape: [batch, 8, 960]
- **Multi-Rate:** Native rates (PPG: 64Hz, ACC: 32Hz, Temp: 1Hz)

**Preprocessing:**
- **Normalization:** Subject-wise z-score (per channel, across subject's entire data)
- **Missing data:** Linear interpolation for short gaps, NaN for longer gaps
- **Outlier handling:** Physiologically plausible ranges enforced

### 4.2 Statistical Features (Classical ML)

**61 Features** extracted per 120-second window:

**Accelerometer (41 features):**
- Time domain: mean, std, min, max, range, SMA, energy, ZCR
- Frequency domain: dominant frequency, spectral entropy, band powers
- Per-axis features: acc_x, acc_y, acc_z statistics

**Temperature/Heat Flux (12 features):**
- Skin temp: mean, std, min, max, range, slope, change
- Heat flux: mean, std, range
- CBT: mean, change
- Pulse rate (from heat flux sensor)


**Note:** No PPG features at 1Hz (waveform destroyed). Use HR/HRV from HeartPy instead.

### 4.3 Signal Alignment Pipeline

**Multi-Rate Signals → Common Grid:**

```
ACC (32Hz)     ──┐
PPG (64Hz)     ──┤
                 ├─→ HeartPy → HR/HRV (1Hz) ──┐
Heatflux (1Hz) ──┘                            ├─→ Aligned Grid (1Hz or 8Hz)
                                              │
                                              └─→ Windowing → Labeled Samples
```

**Process:**
1. Extract HR/HRV from PPG at native 64Hz (HeartPy)
2. Align all signals to target rate (1Hz for MOMENT, 8Hz for SSL)
3. Create 120-second windows with prediction horizon labels
4. Apply subject-wise normalization

Deliverables: `experiments/shared/alignment.py`, `src/features/hrv_extractor.py`

---

## 5. Data Augmentation (if needed)

**Use sparingly and only if class imbalance/data scarcity is severe.**

* **Signal-level augmentations:** jitter, time-warping, window cropping, noise injection within physiologically plausible bounds.
* **Considerations:** Validate that augmentations preserve physiological realism (e.g., HR–HRV relationships, EDA refractory periods).
* **Primary focus:** Train on real data first; augmentation as secondary strategy if performance is limited by sample size.

Deliverables: `src/augment/` (if implemented), augmentation ablation in evaluation report.

---

## 6. Modelling Plan (Addresses RQ3)

This section directly addresses **RQ3: Model Architecture Comparison**. We will implement and compare multiple model families to identify the best-performing architecture for VitaStress stress prediction.

### 6.1 Baselines (Classical ML)

* **Logistic Regression** on aggregate features (mean, std, min, max over input window).
* **Random Forest** (ensemble of decision trees, handles feature interactions).
* **XGBoost/LightGBM** (gradient boosting, strong baseline, provides feature importance for RQ2).

**Rationale:** Establish performance floor; provides interpretable feature importance; computationally efficient.

### 6.2 Deep Learning Sequence Models

* **1D CNN / TCN (Temporal Convolutional Network):**
  * Captures local temporal patterns and motifs in physiological signals
  * Fast training and inference
  * Good for short-to-medium range dependencies
  
* **LSTM/GRU (Recurrent Neural Networks):**
  * Explicitly models temporal dependencies
  * Suitable for moderate-length windows (up to ~180s)
  * Captures sequential dynamics in stress progression
  
* **Transformers (Self-Attention):**
  * Attention mechanism learns which time steps and features are most relevant
  * Handles long-range dependencies naturally
  * **Architecture:** Encoder + classification head (binary stress prediction)
  * **Modality fusion:** Multi-channel input (each physiological signal as a channel) or channel-as-token design
  * **Interpretability:** Attention weights provide insights for RQ2 (feature importance)

**Rationale:** Compare inductive biases (CNNs: locality, RNNs: sequentiality, Transformers: global attention).

### 6.3 Personalisation (Optional Extension)

* **Subject-specific fine-tuning:** Global model pre-trained on all subjects, fine-tuned on per-subject data.
* **Transfer learning:** Evaluate how well models generalize to new subjects (leave-one-subject-out validation).

### 6.4 Training Strategy

* **Loss function:** Binary cross-entropy with class weights (handle imbalance) or focal loss.
* **Optimization:** Adam optimizer, learning rate scheduling.
* **Regularization:** Dropout, early stopping on validation set.
* **Hyperparameter tuning:** Grid or Bayesian search for each architecture.

Deliverables: `src/models/`, `src/train.py`, `configs/*.yaml`, `notebooks/model_comparison.ipynb`.

---

## 7. Evaluation & Metrics (IMPLEMENTED ✅)

### 7.1 Performance Metrics

**Primary Metrics:**
* **AUROC** - Threshold-independent discrimination (overall model quality)
* **G-mean** - Geometric mean of recall and specificity (balanced performance)
  - Used for model selection (not AUROC)
  - Better reflects performance at operating threshold
* **Recall/Sensitivity** - What % of stress events are predicted (target: ≥85%)
* **Specificity** - What % of non-stress correctly identified (target: ≥75%)
* **Precision** - What % of predictions are correct (low due to imbalance)
* **F1-score** - Harmonic mean of precision and recall
* **Balanced Accuracy** - (Recall + Specificity) / 2

**Threshold-Related Metrics:**
* **Per-fold threshold** - Optimal threshold for each subject (0.28-0.35 range)
* **Threshold distribution** - Mean ± std, min-max across folds
* **NOT mean threshold** - Each fold uses its own optimal threshold

**Reported Per-Fold and Aggregate:**
- Fold metrics: Per-subject performance with fold-specific threshold
- Aggregate metrics: Overall performance using actual fold predictions
- Consistency check: aggregate ≈ average(fold_metrics)

### 7.2 Threshold Optimization: Constrained G-Mean

**Method:** Maximize G-mean subject to clinical constraints

**Algorithm:**
```python
1. Sweep thresholds from 0 to 1
2. Filter candidates where:
   - Recall ≥ 0.85 (catch ≥85% of stress events)
   - FPR ≤ 0.20 (acceptable false alarm rate)
3. Among valid candidates, maximize:
   G-mean = √(recall × specificity)
4. If no candidates meet constraints:
   Fall back to unconstrained G-mean
```

**Applied per-fold on TRAINING data:**
- Prevents data leakage (never uses test data)
- Adapts to each subject's training set
- Ensures clinical requirements are met

### 7.3 Validation Strategy (LOSO)

**Leave-One-Subject-Out Cross-Validation:**

```python
for test_subject in all_21_subjects:
    # Train on 20 subjects
    train_subjects = all_subjects - {test_subject}
    model = train_model(train_subjects)
    
    # Find threshold on TRAINING data
    train_proba = model.predict(train_subjects)
    threshold = find_optimal_threshold(train_proba, train_labels)
    
    # Test on held-out subject
    test_proba = model.predict(test_subject)
    test_pred = (test_proba >= threshold).astype(int)
    
    # Store fold results
    fold_metrics.append(compute_metrics(test_pred, test_labels))

# Aggregate using fold-level predictions (no re-thresholding)
aggregate_metrics = compute_metrics(
    all_fold_predictions,  # Concatenated predictions
    all_fold_labels
)
```

**Why LOSO:**
- Tests generalization to new subjects (most realistic)
- Each subject is completely unseen during their fold
- No data leakage between subjects
- Standard practice for wearable physiological monitoring

**Advantages over k-fold:**
- Simulates real deployment (new user, no personal data)
- 21 independent evaluations (one per subject)
- Clear interpretation (performance on unseen subjects)

### 7.4 Aggregation Strategy (CRITICAL)

**Correct Approach (Implemented):**
```python
# ✅ Store predictions made with fold-specific thresholds
for fold in folds:
    y_pred = (y_proba >= fold_threshold).astype(int)
    all_predictions.extend(y_pred)

# ✅ Aggregate uses actual predictions
aggregate_metrics = evaluate(all_predictions, all_labels)
```

**Wrong Approach (Avoided):**
```python
# ❌ Re-threshold with mean (statistically invalid!)
mean_threshold = np.mean([fold_thresholds])
all_predictions = (all_proba >= mean_threshold)
```

**Why This Matters:**
- Averaging thresholds from different distributions is meaningless
- Re-thresholding can violate constraints (recall, FPR)
- Aggregate must reflect what actually happened
- Scientific integrity: report actual performance, not re-computed

### 7.3 Experimental Design (Addressing Research Questions)

* **RQ1 experiments (Window size):**
  * Train separate models for each window size (30s, 60s, 90s, 120s, 150s, 180s)
  * Compare performance metrics across window sizes
  * Plot accuracy vs. latency trade-off curves
  
* **RQ2 experiments (Feature importance):**
  * Permutation importance on best-performing model
  * SHAP values for tree-based models
  * Attention weight analysis for Transformer models
  * Ablation studies: systematically remove modality groups (e.g., train without EDA, without HR/HRV, etc.)
  
* **RQ3 experiments (Model comparison):**
  * Train all model architectures with same data preprocessing and features
  * Compare performance, training time, inference speed, and model size
  * Statistical significance tests (e.g., McNemar's test, paired t-test on AUROC)

Deliverables: `reports/evaluation_results.md`, `reports/figures/`, `notebooks/evaluation_analysis.ipynb`.

---

## 8. System Design Notes

* **Data loader:** VitaStress-specific loader with time-alignment, resampling, masking for missing channels; unified schema (`subject_id, timestamp, features[], label`).
* **Config-first approach:** Use YAML configs to specify:
  * Window size (for RQ1 experiments)
  * Feature groups (for RQ2 ablations)
  * Model architecture (for RQ3 comparisons)
  * Training hyperparameters
* **Reproducibility:** 
  * Deterministic random seeds
  * Version control for code (git)
  * Experiment tracking (MLflow, Weights & Biases, or simple JSON logs)
  * Saved model checkpoints and training curves
* **Modularity:** Clean separation between data loading, feature extraction, model definition, training, and evaluation.

---

## 9. Repository Layout (Current Structure)

```
.
├─ Datasets/
│  └─ VitaStress/
│     └─ data/                          # 21 subjects with multimodal signals
│        └─ preprocessed/
│           └─ hr_data/                 # HR/HRV extracted from PPG
│
├─ src/                                 # Core pipeline utilities
│  ├─ data/
│  │  ├─ vitastress_loader.py          # Data loading
│  │  ├─ window_creator.py             # Windowing logic
│  │  └─ time_aligner.py               # Multi-rate alignment
│  ├─ features/
│  │  ├─ hrv_extractor.py              # HeartPy-based HR/HRV extraction
│  │  ├─ activity_features.py          # Activity/motion features
│  │  └─ feature_extractor.py          # Master feature pipeline (39 features)
│  └─ pipeline.py                      # End-to-end processing pipeline
│
├─ experiments/
│  ├─ shared/                           # Shared utilities across experiments
│  │  ├─ config.py                     # Common configuration
│  │  ├─ evaluation.py                 # Metrics and threshold optimization
│  │  ├─ alignment.py                  # Signal alignment to 1Hz
│  │  ├─ windowing.py                  # Window creation and labeling
│  │  └─ raw_loader.py                 # Raw signal loading
│  │
│  ├─ 01_classical_ml/                 # Classical ML baselines ✅
│  │  ├─ train.py                      # LOSO training with 61 features
│  │  ├─ feature_extraction.py         # Statistical feature extraction
│  │  └─ results/                      # Model outputs and metrics
│  │
│  ├─ 02_moment/                        # MOMENT foundation model ✅
│  │  ├─ train.py                      # Fine-tuning script
│  │  ├─ dataset.py                    # 8-channel dataset (1Hz → 512 samples)
│  │  ├─ model.py                      # Model creation and utilities
│  │  └─ results/
│  │     ├─ checkpoints/               # Per-fold and best models
│  │     ├─ config/                    # Training configuration
│  │     └─ moment_fold_metrics.csv    # Per-fold performance
│  │
│  ├─ 05_subject_aware_ssl/            # Self-supervised learning ✅
│  │  ├─ pretrain.py                   # Contrastive pre-training
│  │  ├─ train.py                      # Fine-tuning script
│  │  ├─ dataset.py                    # 8-channel dataset (8Hz, 960 samples)
│  │  ├─ config.py                     # SSL configuration
│  │  ├─ model.py                      # Encoder and classification head
│  │  └─ results/                      # Model checkpoints and metrics
│  │
│  ├─ 06_multirate_fusion/             # Multi-rate fusion ✅
│  │  ├─ train.py                      # Training script
│  │  ├─ model.py                      # Late fusion architecture
│  │  ├─ encoders.py                   # PPG, ACC, Temp encoders
│  │  └─ results/                      # Model outputs
│  │
│  ├─ AGGREGATION_FIX.md               # Technical documentation
│  ├─ THRESHOLD_METHOD_UPDATE.md       # Threshold optimization
│  ├─ LOSO_MODEL_SAVING.md             # Model checkpointing
│  ├─ HRV_RMSSD_ADDITION.md            # RMSSD integration
│  └─ WINDOWING_LABELING_PIPELINE.md   # Labeling strategy
│
├─ notebooks/
│  ├─ vitastress_eda.ipynb             # Dataset exploration
│  └─ data_quality_analysis.ipynb      # Signal quality assessment
│
├─ papers/                              # Paper summaries and notes
│  ├─ vitastress_paper.md
│  └─ [other paper summaries]
│
├─ reports/                             # Analysis reports
│  └─ heatflux_quality_analysis.csv
│
├─ project_descritption.md              # This file
└─ README.md
```

---

## 10. Project Milestones

### ✅ **M1 — Data Pipeline & EDA (Completed)**

* ✅ VitaStress data loading and exploration
* ✅ Signal quality analysis across all modalities
* ✅ HR/HRV extraction pipeline using HeartPy (gold standard)
* ✅ Multi-rate signal alignment (1Hz and 8Hz grids)
* ✅ Windowing and labeling pipeline
* ✅ Subject-wise normalization strategy
* **Deliverables:** `notebooks/`, `src/features/hrv_extractor.py`, `experiments/shared/alignment.py`

### ✅ **M2 — Classical ML Baselines (Completed)**

* ✅ Feature extraction: 61 statistical features from 8-channel input
* ✅ Implemented: Logistic Regression, Random Forest, XGBoost
* ✅ LOSO cross-validation with constrained G-mean thresholding
* ✅ Proper aggregation strategy (no re-thresholding)
* **Deliverables:** `experiments/01_classical_ml/`

### ✅ **M3 — MOMENT Foundation Model (Completed)**

* ✅ Fine-tuning MOMENT pre-trained model for stress prediction
* ✅ 8-channel × 512-timestep input (resampled from 120s)
* ✅ Selective backbone unfreezing (last 2 blocks)
* ✅ Discriminative learning rates (backbone vs head)
* ✅ G-mean based model selection
* ✅ Compact logging (prevents Kaggle/Colab crashes)
* ✅ Epoch-level logging (every 50 epochs)
* **Deliverables:** `experiments/02_moment/`
* **Performance:** AUROC ~0.80, G-mean ~0.78, Recall ~0.87

### ✅ **M4 — Self-Supervised Learning (Completed)**

* ✅ Subject-aware contrastive pre-training at 8Hz
* ✅ InfoNCE loss + adversarial subject-invariance
* ✅ Multiple threshold method comparison per fold
* ✅ Fine-tuning with classification head
* **Deliverables:** `experiments/05_subject_aware_ssl/`

### ✅ **M5 — Multi-Rate Late Fusion (Completed)**

* ✅ Modality-specific encoders at native rates
* ✅ PPG (64Hz), ACC (32Hz), Temp (1Hz)
* ✅ Late fusion architecture with attention
* **Deliverables:** `experiments/06_multirate_fusion/`

### 🔄 **M6 — Model Optimization (In Progress)**

* ✅ Added RMSSD as 8th channel (awaiting retraining)
* ✅ Fixed aggregation strategy across all experiments
* ✅ Implemented constrained G-mean thresholding
* ⏳ Retrain all models with 8 channels
* ⏳ Hyperparameter optimization
* ⏳ Window size experiments (RQ1)

### ⏳ **M7 — Research Questions & Analysis (Planned)**

* **RQ1:** Window size comparison (60s, 120s, 180s)
* **RQ2:** Feature importance and ablation studies
* **RQ3:** Comprehensive model comparison with statistical tests
* **Deliverables:** Performance tables, plots, statistical analysis

### ⏳ **M8 — Dissertation Writing (Planned)**

* Synthesize findings across all experiments
* Generate publication-quality figures
* Document limitations and future work
* **Deliverable:** Complete dissertation chapters

---

## 11. Risks & Mitigations (IMPLEMENTED ✅)

* **Class imbalance (stress ~10%):** ✅ **MITIGATED**
  * ✅ Weighted cross-entropy loss (9:1 ratio)
  * ✅ Constrained G-mean threshold optimization
  * ✅ PR-AUC and G-mean metrics (not just accuracy)
  * ❌ NOT using SMOTE (inappropriate for time series)
  
* **Inter-subject variability:** ✅ **MITIGATED**
  * ✅ Subject-wise z-score normalization (per channel)
  * ✅ LOSO cross-validation (tests generalization to new subjects)
  * ✅ Per-fold threshold adaptation (optimal for each subject's training set)
  
* **Missing data / sensor dropouts:** ✅ **HANDLED**
  * ✅ Linear interpolation for short gaps
  * ✅ Quality checks (skip windows with <50% data)
  * ✅ HeartPy rejects artifacts (>50% rejection rate → skip window)
  * ✅ NaN handling in features and alignment
  
* **Activity confounds (HR/HRV affected by movement):** ✅ **ADDRESSED**
  * ✅ Physical stress (exercise) NOT labeled as emotional stress
  * ✅ Accelerometer data included (model learns to use movement context)
  * ✅ Model learns: "High HR + sitting = stress" vs "High HR + running = exercise"
  
* **False alarms (user trust):** ✅ **CONTROLLED**
  * ✅ FPR constrained to ≤20% during threshold optimization
  * ✅ Precision reported alongside recall
  * ✅ Per-subject thresholds adapt to individual baselines
  * ⏳ Future: False alarm rate per hour calculation
  
* **Computational limits:** ✅ **MANAGED**
  * ✅ MOMENT at 1Hz (512 timesteps, manageable)
  * ✅ SSL at 8Hz (960 timesteps, efficient)
  * ✅ Selective unfreezing (only 7.5% of params trainable)
  * ✅ Compact logging (prevents Kaggle/Colab crashes)
  * ✅ GPU-efficient batch sizes (16-32)

---

## 12. Comparison with VitaStress Paper & Related Work

### VitaStress Baseline Paper
**Their Approach:**
- Task: **Detection** of current stress state
- Methods: SVM, Random Forest on statistical features
- Validation: 10-fold cross-validation
- Best: Random Forest with 74% accuracy, 69% F1-score

**Our Approach:**
- Task: **Prediction** of future stress onset (5 min ahead)
- Methods: Classical ML + MOMENT + SSL + Multi-Rate Fusion
- Validation: LOSO cross-validation (more rigorous)
- Input: 8-channel time series (they used aggregated features)
- Added: RMSSD/HRV, separate emotional vs physical stress

**Key Differences:**
- ✅ Prediction (harder) vs detection (easier)
- ✅ LOSO (tests new subjects) vs k-fold
- ✅ Raw time series + foundation models vs hand-crafted features only
- ✅ Constrained threshold optimization vs default thresholds
- ✅ Proper aggregation strategy vs simple averaging

### Novel Contributions Beyond Prior Work

1. **Foundation Model Application:** First use of MOMENT for stress prediction
2. **Subject-Aware SSL:** Adversarial subject-invariance for wearable data
3. **8-Channel Multimodal:** Includes RMSSD (novel for stress prediction)
4. **Constrained Optimization:** Clinical requirements (recall ≥ 85%) built into threshold selection
5. **Pure Prediction Focus:** Separate emotional from physical, predict future only

---

## 13. Non-Goals (Current Scope)

* ❌ Clinical diagnosis or treatment recommendations (research prototype only)
* ❌ Real-time on-device deployment (offline analysis first)
* ❌ Multi-dataset fusion (VitaStress only)
* ❌ Complex intervention systems (focus on prediction model)
* ❌ Detection model (only prediction, not mixed tasks)
* ❌ Personalized adaptation (global models with LOSO validation)

---

## 13. Current Progress Summary (As of December 2024)

### ✅ Completed Work

**Data Loading & EDA (`vitastress_draft.ipynb`):**
* ✅ All 21 subjects loaded for accelerometer, activity, and bioimpedance modalities
* ✅ Comprehensive missing data analysis with zero-as-missing correction for physiological features
* ✅ Statistical summaries and distribution visualizations for all modalities
* ✅ Sampling rate estimation: Acc (~32 Hz), Activity (~30s), Bioz (~25 Hz)
* ✅ Data quality assessment with quality flags (`bpm_q`, `resp_q`, `spo2_q`, `wearing`)
* ✅ Saved combined datasets: `vitastress_acc_combined.csv`, `vitastress_activity_combined.csv`, `vitastress_bioz_combined.csv`

**Feature Extraction:**
* ✅ Accelerometer features extracted using TSFEL (120s windows, no overlap)
* ✅ Time + frequency domain features: mean, std, skewness, kurtosis, RMS, SMA, autocorrelation, PSD, spectral entropy, etc.
* ✅ Added temporal metadata: `window_start_time`, `window_end_time`, `window_center_time`
* ✅ Saved: `vitastress_acc_features.csv` (~3.4M samples → ~880 feature windows)

**Infrastructure (`src/` modules):**
* ✅ `src/data/vitastress_loader.py` — Modular data loader for all VitaStress modalities
* ✅ `src/data/preprocessing.py` — Preprocessing utilities (normalization, artifact handling)
* ✅ `src/features/` — Feature extractors for HR, HRV, EDA, respiration, activity, temperature, PPG
* ✅ `src/models/baselines.py`, `rnn.py`, `transformer.py` — Model architecture skeletons
* ✅ `src/evaluate.py` — Evaluation metrics and plotting functions
* ✅ `src/utils.py` — Helper utilities

### 🔍 Key Data Quality Findings

**Critical Issues Identified:**
1. **Zeros as Missing Data:** HR, RR, SpO2, BP, and Bioimpedance contain physiologically impossible zeros (60-95% missingness)
2. **Bioimpedance Outliers:** Extreme values up to 10,000,000 Ω detected (normal range: 500-2000 Ω)
   - **Action Required:** Filter to 100-5000 Ω before feature extraction
3. **High Missingness in Activity Data:** Most physiological features have 65-95% missing/invalid data
   - Implies heavy reliance on accelerometer, PPG-derived features, and EDA/emography

**Usable Modalities (Priority Order):**
1. ✅ **Accelerometer:** Excellent quality, 0% missing, ~32 Hz
2. 🔄 **PPG Signals:** Available but not yet processed (HRV extraction needed)
3. 🔄 **EDA/Emography:** Available but not yet processed (critical for stress detection)
4. ⚠️ **Activity-derived HR/RR:** High missingness (use as supplementary features only)
5. ⚠️ **Bioimpedance:** Requires aggressive outlier filtering (50% zeros + extreme outliers)

### 🔄 In Progress

* **Bioimpedance outlier removal:** Filtering to physiologically plausible range (100-5000 Ω)
* **PPG processing:** HRV feature extraction from `rr_interval` and PPG signals
* **EDA feature extraction:** Tonic/phasic decomposition, SCR counting from `emography` files

### ✅ Major Completed Work (December 2024)

**Complete Pipeline Implementation:**
* ✅ **HR/HRV Extraction:** HeartPy-based pipeline extracts HR and RMSSD from PPG at native 64Hz
* ✅ **Signal Alignment:** All signals aligned to common 1Hz grid for MOMENT (8Hz for SSL)
* ✅ **Windowing & Labeling:** 120-second windows with 3/5/10-minute prediction horizons
  - Labels predict FUTURE stress onset (not current state)
  - Separates emotional stress (cognitive, public speaking) from physical stress (exercise)
  - Prevents data leakage by only labeling pre-stress windows
* ✅ **8-Channel Input:** acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd

**Model Implementations:**
* ✅ **Classical ML (`experiments/01_classical_ml/`):** 
  - Logistic Regression, Random Forest, XGBoost with 61 statistical features
  - LOSO cross-validation with constrained G-mean thresholding
  
* ✅ **MOMENT Foundation Model (`experiments/02_moment/`):**
  - Pre-trained time-series foundation model fine-tuned for stress prediction
  - 8 channels × 512 timesteps input (resampled from 120s at 1Hz)
  - Selective backbone unfreezing (last 2 blocks)
  
* ✅ **Self-Supervised Learning (`experiments/05_subject_aware_ssl/`):**
  - Subject-aware contrastive pre-training at 8Hz
  - InfoNCE loss + adversarial subject-invariance
  - Fine-tuning with classification head
  
* ✅ **Multi-Rate Late Fusion (`experiments/06_multirate_fusion/`):**
  - PPG at 64Hz, ACC at 32Hz, Temp at 1Hz (native rates)
  - Modality-specific encoders with late fusion

**Advanced Training Features:**
* ✅ **Threshold Optimization:** Constrained G-mean method (recall ≥ 85%, FPR ≤ 20%)
* ✅ **Class Imbalance Handling:** Weighted cross-entropy + threshold optimization
* ✅ **LOSO Cross-Validation:** Leave-One-Subject-Out for generalization testing
* ✅ **Proper Aggregation:** Fold-level predictions (no re-thresholding)
* ✅ **Model Checkpointing:** Best model saved based on G-mean metric

### 📊 Current Experimental Setup

**Data Configuration:**
```
Input: 8 channels (acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd)
Window: 120 seconds (2 minutes)
Overlap: 0% (no overlap to prevent leakage)
Horizons: 3, 5, 10 minutes (default: 5 min)
Sampling: 1Hz for MOMENT/Classical ML, 8Hz for SSL
Skip: First 5 minutes (sensor settling)
```

**Labeling Strategy:**
```python
# Pure prediction approach (no detection)
label = 1 if emotional_stress_onset in [window_end, window_end + horizon]
label = 0 otherwise

# Key decisions:
- Only EMOTIONAL stress labeled (cognitive, public speaking)
- Physical stress (exercise) NOT labeled as stress
- Only FUTURE onset labeled (not current/ongoing stress)
- Prevents model from learning easy detection patterns
- Forces learning of subtle pre-stress signals
```

**Evaluation Strategy:**
```python
# Per-fold:
1. Find optimal threshold on TRAINING data (constrained G-mean)
2. Apply to TEST data (held-out subject)
3. Store predictions with fold-specific threshold

# Aggregate:
1. Concatenate all fold-level predictions
2. Compute metrics on actual predictions (no re-thresholding)
3. Report threshold distribution (mean ± std, min-max)
```

### 🔧 Key Technical Improvements

**1. Aggregation Fix:**
- ❌ Before: Re-applied mean threshold to all predictions (incorrect)
- ✅ After: Use actual fold-level predictions (correct)
- Impact: Aggregate metrics now match average of fold metrics

**2. Threshold Reporting:**
- ❌ Before: Reported single "mean threshold" (misleading)
- ✅ After: Report threshold distribution (min, max, mean ± std)
- Impact: Honest reporting of per-fold adaptation

**3. LOSO Model Saving:**
- ❌ Before: Saved untrained "final model" with random weights
- ✅ After: Best fold model saved, no misleading "final" model
- Impact: Only trained, usable models are saved

**4. HRV Integration:**
- ❌ Before: 7 channels (no HRV)
- ✅ After: 8 channels (added RMSSD from HeartPy)
- Impact: Additional autonomic activity information for stress detection

### 📈 Current Results (MOMENT with 7 channels, needs retraining with 8)

```
Aggregate Performance (LOSO):
- AUROC: 0.78-0.82 (good discrimination)
- G-mean: 0.75-0.80 (balanced performance)
- Recall: 0.85-0.90 (high sensitivity, as targeted)
- Precision: 0.20-0.30 (low due to severe imbalance ~10% stress)
- Specificity: 0.70-0.75 (acceptable false alarm rate)

Class Imbalance: ~10% stress windows, ~90% no-stress windows
Threshold Range: 0.28-0.35 (adapts per subject)
```

### ⏳ Immediate Next Actions

**Phase 3A — Model Retraining with 8 Channels:**
* [ ] Retrain MOMENT with 8 channels (including RMSSD)
* [ ] Retrain SSL with 8 channels (hr_bpm + rmssd instead of ppg_mean)
* [ ] Compare 7-channel vs 8-channel performance
* [ ] Expected: +2-5% improvement in recall/G-mean

**Phase 3B — Hyperparameter Optimization:**
* [ ] Grid search for optimal learning rates
* [ ] Test different window sizes (60s, 120s, 180s) for RQ1
* [ ] Experiment with different horizons (3 min vs 5 min vs 10 min)
* [ ] Ablation studies for feature importance (RQ2)

**Phase 3C — Model Comparison & Analysis:**
* [ ] Statistical significance tests between models
* [ ] Computational cost analysis (training time, inference speed)
* [ ] Attention weight analysis for interpretability
* [ ] Per-subject performance analysis (identify difficult subjects)

---

## 14. Implemented Models & Current Results

### 14.1 Model Architectures

#### Classical ML (`experiments/01_classical_ml/`)
**Architecture:**
```
61 Statistical Features → StandardScaler → Classifier
Features: ACC (41), Temp/Heatflux (12), EDA (8)
Models: Logistic Regression, Random Forest, XGBoost
```

**Training:**
- LOSO cross-validation (21 folds)
- Constrained G-mean threshold per fold
- Class weights for imbalance

#### MOMENT Foundation Model (`experiments/02_moment/`)
**Architecture:**
```
8 Channels × 512 Timesteps → MOMENT Encoder (frozen) → Unfreeze Last 2 Blocks → Classification Head → 2 Classes
341M total params, 25.7M trainable (7.5%)
```

**Training:**
- Fine-tuning with selective unfreezing
- Weighted cross-entropy loss (9:1 ratio)
- Discriminative learning rates (0.1× for backbone, 1× for head)
- Early stopping based on training loss
- 50 epochs, batch size 16

**Current Results (7 channels, needs retraining with 8):**
```
AUROC:     0.80 ± 0.12
G-mean:    0.78 ± 0.15
Recall:    0.87 ± 0.18 (meets 85% target)
Precision: 0.25 ± 0.15 (low due to imbalance)
Threshold: 0.31 ± 0.02 (range: 0.28-0.35)
```

#### Self-Supervised Learning (`experiments/05_subject_aware_ssl/`)
**Architecture:**
```
Pre-training: 8 Channels × 960 Samples (8Hz) → ResNet1D Encoder → InfoNCE + Adversarial Loss
Fine-tuning: Frozen Encoder → Classification Head → 2 Classes
```

**Pre-training:**
- Subject-aware contrastive learning
- InfoNCE loss (temperature=0.07)
- Adversarial subject classifier (λ=0.5)
- Embedding dim: 128

**Fine-tuning:**
- Per-fold fine-tuning with frozen encoder
- 5 threshold methods compared per fold
- Best method selected based on G-mean

#### Multi-Rate Late Fusion (`experiments/06_multirate_fusion/`)
**Architecture:**
```
PPG (64Hz × 480 samples) → PPGEncoder → 128-d
ACC (32Hz × 240 samples) → ACCEncoder → 128-d  →  Concat → Fusion → Classifier
Temp (1Hz × 60 samples)  → TempEncoder → 64-d
```

**Training:**
- Each modality processed at native rate
- Late fusion combines embeddings
- Preserves temporal resolution per modality

### 14.2 Training Configuration Summary

| Model | Input Shape | Sampling Rate | Training Time | Best Fold G-mean |
|-------|-------------|---------------|---------------|------------------|
| Classical ML | [61 features] | 1Hz → stats | ~5 min/fold | ~0.70 |
| MOMENT | [8, 512] | 1Hz → 512 | ~15 min/fold | ~0.78 |
| SSL | [8, 960] | 8Hz | ~20 min/fold | ~0.75 |
| Multi-Rate | Native rates | Mixed | ~18 min/fold | ~0.76 |

*Note: Results are from 7-channel versions, awaiting 8-channel retraining*

---

## 15. Technical Architecture & Key Decisions

### 14.1 Labeling Strategy: Pure Prediction (Not Detection)

**Task Definition:** "Will emotional stress START in the next 5 minutes?"

**Design Decision:** Label ONLY future stress onset, NOT ongoing stress
```python
# Current approach:
label = 1 if (window_end ≤ stress_onset ≤ window_end + horizon)
label = 0 if stress already started OR too far away

# Rationale:
✅ Clear research question (prediction, not detection)
✅ Forces model to learn subtle pre-stress patterns
✅ Prevents easy shortcut (just detecting obvious stress)
✅ Novel contribution (fewer papers on prediction)
✅ Clinical value (enables preventive intervention)
```

**Why NOT label ongoing stress:**
- Would mix two tasks (prediction + detection)
- Model would learn easy detection, ignore hard prediction
- Evaluation metrics would be ambiguous
- Less suitable for dissertation (unclear contribution)

### 14.2 Emotional vs Physical Stress Separation

**Critical Design Choice:** Only EMOTIONAL stress is labeled as stress (1)

```python
# EMOTIONAL stress (label = 1):
- "Cognitive: Start" → Mental arithmetic, Stroop test
- "Public Speaking Start" → Social anxiety, performance stress

# PHYSICAL stress (label = 0):
- "Physical: Start" → Running, exercise
```

**Rationale:**
- Exercise causes similar physiological response (↑HR, ↑temp)
- But exercise is NOT emotional stress
- Model learns to distinguish: "High HR + sitting = stress" vs "High HR + running = exercise"
- Prevents false positives from physical activity

### 14.3 Multi-Channel Input Architecture

**8 Channels:**
1. `acc_x`, `acc_y`, `acc_z` - Separate accelerometer axes (directional movement)
2. `skin_temp` - Skin temperature (1Hz from heat flux sensor)
3. `heatflux` - Thermal energy transfer rate (1Hz)
4. `cbt` - Core body temperature (1Hz)
5. `hr_bpm` - Heart rate from HeartPy (extracted at 64Hz, aligned to 1Hz)
6. `rmssd` - HRV (Root Mean Square of Successive Differences, parasympathetic activity)

**Design Rationale:**
- Separate acc axes (not magnitude): Preserves directional information
- HR from HeartPy (not raw PPG): Downsampling PPG destroys cardiac waveform
- RMSSD addition: Highly sensitive to stress, captures autonomic balance
- No raw PPG: Meaningless at 1Hz or 8Hz (need native 64Hz for waveform)
- No EDA: Too low sampling rate (~0.017 Hz, 1 sample/minute)

### 14.4 Threshold Optimization Strategy

**Problem:** With severe class imbalance (10% stress), default threshold (0.5) gives poor recall.

**Solution:** Constrained G-Mean optimization
```python
# Algorithm:
1. Sweep thresholds in [0, 1]
2. Keep only thresholds where:
   - Recall ≥ 0.85 (clinical requirement: catch 85%+ of stress)
   - FPR ≤ 0.20 (acceptable false alarm rate)
3. Among valid candidates, maximize G-Mean = √(recall × specificity)
4. If no candidates satisfy constraints, fall back to unconstrained G-mean

# Applied per-fold on TRAINING data (no leakage)
```

**Why This Works:**
- Ensures clinical requirements (high recall)
- Controls false alarm rate
- Maximizes balance between sensitivity and specificity
- Better than default 0.5 or Youden's J (which can under-predict)

### 14.5 LOSO Cross-Validation & Aggregation

**Validation Strategy:** Leave-One-Subject-Out Cross-Validation
```python
for test_subject in all_subjects:
    # 1. Train on N-1 subjects
    train_data = all_subjects - test_subject
    model.fit(train_data)
    
    # 2. Find optimal threshold on TRAINING data
    fold_threshold = find_optimal_threshold(train_proba, train_labels)
    
    # 3. Apply to TEST subject
    test_proba = model.predict(test_subject)
    test_pred = (test_proba >= fold_threshold).astype(int)
    
    # 4. Store predictions made with fold-specific threshold
    all_predictions.extend(test_pred)
```

**Aggregation (CRITICAL FIX):**
```python
# ❌ WRONG (old approach):
mean_threshold = np.mean([fold1_thresh, fold2_thresh, ...])
all_pred = (all_proba >= mean_threshold)  # Re-thresholding!

# ✅ CORRECT (current approach):
# Use actual predictions made with fold-specific thresholds
aggregate_metrics = compute_metrics(
    all_true_labels,
    all_predictions  # No re-thresholding!
)

# Report threshold distribution:
threshold_mean = 0.31 ± 0.02
threshold_range = [0.28, 0.35]
```

**Why This is Critical:**
- Each fold has different class balance, physiology, optimal threshold
- Averaging thresholds is statistically meaningless
- Re-thresholding violates constraints and creates inconsistency
- Aggregate must reflect what actually happened (fold-specific thresholds)

### 14.6 Model Saving Strategy

**LOSO produces 21 different models** (one per fold), not a single "final" model.

**Saved Artifacts:**
```
results/checkpoints/
├── fold_1_subject_ABC.pt    # Trained on 20 subjects (excluding ABC)
├── fold_2_subject_DEF.pt    # Trained on 20 subjects (excluding DEF)
├── ...
├── fold_21_subject_XYZ.pt   # Trained on 20 subjects (excluding XYZ)
└── best_model.pt            # Copy of fold with highest G-mean

results/config/
├── training_config.json     # Hyperparameters
└── training_summary.txt     # Complete results

results/
├── moment_fold_metrics.csv  # Per-fold performance
└── predictions.npz          # All predictions for analysis
```

**Key Decision:** No "final model" for LOSO
- LOSO is for EVALUATION, not deployment
- Best fold model represents best generalization
- For deployment: Retrain on all 21 subjects (future work)

### 14.7 Loss Functions & Class Imbalance

**Strategy:** Combine weighted loss + threshold optimization

**During Training:**
```python
# Weighted Cross-Entropy
weight = [1.0, n_neg / n_pos]  # e.g., [1.0, 9.0] for 10% stress
criterion = nn.CrossEntropyLoss(weight=weight)

# Effect: Model pays 9× more attention to stress examples
```

**During Inference:**
```python
# Threshold optimization on training data
optimal_threshold = find_best_threshold(train_predictions)

# Apply to test data
test_pred = (test_proba >= optimal_threshold)
```

**Why Both?**
- Weighted loss: Improves feature learning, better probability calibration
- Threshold optimization: Adapts decision boundary to requirements
- Complementary: Loss trains model, threshold optimizes decisions

**Alternatives considered:**
- SMOTE: ❌ Not recommended for time series (creates unrealistic interpolations)
- Focal Loss: Could be explored, but weighted CE + threshold works well

---

## 15. Best Practices & Lessons Learned

### 15.1 Data Processing

**Signal Quality:**
* ✅ Extract HR/HRV at native PPG rate (64Hz), THEN downsample to target rate
* ❌ Never downsample raw PPG to 1Hz or 8Hz (destroys cardiac waveform)
* ✅ Use HeartPy with proper preprocessing (bandpass filter, artifact removal, RR cleaning)
* ✅ Handle missing data explicitly (NaN, not zeros)

**Temporal Alignment:**
* ✅ All signals aligned to common time grid (1Hz for MOMENT, 8Hz for SSL)
* ✅ Use interpolation for upsampling (1Hz → 8Hz), decimation for downsampling (32Hz → 8Hz)
* ✅ Skip first 5 minutes (sensor settling period)
* ✅ No overlap in windows (prevents data leakage)

### 15.2 Labeling & Evaluation

**Labeling Strategy:**
* ✅ Pure prediction (future onset only, not current detection)
* ✅ Separate emotional stress from physical activity
* ✅ Use prediction horizons (3, 5, 10 min) for early warning
* ✅ Label based on window_end, not window_center (prevents leakage)

**Critical Fixes Implemented:**
1. **Aggregation Fix:** Use fold-level predictions, never re-threshold with mean
2. **Threshold Reporting:** Report distribution (min/max/mean±std), not single value
3. **Evaluation Consistency:** `aggregate_metrics ≈ average(fold_metrics)` guaranteed
4. **No Misleading Thresholds:** Only store threshold when actually used

**LOSO Cross-Validation:**
* ✅ Per-fold threshold optimization on TRAINING data only
* ✅ Store predictions made with fold-specific thresholds
* ✅ Aggregate without re-thresholding
* ✅ Save best model based on G-mean (not AUROC)

### 15.3 Model Training

**Handling Class Imbalance:**
* ✅ Use BOTH weighted loss AND threshold optimization (complementary)
* ✅ Weighted cross-entropy during training (improves learning)
* ✅ Constrained threshold optimization during inference (meets requirements)
* ❌ Don't use SMOTE for time series (creates unrealistic interpolations)

**Training Best Practices:**
* ✅ Selective backbone unfreezing for foundation models (prevent catastrophic forgetting)
* ✅ Discriminative learning rates (lower LR for backbone, higher for head)
* ✅ Early stopping with patience (prevent overfitting)
* ✅ Subject-wise normalization (handle inter-subject variability)

**Logging & Monitoring:**
* ✅ Compact logging (avoid Kaggle/Colab crashes from verbose output)
* ✅ Epoch-level logging every 50 epochs for long training
* ✅ Per-fold metrics with all key indicators (G-mean, recall, precision, AUROC)
* ✅ Progress bars with meaningful postfix (loss, metrics)

### 15.4 Model Selection & Checkpointing

**Model Selection Criterion:**
* ✅ Use G-mean (geometric mean of recall and specificity), NOT AUROC
* **Rationale:** AUROC is threshold-independent, doesn't reflect actual deployment
* G-mean better reflects balanced performance at operating point

**Checkpoint Strategy:**
* ✅ Save every 5th fold + best model + last fold
* ✅ Store fold-specific metrics and thresholds in checkpoint
* ✅ Include hyperparameters for reproducibility
* ❌ Don't create "final model" for LOSO (use best fold instead)

### 15.5 Common Pitfalls Avoided

**Aggregation:**
* ❌ NEVER re-threshold aggregate predictions with mean threshold
* ✅ Use actual fold-level predictions made with fold-specific thresholds

**Threshold Reporting:**
* ❌ NEVER report single "mean threshold" as if it was used
* ✅ Report threshold distribution (shows per-subject adaptation)

**Model Saving:**
* ❌ NEVER save untrained model with random weights as "final"
* ✅ Save best fold model (actual trained weights)

**Task Definition:**
* ❌ NEVER mix prediction and detection in one model
* ✅ Focus on pure prediction (future onset only)

**Loss Functions:**
* ❌ NEVER use SMOTE for time series data
* ✅ Use weighted loss + threshold optimization together

---

## 16. Documentation & Reproducibility

### Key Documentation Files

**Technical Fixes:**
- `AGGREGATION_FIX.md` - How to properly aggregate LOSO results
- `THRESHOLD_METHOD_UPDATE.md` - Constrained G-mean thresholding
- `THRESHOLD_PARAMETER_FIX.md` - Honest threshold reporting
- `LOSO_MODEL_SAVING.md` - Proper model checkpointing for LOSO

**Architecture Decisions:**
- `HRV_RMSSD_ADDITION.md` - Why and how RMSSD was added
- `WINDOWING_LABELING_PIPELINE.md` - Complete windowing and labeling explanation

**Experiment-Specific:**
- `experiments/02_moment/HYPERPARAMETERS.md` - MOMENT training configuration
- Various README files in experiment directories

### Reproducibility Checklist

✅ **All experiments use shared utilities** (`experiments/shared/`)
✅ **Configuration-driven** (hyperparameters in config files)
✅ **Deterministic** (fixed random seeds)
✅ **Version controlled** (git-friendly, no large binary files)
✅ **Documented decisions** (markdown files explain key choices)
✅ **Validated approaches** (citations to papers where applicable)

---

## 17. Best Practices & Code Quality

---

## 18. Key Technical Contributions

### 18.1 Novel Methodological Contributions

1. **Pure Prediction vs Detection:**
   - Clear task definition: Predict future onset (not detect current state)
   - Forces learning of subtle pre-stress patterns
   - Clinically valuable (enables preventive intervention)

2. **Emotional vs Physical Stress Separation:**
   - Only emotional stress labeled as positive class
   - Prevents false positives from exercise/physical activity
   - Model learns context-aware stress detection

3. **Constrained Threshold Optimization:**
   - Ensures clinical requirements (recall ≥ 85%)
   - Controls false alarm rate (FPR ≤ 20%)
   - Maximizes balance (G-mean) within constraints

4. **Proper LOSO Aggregation:**
   - Uses fold-level predictions (no re-thresholding)
   - Reports threshold distributions (not misleading single value)
   - Scientifically valid evaluation

5. **Multi-Rate Signal Processing:**
   - HR/HRV extracted at native 64Hz (gold standard)
   - Each signal processed at appropriate rate
   - 8-channel multimodal input with RMSSD (novel for stress prediction)

### 18.2 Implementation Innovations

1. **Foundation Model Fine-Tuning:**
   - MOMENT pre-trained model adapted for stress prediction
   - Selective unfreezing strategy (last 2 blocks)
   - Discriminative learning rates

2. **Subject-Aware Contrastive Learning:**
   - InfoNCE + adversarial subject-invariance
   - Learns subject-invariant stress representations
   - Multiple threshold method comparison per fold

3. **Robust Evaluation Pipeline:**
   - Shared utilities across all experiments
   - Consistent metrics and thresholding
   - Comprehensive logging and checkpointing

### 18.3 Technical Fixes & Best Practices

**Critical Bugs Fixed:**
1. ✅ Aggregation strategy (no re-thresholding)
2. ✅ Threshold reporting (distributions, not misleading means)
3. ✅ LOSO model saving (no untrained "final" models)
4. ✅ Evaluation parameter handling (threshold only when used)

**Best Practices Established:**
1. ✅ Weighted loss + threshold optimization (complementary)
2. ✅ Subject-wise normalization (handles variability)
3. ✅ No overlap in windows (prevents leakage)
4. ✅ Future-only labeling (pure prediction)
5. ✅ Compact logging (prevents system crashes)

---

## 19. License & Ethics

* **VitaStress dataset:** Publicly available research dataset, respect usage terms
* **Privacy:** All data remains local; subject IDs anonymized in code/reports
* **Research ethics:** This project advances stress prediction research, not clinical care
* **Disclaimers:** Not a medical device, does not replace professional healthcare
* **Reproducibility:** Code and methods fully documented for scientific validation

---

## 20. Future Work & Extensions

### Immediate (Post-Retraining):
- Retrain all models with 8 channels (including RMSSD)
- Compare 7-channel vs 8-channel performance
- Window size experiments (60s, 120s, 180s) for RQ1

### Short-Term:
- Attention weight analysis for interpretability (RQ2)
- Feature ablation studies (impact of each channel)
- Per-subject performance analysis (identify difficult cases)
- Statistical significance testing between models

### Long-Term (Post-Dissertation):
- Separate detection model (complementary to prediction)
- Real-time deployment considerations
- Personalized threshold adaptation
- Multi-subject pre-training strategies
- Integration with intervention mechanisms

---

## 21. Current Project Status (December 2024)

### What's Been Accomplished ✅

**Complete End-to-End Pipeline:**
- ✅ Data loading and multi-rate signal alignment
- ✅ HR/HRV extraction using HeartPy (gold standard)
- ✅ Windowing and labeling (120s windows, 5-min prediction)
- ✅ 4 model architectures implemented and trained
- ✅ LOSO validation with proper aggregation
- ✅ Constrained threshold optimization
- ✅ Comprehensive evaluation metrics

**Models Trained and Validated:**
- ✅ Classical ML (Logistic Regression, Random Forest, XGBoost)
- ✅ MOMENT Foundation Model (341M params, 7.5% trainable)
- ✅ Self-Supervised Learning (subject-aware contrastive)
- ✅ Multi-Rate Late Fusion (native sampling rates)

**Technical Infrastructure:**
- ✅ Shared utilities (`experiments/shared/`)
- ✅ Reproducible configuration management
- ✅ Comprehensive documentation (6+ technical docs)
- ✅ Best practices for LOSO, thresholding, aggregation

### What's Next ⏳

**Immediate (Week 1-2):**
- Retrain models with 8 channels (add RMSSD)
- Compare 7-channel vs 8-channel performance
- Expected: +2-5% improvement in recall/G-mean

**Research Questions (Week 3-4):**
- RQ1: Window size experiments (60s, 120s, 180s)
- RQ2: Feature importance and ablation studies
- RQ3: Statistical comparison between models

**Analysis & Writing (Week 5-8):**
- Synthesize findings
- Generate publication-quality figures
- Write dissertation chapters
- Document limitations and future work

---

## 22. Contact & Resources

**Project Directory:** `/Users/jithuazeez/Documents/Msc/Dissertation/`

**Key Directories:**
- `experiments/` - All model implementations
- `src/` - Core pipeline utilities
- `papers/` - Paper summaries and notes
- `notebooks/` - Exploratory analysis

**Documentation:**
- This file: `project_descritption.md`
- Technical docs: `experiments/*.md`
- README files: `experiments/*/README.md`

**Last Updated:** December 21, 2024

---
