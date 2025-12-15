# Stress Prediction from VitaStress Wearable Data — Project Description

**Goal:** Build a system that predicts whether **high stress / anxiety** is likely to occur in the **next *t* minutes (default: 5–10 min)** from wearable physiological data (HR/HRV, EDA, respiration, activity) collected via the VitaStress dataset.

---

## 0. Quick Summary

* **Dataset:** VitaStress — a real-world wearable dataset with multimodal physiological signals and stress labels collected over extended periods.
* **Phase 1 — Dataset EDA & Feature Engineering:** Analyze VitaStress signals (HR, HRV, EDA, respiration, activity), understand label distributions, and extract meaningful time-series features.
* **Phase 2 — Model Development & Research Questions:** Build and compare multiple model architectures (baselines → RNNs/CNNs → Transformers) while systematically investigating window sizes, feature importance, and model performance trade-offs.
* **Output:** A working prototype that ingests multivariate time-series windows and outputs **P(stress in next *t* min)** + actionable insights on optimal configurations (window size, features, architecture).

---

## 1. Dataset: VitaStress

**VitaStress** is a real-world wearable dataset designed for stress monitoring research. The dataset contains multimodal physiological signals collected from participants in their daily lives using consumer-grade wearables.

### 1.1 VitaStress Overview

* **Signals available:**
  * Heart Rate (HR) and Heart Rate Variability (HRV) metrics (RMSSD, SDNN, etc.)
  * Electrodermal Activity (EDA / skin conductance)
  * Respiration rate
  * Activity/step count and movement metrics
  * Timestamps for temporal analysis
  
* **Labels:** Binary or multi-class stress labels (to be confirmed from EDA)
* **Participants:** 21 subjects with varying amounts of data per participant
* **Collection context:** Real-world, naturalistic settings (daily life)
* **Data format:** CSV files organized by participant ID
* **Sampling characteristics:** Variable sampling rates across modalities (detailed in EDA notebooks)

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

## 3. Labels & Ground Truth Strategy  

* **VitaStress labels:** Use the explicit stress labels provided in the VitaStress dataset (binary or multi-class).
* **Label preprocessing:**
  * Handle any class imbalance (e.g., via class weights, oversampling, or focal loss)
  * Define positive samples as windows that precede stress events by *t* minutes (configurable: 5–10 min)
  * Define negative samples from periods with confirmed low/no stress
* **Temporal alignment:** Ensure labels and physiological signals are properly time-aligned, accounting for any sensor lag or timestamp issues.
* **Quality checks:** Exclude windows with excessive missing data or obvious motion artifacts (unless modeling robust to such noise).

---

## 4. Feature Engineering & Windows

* **Prediction horizon:** *t* ∈ {5, 10} min (configurable, addresses RQ1).
* **Input window sizes:** Systematically vary from 30s to 180s (RQ1 investigation).
* **Aggregation:** Resample/aggregate from native sampling rates → **1 Hz** feature vectors to keep sequence length manageable.
* **Feature extraction from VitaStress modalities:**

  * **HR/HRV:** Heart rate, RR intervals, RMSSD, SDNN, pNN50, LF/HF ratio (if calculable), HR slope/volatility over rolling windows.
  * **EDA:** Tonic level (slow-varying baseline), phasic component (SCR count, amplitude, area), EDA slope/rate-of-change.
  * **Respiration:** Respiration rate, variability (breath-to-breath), irregularity metrics.
  * **Activity:** Step count, movement intensity, activity classification (if available) to identify confounding periods.
  * **Temporal context:** Time-of-day (cyclic encoding), weekday/weekend flags.
  
* **Preprocessing pipeline:**
  * Handle missing data (interpolation for short gaps, masking for longer gaps)
  * Motion artifact detection and suppression
  * Per-subject z-normalization (to handle inter-subject variability)
  * Quality flags for windows with excessive missingness

Deliverables: `src/features/` (feature extractors), `vitastress_signal_processing.ipynb` (validation).

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

## 7. Evaluation & Metrics

### 7.1 Performance Metrics

* **Classification performance:**
  * **AUROC** (overall discriminative ability)
  * **PR-AUC** (Precision-Recall AUC, critical for imbalanced classes)
  * **Recall @ fixed precision** (e.g., recall at 90% or 95% precision)
  * **Precision @ fixed recall** (e.g., precision at 80% recall)
  
* **Operational metrics (real-world utility):**
  * **False alarms per hour** (user trust depends on this)
  * **Time-to-warning** (average lead time before stress event)
  * **Detection rate** (percentage of stress events successfully predicted)
  * **Coverage** (percentage of time with valid predictions, accounting for data quality)
  
* **Calibration:**
  * Reliability curves (predicted probability vs. observed frequency)
  * Brier score (measure of prediction accuracy)

### 7.2 Validation Strategy

* **Subject-wise cross-validation:** Leave-one-subject-out (LOSO) or k-fold with subjects as groups to test generalization to new users.
* **Temporal validation:** Train on earlier time periods, test on later periods to simulate real deployment.
* **No data leakage:** Strict separation between train/validation/test at the subject level.

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

## 9. Repository Layout (current + proposed extensions)

```
.
├─ Datasets/
│  └─ VitaStress/          # VitaStress data (21 subjects)
├─ notebooks/
│  ├─ vitastress_eda.ipynb              # Completed EDA
│  ├─ vitastress_signal_processing.ipynb # Signal quality & feature extraction
│  ├─ dataset_comparison.ipynb          # (legacy)
│  └─ model_evaluation.ipynb            # Model comparison & RQ analysis (to create)
├─ src/                                  # (to create)
│  ├─ data/
│  │  ├─ vitastress_loader.py           # Data loading and windowing
│  │  └─ preprocessing.py               # Cleaning, normalization
│  ├─ features/
│  │  ├─ hrv_features.py
│  │  ├─ eda_features.py
│  │  ├─ respiration_features.py
│  │  └─ activity_features.py
│  ├─ models/
│  │  ├─ baselines.py                   # LogReg, RF, XGBoost
│  │  ├─ cnn.py                         # 1D CNN / TCN
│  │  ├─ rnn.py                         # LSTM/GRU
│  │  └─ transformer.py                 # Self-attention model
│  ├─ train.py                          # Training script
│  ├─ evaluate.py                       # Evaluation & metrics
│  └─ utils.py                          # Helper functions
├─ configs/
│  ├─ experiments/                      # RQ-specific configs
│  │  ├─ rq1_window_sizes.yaml
│  │  ├─ rq2_ablations.yaml
│  │  └─ rq3_architectures.yaml
│  └─ base_config.yaml
├─ reports/
│  ├─ evaluation_results.md
│  └─ figures/
├─ requirements.txt                     # (or pyproject.toml)
├─ project_descritption.md              # This file
└─ README.md
```

---

## 10. Project Milestones

1. **M1 — VitaStress EDA & Signal Processing (Weeks 1–2)** ✅ *Completed*

   * ✅ Exploratory data analysis (`vitastress_draft.ipynb`)
   * ✅ **Accelerometer Data:** Loaded all 21 subjects (~3.4M samples), performed EDA, extracted TSFEL features with 120s windows
   * ✅ **Activity Data:** Combined all subjects, identified physiological features, corrected zero-as-missing analysis for HR, RR, SpO2, BP
   * ✅ **Bioimpedance Data:** Combined all subjects (~25 Hz), analyzed data quality, identified extreme outliers (10M Ω), recommended outlier filtering (100-5000 Ω range)
   * ✅ Signal quality assessment and missing data patterns documented
   * ✅ **Key Findings:**
     - Accelerometer: ~32 Hz sampling, 0 NaN missing values, quality flags present
     - Activity: ~30s intervals, significant zeros in physiological features (HR: 65.6%, RR: 83.9%, SpO2: 81.4%, BP: 83-95%)
     - Bioimpedance: ~25 Hz sampling, 50.2% zeros, extreme outliers detected (requires filtering)
   * 🔄 **In Progress:** Outlier removal for bioimpedance, PPG/EDA feature extraction
   
2. **M2 — Feature Engineering Pipeline (Weeks 2–3)** 🔄 *In Progress*

   * ✅ Set up `src/` directory structure with modular components
   * ✅ Implemented feature extractors: `hrv_features.py`, `eda_features.py`, `respiratory_features.py`, `activity_features.py`, `hr_features.py`, `temperature_features.py`, `ppg_processing.py`
   * ✅ Created data loader: `vitastress_loader.py` (supports all modalities)
   * ✅ Preprocessing utilities: `preprocessing.py` (normalization, artifact handling)
   * ✅ Accelerometer features extracted using TSFEL (time + frequency domain)
   * 🔄 **Next:** Validate features, aggregate activity/bioz to 120s windows, extract PPG/EDA features, merge all modalities
   
3. **M3 — Baseline Models (Weeks 3–4)** 📋 *Planned*

   * ✅ `baselines.py` module created (structure ready)
   * ⏳ Implement classical ML baselines (LogReg, RF, XGBoost)
   * ⏳ Train on multiple window sizes (RQ1 preliminary investigation)
   * ⏳ Establish performance floor and feature importance baseline (RQ2 preliminary)
   * **Deliverable:** `src/models/baselines.py`, initial results
   
4. **M4 — Deep Learning Models (Weeks 4–6)** 📋 *Planned*

   * ✅ `rnn.py` and `transformer.py` modules created (structure ready)
   * ⏳ Implement CNN/TCN, LSTM/GRU architectures
   * ⏳ Implement Transformer with self-attention
   * ⏳ Train and compare all models (RQ3)
   * ⏳ Hyperparameter tuning for best-performing architectures
   * **Deliverable:** `src/models/`, trained model checkpoints
   
5. **M5 — Research Question Investigations (Weeks 6–8)**

   * **RQ1:** Systematic window size experiments (30s to 180s)
   * **RQ2:** Feature importance analysis (permutation, SHAP, attention, ablations)
   * **RQ3:** Comprehensive model comparison with statistical tests
   * Deliverable: Complete experimental results, plots, tables
   
6. **M6 — Analysis, Reporting & Write-up (Weeks 8–9)**

   * Synthesize findings across all RQs
   * Generate figures and tables for dissertation
   * Document limitations and future work
   * Deliverable: `reports/evaluation_results.md`, dissertation chapters

---

## 11. Risks & Mitigations

* **Class imbalance (stress events rare):** 
  * Use class weights in loss function
  * Consider focal loss or oversampling techniques
  * Evaluate with PR-AUC in addition to AUROC
  
* **Inter-subject variability:** 
  * Per-subject normalization
  * Leave-one-subject-out validation to test generalization
  * Consider subject-specific fine-tuning (optional)
  
* **Missing data / sensor dropouts:** 
  * Robust preprocessing with interpolation and masking
  * Train models to handle missing modalities (if feasible)
  * Quality flags to exclude unreliable windows
  
* **Activity confounds (HR/HRV affected by movement):** 
  * Include activity features explicitly
  * Ablation studies to understand confound impact (RQ2)
  * Consider activity-aware models or filtering
  
* **False alarms (user trust):** 
  * Optimize for precision at acceptable recall
  * Evaluate false alarm rate per hour
  * Consider confidence thresholds and uncertainty quantification
  
* **Computational limits (Transformer training):** 
  * Work at 1 Hz aggregated features (manageable sequence lengths)
  * Use efficient attention if needed (linear attention, sparse attention)
  * Start with smaller models, scale up if needed

---

## 12. Non-Goals (Current Scope)

* Clinical diagnosis or treatment recommendations (research prototype only)
* Real-time on-device deployment (offline analysis in notebooks/scripts first)
* Multi-dataset fusion or transfer learning (VitaStress only for this project)
* Complex intervention systems (focus on binary prediction and analysis)

---

## 13. Current Progress Summary (As of November 2024)

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

### ⏳ Immediate Next Actions

**Phase 2A — Complete Feature Engineering (Week 3):**
* [ ] Fix bioimpedance outliers and re-analyze distribution
* [ ] Extract PPG-derived features (HRV: RMSSD, SDNN, pNN50, LF/HF ratio)
* [ ] Extract EDA features (tonic level, phasic SCRs, slope, rate-of-change)
* [ ] Aggregate Activity data to 120s windows (with quality flags)
* [ ] Aggregate Bioimpedance to 120s windows (respiratory rate estimation)
* [ ] Merge all modalities using time-based alignment (window center times)
* [ ] Create unified feature matrix with subject_id and timestamps

**Phase 2B — Label Alignment (Week 3-4):**
* [ ] Parse annotation files to extract stress periods (cognitive, physical, public speaking tasks)
* [ ] Define labeling strategy: 5-10 min prediction horizon
* [ ] Shift labels forward for proactive stress prediction
* [ ] Handle temporal split (no data leakage between train/test)
* [ ] Save labeled dataset ready for modeling

**Phase 3 — Baseline Models (Week 4):**
* [ ] Implement LogReg, RandomForest, XGBoost in `baselines.py`
* [ ] Create training script (`src/train.py`) with LOSO cross-validation
* [ ] Run initial experiments with 120s window size
* [ ] Generate feature importance rankings (RQ2 preliminary results)
* [ ] Document performance floor and training time

---

## 14. Best Practices & Lessons Learned (From EDA Phase)

### Data Quality Validation
* **Always check for physiologically impossible values:** Zeros in HR, RR, SpO2, BP indicate missing data, not actual measurements
* **Visualize distributions before analysis:** Outliers (e.g., 10M Ω bioimpedance) are immediately visible in histograms
* **Cross-reference with paper:** VitaStress paper documents known issues (e.g., RR intervals high missingness)
* **Quality flags are critical:** Activity data includes `*_q` flags that identify unreliable measurements

### Feature Extraction
* **Document temporal metadata:** Include `window_start_time`, `window_end_time`, `window_center_time` for multimodal alignment
* **No overlap for predictive tasks:** Each window should be independent to avoid data leakage
* **Suppress expected warnings:** RuntimeWarnings for kurtosis on low-variance signals are normal (use context managers)
* **Test on single subject first:** Validate logic on one subject before scaling to all 21 subjects

### Code Organization
* **Notebook for exploration, src/ for production:** Prototype in `vitastress_draft.ipynb`, refactor validated code into `src/` modules
* **Type hints and docstrings mandatory:** Future you will thank present you for clear documentation
* **Handle missing data explicitly:** Create `*_cleaned` or `*_analysis` copies instead of modifying originals
* **Use pathlib.Path:** More robust than os.path for file operations

### Computational Efficiency
* **Batch processing with tqdm:** Provides progress bars and ETA for long-running operations
* **Cache intermediate results:** Save combined datasets to avoid re-loading all subjects repeatedly
* **Test on subset first:** Validate pipeline on 2-3 subjects before running on all 21 subjects

### Reproducibility
* **Document sampling rates:** Each modality has different rates (Acc: ~32 Hz, Activity: ~30s, Bioz: ~25 Hz)
* **Save feature extraction parameters:** Window size, overlap, TSFEL config version
* **Use `format='ISO8601'` for timestamps:** Handles variable timestamp formats in VitaStress data
* **Track file paths in code:** Use absolute paths or Path objects to avoid ambiguity

---

## 15. License & Ethics

* **VitaStress dataset:** Respect any usage constraints and licensing terms
* **Privacy:** All data remains local; no personal identifiers in code or reports
* **Research ethics:** This project aims to advance stress prediction research, not to provide clinical care
* **Disclaimers:** Any future application of this work should include clear disclaimers that it is not a medical device and does not replace professional healthcare

---


