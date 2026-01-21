# Signal Preprocessing Pipeline

This document describes the complete preprocessing pipeline used for classical machine learning models (LR, RF, SVM) and the Temporal Convolutional Network (TCN) in the final models experiments.

---

## Table of Contents

1. [Overview](#overview)
2. [Common Preprocessing Steps](#common-preprocessing-steps)
3. [Classical ML Preprocessing](#classical-ml-preprocessing)
4. [TCN Preprocessing](#tcn-preprocessing)
5. [Comparison Summary](#comparison-summary)
6. [Rationale and Design Decisions](#rationale-and-design-decisions)

---

## 1. Overview

### Input Data

**Raw sensor signals** from VitaStress dataset:
- Accelerometer: ~32 Hz (3 axes: X, Y, Z)
- PPG (Green LED): ~64 Hz (for HR/HRV extraction)
- Heat Flux Sensor: 1 Hz (skin_temp, heatflux, cbt)

### Processing Goals

1. **Temporal alignment:** Synchronize all modalities to a common time grid
2. **Subject-wise normalization:** Remove inter-subject physiological differences
3. **Quality filtering:** Exclude low-quality windows
4. **Missing data handling:** Impute or replace missing values
5. **Feature extraction (ML only):** Extract statistical features from raw signals
6. **Sequence preparation (TCN only):** Preserve temporal structure for deep learning

---

## 2. Common Preprocessing Steps

These steps are applied to **both Classical ML and TCN** models.

### Step 1: Raw Signal Loading

**File:** `experiments/shared/raw_loader.py` → `load_raw_signals()`

**Inputs:**
- `<subject_id>_acc.csv` - Accelerometer data (~32 Hz)
- `<subject_id>_ppg2_green_6.csv` - PPG data (~64 Hz)
- `<subject_id>_heat_flux_sensor_temperature.csv` - Thermal data (1 Hz)
- `<subject_id>_annotation.csv` - Event timestamps

**Processing:**
```python
# Load each CSV file
acc_df = pd.read_csv(acc_file)
ppg_df = pd.read_csv(ppg_file)
heatflux_df = pd.read_csv(heatflux_file)
annotation_df = pd.read_csv(annotation_file)

# Standardize timestamp column names
df.rename(columns={"date": "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

# Handle zeros as missing in PPG
ppg_df.loc[ppg_df["value"] == 0, "value"] = np.nan
```

**Output:** Dictionary of DataFrames per subject

---

### Step 2: HR/HRV Extraction from Raw PPG

**File:** `experiments/shared/hrv_extractor.py` → `extract_hrv_from_window()`

**Why extract at native 64 Hz?**
- Heart rate variability requires high temporal resolution
- Beat-to-beat intervals (RR) are ~600-1200 ms
- Lower sampling rates miss subtle HRV variations

**Processing:**
```python
# Extract HR/HRV using HeartPy at 64 Hz
def extract_hrv_from_window(ppg_df, window_start, window_end, sample_rate=64.0):
    # 1. Filter PPG segment for window
    segment = ppg_df[(ppg_df['timestamp'] >= window_start) & 
                     (ppg_df['timestamp'] <= window_end)]
    
    # 2. Preprocess PPG signal
    #    - Remove outliers (|z-score| > 3)
    #    - Interpolate to uniform grid
    #    - Detrend (remove DC offset)
    #    - Bandpass filter (0.5-4 Hz for HR)
    
    # 3. Extract HR/HRV with HeartPy
    working_data, measures = hp.process(
        ppg_signal,
        sample_rate=64.0,
        high_precision=True,
        calc_freq=True
    )
    
    # 4. Return 8 HRV features
    return {
        'hr_bpm': measures['bpm'],
        'rmssd': measures['rmssd'],
        'sdnn': measures['sdnn'],
        'pnn50': measures['pnn50'],
        # ... frequency-domain features
    }
```

**Output:** HR time series at 1 Hz + HRV metrics per window

---

### Step 3: Temporal Alignment to 4 Hz Grid

**File:** `experiments/shared/alignment.py` → `align_signals()`

**Why 4 Hz?**
- Balance between temporal resolution and data size
- Sufficient for stress-related physiological changes (HR varies on seconds scale)
- Enables 120-second windows = 480 samples (manageable for both ML and TCN)

**Strategy:**

| Signal | Native Rate | Target Rate | Method |
|--------|-------------|-------------|--------|
| Accelerometer | ~32 Hz | 4 Hz | **Downsample** via mean aggregation |
| Heat Flux | 1 Hz | 4 Hz | **Upsample** via linear interpolation |
| HR/HRV | Derived at 1 Hz | 4 Hz | **Interpolate** via scipy |

**Processing:**

```python
def align_signals(signals, start_time, end_time, target_hz=4.0):
    # 1. Create uniform time grid
    period_ms = int(1000 / target_hz)  # 250 ms for 4 Hz
    freq_str = f"{period_ms}ms"
    time_grid = pd.date_range(start_time, end_time, freq=freq_str)
    
    aligned = pd.DataFrame({"timestamp": time_grid})
    
    # 2. Align accelerometer (32 Hz → 4 Hz)
    #    Downsample by averaging samples within each 250ms bin
    for col in ["acc_x", "acc_y", "acc_z"]:
        aligned[col] = downsample_mean(acc_df, col, time_grid, 
                                       resample_period=freq_str)
    
    # Calculate magnitude
    aligned["acc_magnitude"] = np.sqrt(
        aligned["acc_x"]**2 + 
        aligned["acc_y"]**2 + 
        aligned["acc_z"]**2
    )
    
    # 3. Align heat flux (1 Hz → 4 Hz)
    #    Upsample via linear interpolation
    for col in ["skin_temp", "heatflux", "cbt"]:
        aligned[col] = resample_signal_scipy(
            hf_df["timestamp"], hf_df[col], 
            time_grid, method="linear"
        )
    
    # 4. Align HR/HRV (1 Hz → 4 Hz)
    #    Interpolate pre-extracted HR time series
    aligned["hr_bpm"] = resample_signal_scipy(
        hr_df["timestamp"], hr_df["hr_bpm"],
        time_grid, method="linear"
    )
    aligned["rmssd"] = resample_signal_scipy(
        hr_df["timestamp"], hr_df["rmssd"],
        time_grid, method="linear"
    )
    
    return aligned
```

**Output:** Aligned DataFrame with 8 channels at 4 Hz

---

### Step 4: Subject-Wise Normalization Statistics

**File:** `experiments/shared/windowing.py` → `compute_subject_stats()`

**Why subject-wise normalization?**
- Removes inter-subject physiological differences (e.g., resting HR varies 60-90 BPM)
- Preserves intra-subject stress-related changes
- Prevents model from learning subject identity instead of stress patterns

**Processing:**

```python
def compute_subject_stats(aligned_df):
    """
    Compute mean and std for ENTIRE subject (before windowing).
    
    This ensures normalization is independent of stress labels
    (no data leakage).
    """
    stats = {}
    
    for channel in aligned_df.columns:
        if channel != "timestamp":
            values = aligned_df[channel].dropna().values
            
            stats[channel] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
    
    return stats
    # Example output:
    # {
    #   "acc_x": {"mean": 0.05, "std": 0.3},
    #   "skin_temp": {"mean": 32.5, "std": 1.2},
    #   "hr_bpm": {"mean": 75.0, "std": 10.5},
    #   ...
    # }
```

**Normalization formula:**
```
z = (x - subject_mean) / subject_std
```

**Output:** Dictionary of per-channel mean/std for each subject

---

### Step 5: Windowing and Labeling

**File:** `experiments/shared/windowing.py` → `create_labeled_windows()`

**Window configuration:**
- Size: 120 seconds
- Overlap: 50% (60 seconds)
- Samples per window: 480 (120s × 4 Hz)

**Labeling strategy:**
```python
# Emotional stress onset prediction (5-minute horizon)
label = 1 if stress_onset in [window_end, window_end + 5min] else 0

# Physical stress is labeled as 0 (negative class)
# This forces the model to distinguish emotional stress from physical activity
```

**Output:** List of window dictionaries with:
- `window_data`: DataFrame (480 samples × 8 channels)
- `label_5min`: Binary label (0/1)
- `subject_stats`: Per-subject normalization statistics
- `window_start`, `window_end`: Timestamps

---

### Step 6: Quality Filtering

**File:** `experiments/final_models/train_classical_ml.py` (lines 146-148)

**Filter criterion:**
```python
# Keep only windows where HR extraction succeeded
df_filtered = df[df['hr_bpm'].notna()].copy()
```

**Rationale:**
- HR/HRV are critical stress indicators
- Failed HR extraction indicates poor PPG quality (motion artifacts)
- Removing these windows improves model reliability

**Impact:** Typically retains ~80-90% of windows

---

## 3. Classical ML Preprocessing

### Overview

Classical ML models require **hand-crafted features** that summarize temporal patterns within each window.

### Step 7a: Feature Extraction

**File:** `experiments/classical_ml/feature_extraction.py` → `BasicFeatureExtractor`

**Total features:** 39 (after feature selection)

#### 3.1 Accelerometer Features (15 features)

**Normalization:** Applied **before** feature extraction
```python
acc_x = (acc_x - subject_stats['acc_x']['mean']) / subject_stats['acc_x']['std']
acc_y = (acc_y - subject_stats['acc_y']['mean']) / subject_stats['acc_y']['std']
acc_z = (acc_z - subject_stats['acc_z']['mean']) / subject_stats['acc_z']['std']
```

**Extracted features:**

| Feature | Description | Physiological Relevance |
|---------|-------------|-------------------------|
| `acc_sma` | Signal magnitude area | Overall movement intensity |
| `acc_magnitude_mean` | Mean magnitude | Average activity level |
| `acc_magnitude_std` | Magnitude std deviation | Movement variability |
| `acc_magnitude_max` | Peak acceleration | Maximum movement intensity |
| `acc_energy` | Energy (sum of squares) | Total movement energy |
| `acc_zcr` | Zero-crossing rate | Movement frequency (fidgeting) |
| `acc_dominant_freq_power` | FFT peak power | Periodic movement strength |
| `acc_entropy` | Signal entropy | Movement randomness (stress-related fidgeting) |
| `acc_correlation_xy` | X-Y axis correlation | Coordinated movement patterns |
| `acc_correlation_xz` | X-Z axis correlation | Coordinated movement patterns |
| `acc_correlation_yz` | Y-Z axis correlation | Coordinated movement patterns |
| `acc_tilt_angle` | Tilt from vertical | Posture indicator |
| `acc_magnitude_iqr` | Interquartile range | Movement consistency |
| `acc_jerk_mean` | Mean jerk (rate of acc change) | Movement smoothness |
| `acc_spectral_entropy` | Frequency-domain entropy | Movement pattern complexity |

**Note:** Feature selection reduced original 65 features to top 15 based on LR coefficients.

---

#### 3.2 Temperature Features (7 features)

**Normalization:** Applied **before** feature extraction
```python
temp = (temp - subject_stats['skin_temp']['mean']) / subject_stats['skin_temp']['std']
```

**Extracted features:**

| Feature | Description | Physiological Relevance |
|---------|-------------|-------------------------|
| `temp_mean` | Mean skin temperature | Baseline thermoregulation |
| `temp_std` | Temperature std deviation | Temperature variability |
| `temp_min` | Minimum temperature | Peripheral vasoconstriction (stress) |
| `temp_max` | Maximum temperature | Peak thermal response |
| `temp_range` | Max - Min | Temperature variation range |
| `temp_slope` | Linear trend (regression) | Cooling/warming trend (vasoconstriction) |
| `temp_change` | End - Start difference | Temporal temperature change |

---

#### 3.3 Heat Flux Features (6 features)

**Normalization:** Applied **before** feature extraction
```python
heatflux = (heatflux - subject_stats['heatflux']['mean']) / subject_stats['heatflux']['std']
```

**Extracted features:**

| Feature | Description | Physiological Relevance |
|---------|-------------|-------------------------|
| `heatflux_mean` | Mean heat flux | Thermal energy transfer rate |
| `heatflux_std` | Heat flux std deviation | Thermal regulation variability |
| `heatflux_min` | Minimum heat flux | Reduced heat loss (vasoconstriction) |
| `heatflux_max` | Maximum heat flux | Peak heat loss |
| `heatflux_range` | Max - Min | Heat flux variation |
| `heatflux_change` | End - Start difference | Temporal heat flux change |

---

#### 3.4 Core Body Temperature Features (3 features)

**Normalization:** Applied **before** feature extraction
```python
cbt = (cbt - subject_stats['cbt']['mean']) / subject_stats['cbt']['std']
```

**Extracted features:**

| Feature | Description | Physiological Relevance |
|---------|-------------|-------------------------|
| `cbt_mean` | Mean core body temperature | Internal temperature regulation |
| `cbt_std` | CBT std deviation | Core temperature variability |
| `cbt_change` | End - Start difference | Temporal CBT change |

---

#### 3.5 HR/HRV Features (8 features)

**Source:** Pre-extracted from 64 Hz PPG via HeartPy

**Features:**

| Feature | Description | Physiological Relevance |
|---------|-------------|-------------------------|
| `hr_bpm` | Mean heart rate | Cardiac response to stress |
| `rmssd` | Root mean square of successive differences | Parasympathetic activity (HRV) |
| `sdnn` | Standard deviation of NN intervals | Overall HRV |
| `pnn50` | % of intervals differing >50ms | Short-term HRV |
| `hrv_lf` | Low-frequency power (0.04-0.15 Hz) | Sympathetic + parasympathetic |
| `hrv_hf` | High-frequency power (0.15-0.4 Hz) | Parasympathetic (respiratory sinus arrhythmia) |
| `hrv_lf_hf_ratio` | LF/HF ratio | Sympathovagal balance |
| `peak_rejection_rate` | % of rejected PPG peaks | Signal quality indicator |

**Note:** Frequency-domain features (lf, hf, ratio) have ~50% missingness due to short window length (120s).

---

### Step 8a: Missing Data Imputation (Classical ML)

**File:** `experiments/final_models/train_classical_ml.py` (lines 172-179)

**Strategy:** Per-fold median imputation (zero data leakage)

**Processing:**
```python
def loso_with_ablation_b(X, y, subjects, model_name, model, ...):
    for test_subject in subjects:
        # Split by subject
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # FIT imputer on TRAIN only
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        
        # TRANSFORM test using train statistics
        X_test = imputer.transform(X_test)
        
        # Train model
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # ... threshold selection and evaluation
```

**Rationale:**
- **Per-fold:** Imputer fitted separately for each LOSO fold
- **No leakage:** Test data never influences imputation statistics
- **Median strategy:** Robust to outliers, preserves data distribution

---

## 4. TCN Preprocessing

### Overview

TCN models operate directly on **raw multivariate time series**, preserving temporal structure for convolutional learning.

### Step 7b: Sequence Preparation

**File:** `experiments/tcn/dataset.py` → `VitaStressTCNDataset`

**Input channels (8):**
- `acc_x`, `acc_y`, `acc_z`
- `skin_temp`, `heatflux`, `cbt`
- `hr_bpm`, `rmssd`

**Output shape:** `(batch, 8 channels, 480 timesteps)`

---

#### 4.1 Channel Extraction

```python
def _prepare_window(self, window):
    df = window["window_data"]  # 480 samples × 8 channels
    subject_stats = window.get("subject_stats", {})
    
    channels = []
    
    for channel_name in ["acc_x", "acc_y", "acc_z", "skin_temp", 
                         "heatflux", "cbt", "hr_bpm", "rmssd"]:
        # 1. Extract channel values
        values = df[channel_name].values  # Shape: (480,)
        
        # 2. Handle missing data
        values = np.nan_to_num(values, nan=0.0)  # NaN → 0
        
        # 3. Pad/truncate to seq_len (480)
        if len(values) < 480:
            values = np.pad(values, (0, 480 - len(values)))
        elif len(values) > 480:
            values = values[:480]
        
        # 4. Normalize
        if subject_stats and channel_name in subject_stats:
            mean = subject_stats[channel_name]["mean"]
            std = subject_stats[channel_name]["std"]
            
            if std > 0:
                values = (values - mean) / std
        
        channels.append(values)
    
    # Stack: (8, 480)
    x = np.stack(channels, axis=0)
    return x.astype(np.float32)
```

---

#### 4.2 Missing Data Handling (TCN)

**Current implementation:**
```python
values = np.nan_to_num(values, nan=0.0)  # Replace NaN with 0
```

**Issue:** Fills NaN with 0 **before** normalization, which creates extreme outliers after z-score transformation.

**Example:**
- HR: mean=75 BPM, std=10
- NaN → 0 → `(0-75)/10 = -7.5` (extreme outlier!)

**Better approach (proposed):**
```python
# Option 1: Forward-fill (temporal continuity)
values = pd.Series(values).fillna(method='ffill').fillna(method='bfill').values

# Option 2: Fill with subject mean (neutral after normalization)
if subject_stats and channel_name in subject_stats:
    fill_value = subject_stats[channel_name]["mean"]
    values = np.where(np.isnan(values), fill_value, values)
```

**Impact:** Current approach likely has minimal impact because:
- Quality filtering removes windows with high HR missingness
- Other channels (acc, temp, heatflux, cbt) have <1% missing data

---

#### 4.3 Subject-Wise Normalization (TCN)

**Processing:**
```python
# Per-channel normalization using subject-level statistics
if subject_stats and channel_name in subject_stats:
    mean = subject_stats[channel_name]["mean"]
    std = subject_stats[channel_name]["std"]
    
    if std > 0:
        values = (values - mean) / std
    else:
        values = values - mean  # Avoid division by zero
```

**Output:** Normalized sequences where:
- Each channel has mean ≈ 0, std ≈ 1 (per subject)
- Preserves relative magnitudes of stress-related changes

---

### Step 8b: No Explicit Imputation

**TCN approach:** Missing values handled inline during sequence preparation (see Step 4.2 above).

**Difference from ML:**
- **No per-fold imputation** (imputation happens once during dataset creation)
- **Simpler but less principled** than ML's per-fold median imputation

---

## 5. Comparison Summary

### Preprocessing Steps Comparison

| Step | Classical ML | TCN |
|------|-------------|-----|
| **1. Raw signal loading** | ✅ Same | ✅ Same |
| **2. HR/HRV extraction (64 Hz PPG)** | ✅ HeartPy | ✅ HeartPy |
| **3. Temporal alignment (4 Hz grid)** | ✅ Downsample/upsample | ✅ Downsample/upsample |
| **4. Subject-wise stats** | ✅ Compute mean/std | ✅ Compute mean/std |
| **5. Windowing** | ✅ 120s, 50% overlap | ✅ 120s, 50% overlap |
| **6. Quality filtering** | ✅ HR availability | ✅ HR availability |
| **7. Feature extraction** | ✅ 39 features | ❌ Raw sequences (8 channels) |
| **8. Normalization** | ✅ Before feature extraction | ✅ Per-channel |
| **9. Missing data** | ✅ Per-fold median imputation | ⚠️ NaN → 0 (inline) |
| **10. Output shape** | `(n_samples, 39)` | `(n_samples, 8, 480)` |

---

### Key Differences

#### 1. **Feature Representation**

**Classical ML:**
- Hand-crafted statistical features (39 total)
- Time/frequency domain aggregation
- **Pros:** Interpretable, domain knowledge encoded
- **Cons:** Loses temporal structure, fixed feature set

**TCN:**
- Raw sensor channels (8 channels)
- Preserves full temporal sequence
- **Pros:** End-to-end learning, captures complex patterns
- **Cons:** Requires more data, less interpretable

---

#### 2. **Missing Data Handling**

**Classical ML:**
- Per-fold median imputation
- Zero data leakage (imputer fitted on train only)
- Principled statistical approach

**TCN:**
- Inline replacement (NaN → 0)
- Imputation happens before normalization (suboptimal)
- Simpler but less rigorous

---

#### 3. **Computational Cost**

**Classical ML:**
- Feature extraction: ~1-2 seconds per window
- Training: Seconds to minutes (shallow models)

**TCN:**
- No feature extraction
- Training: Minutes to hours (deep learning, GPU required)

---

## 6. Rationale and Design Decisions

### 6.1 Why 4 Hz Alignment?

**Alternatives considered:**
- 1 Hz: Too coarse, loses ACC dynamics
- 32 Hz: Redundant for stress signals, increases data size

**Chosen: 4 Hz**
- Sufficient temporal resolution for stress (HR changes on seconds scale)
- 120s window = 480 samples (manageable for TCN)
- Balances information retention and computational cost

---

### 6.2 Why Subject-Wise Normalization?

**Problem:** Physiological signals vary widely between individuals
- Resting HR: 60-90 BPM (50% range)
- Skin temperature: 28-35°C (depending on environment)

**Solution:** Z-score normalization per subject
```
z = (x - subject_mean) / subject_std
```

**Benefits:**
- Removes inter-subject differences
- Preserves intra-subject stress-related changes
- Prevents model from learning subject identity

**Alternative (rejected):** Global normalization
- Would preserve absolute physiological values
- But inter-subject variability would dominate signal
- Model would learn subject identity, not stress patterns

---

### 6.3 Why Extract HR/HRV at 64 Hz?

**Problem:** Activity CSV provides HR at ~0.033 Hz (30-second intervals)
- Too coarse for HRV (requires beat-to-beat intervals)
- Device processing is a "black box" (unclear quality filtering)

**Solution:** Extract HR/HRV directly from 64 Hz PPG using HeartPy
- High temporal resolution captures RR interval variability
- Transparent processing pipeline
- Quality metrics (peak rejection rate) available

**Downside:** ~10-20% of windows have HR extraction failures
- Handled via quality filtering (Step 6)

---

### 6.4 Why 120-Second Windows?

**Literature evidence:**
- Minimum window for reliable HRV: 60-90 seconds
- Stress-related physiological changes: 30-120 seconds

**Trade-offs:**
| Window Size | Pros | Cons |
|-------------|------|------|
| 60s | More windows, higher temporal resolution | Noisy HRV, reactive (not anticipatory) |
| 120s (chosen) | Stable HRV, captures anticipatory patterns | Fewer windows |
| 180s | Very stable HRV | Reduced temporal resolution, fewer windows |

---

### 6.5 Why 50% Overlap?

**Rationale:**
- Increases training data (2× more windows than non-overlapping)
- Captures stress onset at different window positions
- Standard practice in time series classification

**Data leakage concern:** Overlapping windows within subject are correlated
- **Mitigated by:** LOSO cross-validation (no overlap between train/test subjects)

---

### 6.6 Why Per-Fold Imputation (Classical ML)?

**Problem:** Missing values in features (especially HRV frequency-domain)

**Wrong approach:** Impute entire dataset before LOSO
- Test data influences imputation statistics (data leakage!)

**Correct approach:** Per-fold imputation
```python
for each LOSO fold:
    imputer.fit(X_train)        # Fit on train only
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)  # Use train statistics
    train_model(X_train, y_train)
```

**Result:** Zero data leakage, unbiased evaluation

---

### 6.7 Why Not Bandpass Filtering?

**Considered:** Bandpass filter accelerometer to remove gravity (high-pass) and noise (low-pass)

**Decision:** No additional filtering beyond HeartPy's PPG preprocessing

**Rationale:**
- Subject-wise normalization removes DC offsets (gravity)
- 4 Hz downsampling acts as anti-aliasing low-pass filter
- Feature extraction (FFT, entropy) is robust to noise
- Preserves more information for TCN to learn from

---

### 6.8 Why Not Outlier Removal?

**Considered:** Remove outliers (|z-score| > 3) from raw signals

**Decision:** No outlier removal (except in PPG preprocessing)

**Rationale:**
- Extreme values may be genuine stress responses (e.g., HR spike)
- Subject-wise normalization reduces outlier impact
- Quality filtering removes windows with failed HR extraction
- Risk of removing informative signals

---

## Summary

### Classical ML Pipeline

```
Raw sensors (32/64/1 Hz)
    ↓
HR/HRV extraction (64 Hz PPG → HeartPy)
    ↓
Temporal alignment (4 Hz grid)
    ↓
Subject-wise normalization (compute stats)
    ↓
Windowing (120s, 50% overlap)
    ↓
Quality filtering (HR availability)
    ↓
Feature extraction (39 features, normalized)
    ↓
Per-fold median imputation (zero leakage)
    ↓
Train model (LR/RF/SVM)
```

---

### TCN Pipeline

```
Raw sensors (32/64/1 Hz)
    ↓
HR/HRV extraction (64 Hz PPG → HeartPy)
    ↓
Temporal alignment (4 Hz grid)
    ↓
Subject-wise normalization (compute stats)
    ↓
Windowing (120s, 50% overlap)
    ↓
Quality filtering (HR availability)
    ↓
Sequence preparation (8 channels × 480 timesteps)
    ├─ NaN → 0 (inline, before normalization)
    └─ Per-channel z-score normalization
    ↓
Train TCN (end-to-end)
```

---

## References

1. Schreiber, P. et al. (2025). "Stress Detection from Multimodal Wearable Sensor Data." *arXiv:2508.10468v1.*

2. van Gent, P. et al. (2019). "HeartPy: A Python Toolkit for Heart Rate Signal Analysis." *J. Open Res. Software.*

3. Task Force (1996). "Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use." *Circulation*, 93(5), 1043-1065.

4. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv:1803.01271.*
