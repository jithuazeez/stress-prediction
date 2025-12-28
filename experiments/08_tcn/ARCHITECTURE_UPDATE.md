# TCN Architecture Update Summary

## Overview
Updated TCN model to match MOMENT's architecture, using raw multivariate time series instead of extracted features.

## Changes Made

### 1. Dataset (`dataset.py`)

**Before:**
- Extracted ~65 features on-the-fly (acc, temp, heatflux statistics)
- Output: `(n_features, 1)` - feature vector
- Used `BasicFeatureExtractor` from classical ML

**After:**
- Uses 8 raw sensor channels as time series
- Output: `(8 channels, 120 timesteps)` - multivariate sequence
- Channels: `acc_x`, `acc_y`, `acc_z`, `skin_temp`, `heatflux`, `cbt`, `hr_bpm`, `rmssd`
- Subject-wise normalization (recommended for physiological data)
- No feature extraction

**Key Updates:**
- `CHANNELS` class variable defines 8 sensor channels
- `_prepare_window()` replaces `_extract_window_features()`
- Handles missing channels gracefully (fills with zeros)
- Pads/truncates sequences to exactly 120 timesteps
- Subject-level stats used for normalization (stored in `window["subject_stats"]`)

### 2. Model (`model.py`)

**Before:**
- TCN channels: `[64, 64, 64]` (3 blocks)
- Exponential dilations: `2^0, 2^1, 2^2` = `[1, 2, 4]`
- Global average pooling over all timesteps
- Receptive field: ~15

**After:**
- TCN channels: `[16, 16, 16, 16, 16, 16]` (6 blocks, narrower)
- Custom dilations: `[1, 2, 4, 8, 16, 32]`
- **Last timestep pooling** (maintains causality)
- Receptive field: **127 timesteps** (> 120, so covers full window)
- Reduced parameters due to narrower channels

**Architecture Details:**

```
Input: (batch, 8, 120)
  ↓
TCN Block 1: dilation=1,  channels=16
  ↓
TCN Block 2: dilation=2,  channels=16
  ↓
TCN Block 3: dilation=4,  channels=16
  ↓
TCN Block 4: dilation=8,  channels=16
  ↓
TCN Block 5: dilation=16, channels=16
  ↓
TCN Block 6: dilation=32, channels=16
  ↓
Last timestep: (batch, 16, 120) → (batch, 16)
  ↓
FC1: 16 → 128
  ↓
ReLU + Dropout
  ↓
FC2: 128 → 2 (class logits)
```

**Receptive Field Calculation:**
```
RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
RF = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8 + 16 + 32)
RF = 1 + 2 * 2 * 63
RF = 1 + 126
RF = 127 timesteps
```

Since RF=127 > 120, the model can attend to the entire input window.

**Key Updates:**
- `TemporalConvNet`: Now accepts `dilations` list instead of `dilation_base`
- `TCNClassifier`: Added `use_last_timestep` parameter (default `True`)
- `create_tcn_model()`: Updated defaults to match requirements
- `get_receptive_field()`: Moved to `TemporalConvNet` and updated calculation

### 3. Training (`train.py`)

**Before:**
- Dynamically determined number of features from dataset
- No sequence length specification
- Exponential dilation pattern

**After:**
- Fixed 8 input channels (multivariate sensors)
- Fixed 120 timestep sequences
- Custom dilation pattern `[1, 2, 4, 8, 16, 32]`
- Uses last timestep pooling
- Subject-wise normalization

**Training Configuration:**
```python
results = loso_cross_validation(
    windows_by_subject,
    config,
    device,
    logger,
    n_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    threshold_method="geometric_mean",
    tcn_channels=[16, 16, 16, 16, 16, 16],  # Narrower than before
    kernel_size=3,
    dilations=[1, 2, 4, 8, 16, 32],  # Custom dilations
    dropout=0.3,
    fc_hidden_dim=128,
    use_last_timestep=True  # NEW: Use last timestep instead of avg pooling
)
```

**Key Updates:**
- `loso_cross_validation()`: Added `dilations` and `use_last_timestep` parameters
- Removed `dilation_base` parameter
- Dataset creation: Added `seq_len=120` and `normalization_mode="subject"`
- Model logging: Now shows input channels and sequence length
- Hyperparameters: Added `dilations`, `use_last_timestep`, `normalization`

## Verification Checklist

✅ **Use raw multichannel sequences (8 × 120)**
- Dataset outputs `(8, 120)` instead of `(n_features, 1)`
- Channels: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd

✅ **Treat sensors as channels, not features**
- Each sensor is a separate channel
- No statistical feature extraction
- Preserves temporal structure

✅ **Reduce TCN width**
- Old: `[64, 64, 64]` = 3 blocks
- New: `[16, 16, 16, 16, 16, 16]` = 6 blocks (narrower but deeper)

✅ **Ensure receptive field ≥ 120**
- Dilations: `[1, 2, 4, 8, 16, 32]`
- Receptive field: 127 timesteps
- Covers entire 120-second window

✅ **Use last timestep instead of avg pooling**
- Added `use_last_timestep=True` parameter
- Uses `tcn_out[:, :, -1]` instead of `AdaptiveAvgPool1d(1)`
- Maintains causality for real-time prediction

✅ **Keep weighted loss + thresholding**
- Still uses `CrossEntropyLoss` with class weights
- Threshold optimization using geometric mean on training fold
- No changes to loss/threshold logic

✅ **Select threshold inside training fold**
- Threshold found using training data only
- Applied to test data for evaluation
- Prevents information leakage

✅ **Normalize per subject**
- `normalization_mode="subject"` in dataset
- Uses `subject_stats` from windowing pipeline
- Each sensor normalized by subject-level mean/std

## Model Comparison

| Aspect | Old TCN | New TCN |
|--------|---------|---------|
| Input | `(batch, ~65, 1)` | `(batch, 8, 120)` |
| Data type | Extracted features | Raw time series |
| TCN blocks | 3 blocks | 6 blocks |
| Channels | [64, 64, 64] | [16, 16, 16, 16, 16, 16] |
| Dilations | [1, 2, 4] | [1, 2, 4, 8, 16, 32] |
| Receptive field | ~15 | 127 |
| Pooling | Global average | Last timestep |
| Parameters | ~100K | ~20K (much smaller) |
| Normalization | Dataset-wide | Subject-wise |

## Expected Behavior

1. **Data Loading**: Should load 8 channels from aligned 1Hz data
2. **Missing Channels**: Gracefully handles missing channels (fills with zeros)
3. **Sequence Padding**: Ensures all sequences are exactly 120 timesteps
4. **Subject Normalization**: Uses per-subject mean/std for each channel
5. **Model Forward**: Processes `(batch, 8, 120)` → `(batch, 2)` logits
6. **Receptive Field**: Can attend to entire 120-second window
7. **Real-time Ready**: Last timestep pooling maintains causality

## How to Run

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments/08_tcn
python train.py
```

This will:
1. Load all subjects and create labeled windows
2. Run LOSO cross-validation with 8 subjects
3. Train TCN with new architecture on each fold
4. Find optimal threshold on training data per fold
5. Evaluate on held-out test subject
6. Save results, predictions, and fold metrics

## Expected Improvements

1. **Better Temporal Modeling**: Raw sequences preserve temporal patterns that features lose
2. **Reduced Parameters**: Narrower channels = smaller model = less overfitting
3. **Larger Receptive Field**: RF=127 means model can integrate information from entire window
4. **Causal Prediction**: Last timestep pooling maintains temporal causality
5. **Physiological Consistency**: Subject-wise normalization respects individual baselines

## Notes

- HR/RMSSD channels may be all zeros if alignment code is commented out (see MOMENT issue)
- Model should work even with zeros for HR/RMSSD (uses other 6 channels)
- If HR/RMSSD are available, model gets additional cardiac information
- Receptive field of 127 is slightly larger than needed but ensures full coverage

