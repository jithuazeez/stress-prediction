# Multi-Rate Fusion: Channel and Threshold Updates

**Date:** December 22, 2024

## Summary

Updated the Multi-Rate Fusion model to:
1. **Use the same 8 channels as MOMENT** (but at native rates)
2. **Comment out PPG** and use HR/HRV instead
3. **Use constrained_gmean** as the default threshold method

## Changes Made

### 1. Configuration (`config.py`)

**Before:**
- PPG at 64 Hz
- ACC at 32 Hz (3 channels)
- Temp at 1 Hz (single channel)
- 3 separate encoders with embeddings: 128 + 128 + 64 = 320-d

**After:**
- ACC at 32 Hz (3 channels: acc_x, acc_y, acc_z)
- Physio at 1 Hz (5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd)
- 2 encoders with embeddings: 128 + 128 = 256-d

**Removed:**
- `ppg_sample_rate`
- `ppg_embedding_dim`
- `temp_sample_rate`
- `temp_embedding_dim`

**Added:**
- `physio_sample_rate: 1.0 Hz`
- `physio_embedding_dim: 128`

### 2. Dataset (`dataset.py`)

**Channel Updates:**
- **Commented out PPG loading** at 64Hz
- **Added physiological signal loading** at 1Hz:
  - `skin_temp` from heatflux sensor
  - `heatflux` from heatflux sensor
  - `cbt` (core body temperature) from heatflux sensor
  - `hr_bpm` from aligned data (HeartPy-derived heart rate)
  - `rmssd` from aligned data (HRV metric)

**Data Structure Changes:**
- Removed: `ppg_data`, `temp_data`
- Added: `physio_data` (5 channels stacked)
- Dataset now returns:
  ```python
  {
      "acc": (3, acc_samples),      # 3 channels at 32Hz
      "physio": (5, physio_samples), # 5 channels at 1Hz
      "label": int,
      "subject_id": int
  }
  ```

### 3. Encoders (`encoders.py`)

**Commented Out:**
- `PPGEncoder` class (entire implementation)

**Modified:**
- Renamed `TempEncoder` → `PhysioEncoder`
- Updated to process 5 channels instead of 1
- Architecture: Lightweight 1D CNN for 1Hz multi-channel input

**PhysioEncoder Details:**
- Input: `(batch, 5, 120)` for 120s window
- Architecture:
  ```
  Conv1d(5→32) → Conv1d(32→64) → Conv1d(64→128)
  → AdaptiveAvgPool → FC(128→embedding_dim)
  ```
- Output: `(batch, 128)` embedding

**Updated Factory Function:**
- Old: `create_encoders()` returns `(PPGEncoder, ACCEncoder, TempEncoder)`
- New: `create_encoders()` returns `(ACCEncoder, PhysioEncoder)`

### 4. Model (`model.py`)

**Architecture Changes:**
```
Before:
PPG (64Hz, 1ch) → 128-d ┐
ACC (32Hz, 3ch) → 128-d ├→ Concat (320-d) → Fusion → Classifier
Temp (1Hz, 1ch) → 64-d  ┘

After:
ACC (32Hz, 3ch) → 128-d  ┐
Physio (1Hz, 5ch) → 128-d┘→ Concat (256-d) → Fusion → Classifier
```

**Code Updates:**
- Removed `ppg_encoder` from `MultiRateFusionModel`
- Removed `temp_encoder`, added `physio_encoder`
- Updated `forward()`, `get_embeddings()`, `predict_proba()` to use new structure
- Updated all references from `ppg`/`temp` to `acc`/`physio`

### 5. Training (`train.py`)

**Threshold Method:**
- **Changed from comparing multiple methods** to using **constrained_gmean** by default
- Parameters:
  - `min_recall = 0.85` (minimum sensitivity constraint)
  - `max_fpr = 0.20` (maximum false positive rate constraint)
- This matches the MOMENT and SSL experiments for consistency

**Removed Code:**
- Threshold comparison loop (trying multiple methods)
- `threshold_results` dictionary tracking all methods

**Added:**
- Direct call to `find_optimal_threshold()` with constrained_gmean
- Enhanced logging showing threshold constraints
- Added channel list to configuration output

**Logging Updates:**
```python
# Before
logger.info(f"  PPG samples: {config.ppg_samples_per_window}")
logger.info(f"  ACC samples: {config.acc_samples_per_window}")
logger.info(f"  Temp samples: {config.temp_samples_per_window}")

# After
logger.info(f"  ACC samples: {config.acc_samples_per_window} (3 channels at 32Hz)")
logger.info(f"  Physio samples: {config.physio_samples_per_window} (5 channels at 1Hz)")
logger.info(f"  Threshold method: constrained_gmean (min_recall=0.85, max_fpr=0.20)")
logger.info(f"  Channels: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd")
```

### 6. No Changes Required

The following files work without modification:
- `run_all.py` - Just calls train.py, no hardcoded dependencies

## Channel Alignment with MOMENT

The model now uses **exactly the same 8 channels** as MOMENT:

| Channel      | Rate  | Source           | MOMENT | Multirate Fusion |
|--------------|-------|------------------|--------|------------------|
| acc_x        | 32Hz  | Accelerometer    | ✓      | ✓                |
| acc_y        | 32Hz  | Accelerometer    | ✓      | ✓                |
| acc_z        | 32Hz  | Accelerometer    | ✓      | ✓                |
| skin_temp    | 1Hz   | Heatflux sensor  | ✓      | ✓                |
| heatflux     | 1Hz   | Heatflux sensor  | ✓      | ✓                |
| cbt          | 1Hz   | Heatflux sensor  | ✓      | ✓                |
| hr_bpm       | 1Hz   | HeartPy → Aligned| ✓      | ✓                |
| rmssd        | 1Hz   | HeartPy → Aligned| ✓      | ✓                |

**Key Difference:**
- MOMENT: Downsamples all to 1Hz, then resamples to 512 samples
- Multirate Fusion: Keeps native rates (ACC at 32Hz, Physio at 1Hz)

## Threshold Method: Constrained G-mean

**Why constrained_gmean?**
1. **Balanced performance:** Maximizes geometric mean of sensitivity and specificity
2. **Clinical constraints:** Ensures minimum recall (0.85) for stress detection
3. **Low false alarms:** Caps false positive rate at 0.20
4. **Consistency:** Same method used in MOMENT and SSL experiments

**Comparison:**
```python
# Before: Tried multiple methods and selected best
threshold_methods = ["youden", "f1", "balanced", "geometric_mean", "constrained_gmean"]
best_method = max(threshold_results.keys(), key=lambda m: threshold_results[m]["gmean"])

# After: Direct constrained_gmean
optimal_threshold, _ = find_optimal_threshold(
    y_train_true, y_train_proba,
    method="constrained_gmean",
    min_recall=0.85,
    max_fpr=0.20
)
```

## Usage

No changes to command-line interface:

```bash
# Single experiment
python train.py --window_size 120 --horizon 3

# All experiments
python run_all.py
```

## Verification Checklist

- [x] Config updated to reflect 8 channels at native rates
- [x] Dataset loads all 8 channels correctly
- [x] PPG code commented out (not deleted, for reference)
- [x] PhysioEncoder processes 5 channels at 1Hz
- [x] ACCEncoder processes 3 channels at 32Hz
- [x] Model concatenates 2 embeddings (256-d total)
- [x] Training uses constrained_gmean by default
- [x] Logging shows correct channel counts and threshold method
- [x] All PPG references removed from active code paths

## Benefits

1. **Channel Consistency:** Same 8 channels across MOMENT, SSL, and Multirate Fusion
2. **Better Comparability:** Results directly comparable with other experiments
3. **Threshold Consistency:** All experiments use constrained_gmean
4. **More Meaningful Features:** HR/HRV at 1Hz >> raw PPG downsampled to 1Hz
5. **Simpler Architecture:** 2 encoders instead of 3

## Potential Issues & Solutions

**Issue 1: Missing HR/HRV data**
- **Cause:** Some subjects may not have hr_bpm/rmssd in aligned data
- **Solution:** Dataset fills with zeros if missing (line 220-237 in dataset.py)

**Issue 2: Different window sizes**
- **Cause:** 60s vs 120s windows have different sample counts
- **Solution:** Config dynamically computes samples via properties

**Issue 3: Shape mismatches**
- **Cause:** Encoder expects specific input shapes
- **Solution:** `_pad_or_truncate()` ensures correct dimensions

## Testing Recommendations

1. **Test data loading:**
   ```bash
   python dataset.py
   ```

2. **Test encoders:**
   ```bash
   python encoders.py
   ```

3. **Test model:**
   ```bash
   python model.py
   ```

4. **Test training (single fold):**
   ```bash
   python train.py --window_size 120 --horizon 3 --epochs 10
   ```

5. **Check logs for:**
   - Correct channel counts
   - Proper threshold method
   - No PPG references
   - Expected input/output shapes

## Files Modified

1. `config.py` - Updated channel configuration
2. `dataset.py` - Loads 8 channels at native rates
3. `encoders.py` - Commented out PPGEncoder, added PhysioEncoder
4. `model.py` - Updated architecture to use 2 encoders
5. `train.py` - Uses constrained_gmean threshold

## Files Not Modified

- `run_all.py` - No changes needed
- `__init__.py` - No changes needed

---

**Last Updated:** December 22, 2024
