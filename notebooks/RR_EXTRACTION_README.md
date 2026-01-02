# Respiratory Rate Extraction from PPG

## Overview

The `extract_respiratory_rate.ipynb` notebook extracts respiratory rate (RR) features from raw PPG signals using the RRest v3.0 toolbox. It processes 120-second windows and generates comprehensive quality and validity analyses.

## Prerequisites

### 1. Matlab Installation
- **Required**: Matlab R2016b or later
- **Matlab Engine for Python** must be installed:
  ```bash
  cd /Applications/MATLAB_R20XXx.app/extern/engines/python
  python setup.py install
  ```
- Verify installation:
  ```python
  import matlab.engine
  eng = matlab.engine.start_matlab()
  eng.quit()
  ```

### 2. RRest Toolbox
- Located at: `experiments/shared/RRest/RRest/RRest_v3.0/`
- Version: v3.0
- No additional setup needed (handled by notebook)

### 3. Python Dependencies
```bash
pip install pandas numpy matplotlib seaborn tqdm jupyter
```

## Configuration

The notebook uses settings from `backup_code/matlabsettings.md`:

### Key Parameters:
- **Window Size**: 120 seconds (for stress prediction)
- **RRest Sub-window**: 60 seconds (recommended for capturing RR dynamics)
- **PPG Sampling Rate**: 64 Hz
- **Min PPG Coverage**: 50% (windows with <50% data are skipped)

### RRest Configuration:
- **Features**: AM (Amplitude Modulation), FM (Frequency Modulation), BW (Baseline Wander)
- **Beat Detection**: IMS (Incremental-Merge Segmentation)
- **Resampling**: Cubic spline with bandpass filtering
- **RR Estimation**: Time-domain methods (CtO, CtA, PKS, ZeX, PZX)
- **Fusion**: Smart Fusion (median of valid estimates)
- **RR Range**: 4-60 bpm

## Running the Notebook

### Step 1: Start Jupyter
```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/notebooks
jupyter notebook extract_respiratory_rate.ipynb
```

### Step 2: Execute Cells Sequentially
1. **Setup**: Imports and path configuration
2. **Configuration**: Define paths and parameters
3. **RRest Wrapper**: Matlab engine integration
4. **Helper Functions**: Subject-level extraction logic
5. **Extraction**: Process all subjects (this takes ~30-60 min)
6. **Save Results**: Export to CSV
7. **Validation**: Comprehensive quality analysis

### Step 3: Monitor Progress
- The extraction loop shows a progress bar via tqdm
- Matlab engine will display RRest status messages
- Failed subjects are tracked and reported

## Output Files

### Main Output:
- **`reports/rr_features_from_ppg.csv`**
  - Columns: `subject_id`, `window_start`, `window_end`, `rr_mean`, `rr_std`, `rr_min`, `rr_max`, `rr_trend`, `n_sub_windows`, `quality`, `ppg_coverage`
  - One row per 120-second window
  - Ready to merge with stress labels for ML training

### Summary Files:
- **`reports/rr_subject_summary.csv`**: Per-subject statistics
- **`reports/rr_feature_distributions.png`**: 6-panel feature visualization
- **`reports/rr_quality_analysis.png`**: Quality vs physiological validity
- **`reports/rr_temporal_example.png`**: Example time-series for one subject

## Validation Metrics

The notebook automatically validates:

### 1. Physiological Validity
- **Bradypnea**: <8 bpm (abnormally slow, potential artifact)
- **Very Low**: 8-12 bpm (low but possible)
- **Normal Resting**: 12-20 bpm ✓ (expected for resting state)
- **Elevated/Stress**: 20-30 bpm ✓ (expected during stress tasks)
- **High Stress**: 30-40 bpm (high stress or exercise)
- **Tachypnea**: >40 bpm (abnormally fast, potential artifact)

### 2. Extraction Quality
- **Quality Score**: 0.0-1.0 (based on number of successful RRest estimates)
- **High Quality**: >0.8 (both 60s sub-windows succeeded)
- **Medium Quality**: 0.5-0.8 (one sub-window succeeded)
- **Low Quality**: <0.5 (partial estimates only)

### 3. Data Coverage
- **PPG Coverage**: Percentage of expected PPG samples in window
- **Minimum Threshold**: 50% (configurable)
- **Target**: >70% for high-quality windows

## Expected Results

Based on VitaStress dataset characteristics:

- **Total Windows**: ~3,000-5,000 (depends on subjects and quality filtering)
- **Subjects**: ~50-70 (with sufficient PPG data)
- **Physiologically Valid**: >85% (8-40 bpm range)
- **High Quality**: >60% (quality score >0.8)
- **Mean RR**: 15-20 bpm (typical for seated tasks)
- **RR Range**: 8-35 bpm (covering rest to stress)

## Troubleshooting

### Issue: Matlab engine fails to start
**Solution**:
```bash
# Check if Matlab is installed
which matlab

# Reinstall Matlab Engine for Python
cd /Applications/MATLAB_R20XXx.app/extern/engines/python
python setup.py install --user
```

### Issue: RRest not found
**Solution**: Verify RRest path in notebook configuration cell:
```python
RREST_PATH = Path.cwd().parent / "experiments" / "shared" / "RRest" / "RRest" / "RRest_v3.0"
print(f"RRest exists: {RREST_PATH.exists()}")  # Should print True
```

### Issue: Many failed subjects
**Likely causes**:
- Insufficient PPG data coverage (<50%)
- PPG file missing or corrupted
- Experiment duration too short

**Check**: Run data quality analysis first to identify subjects with good PPG coverage

### Issue: Low quality scores
**Solutions**:
1. Increase `RREST_WINDOW_SEC` to 90s (more data per estimate)
2. Decrease `MIN_PPG_COVERAGE` to 0.4 (allow more windows)
3. Check PPG signal quality using `notebooks/data_quality_analysis.ipynb`

### Issue: Physiologically invalid RR values
**Investigation**:
- Check if outliers are from specific subjects (systematic issue)
- Visualize PPG signal for those windows (may be motion artifacts)
- Consider filtering: `rr_df = rr_df[(rr_df['rr_mean'] >= 8) & (rr_df['rr_mean'] <= 40)]`

## Performance

- **Processing Time**: ~1-2 minutes per subject (depends on # windows)
- **Total Time**: ~45-90 minutes for full dataset (60 subjects)
- **Memory Usage**: ~2-4 GB (Matlab engine + Python)
- **CPU Usage**: High (RRest is computationally intensive)

### Optimization Tips:
1. Process during low system load
2. Close other applications (especially Matlab instances)
3. Consider processing subjects in batches if memory limited
4. Use SSD for faster I/O if available

## Integration with ML Pipeline

### For Classical ML:
```python
# In experiments/classical_ml/feature_extraction.py
# Merge RR features with existing features:

rr_df = pd.read_csv("reports/rr_features_from_ppg.csv")
rr_features = rr_df[['subject_id', 'window_start', 'rr_mean', 'rr_std', 
                      'rr_max', 'rr_min', 'rr_trend']]

# Merge with window labels on subject_id + window_start
features = features.merge(rr_features, on=['subject_id', 'window_start'], how='left')
```

### For TCN (Optional):
Create a respiratory signal time-series channel by:
1. Interpolating RR estimates to 4 Hz
2. Adding as 4th input channel alongside temperature, HR, accelerometer
3. Update `VitaStressTCNDataset` to load RR signal

## References

- **RRest**: Charlton PH et al. (2016). "Extraction of respiratory signals from the electrocardiogram and photoplethysmogram: technical and physiological determinants." *Physiological Measurement*, 37(5):610.
- **VitaStress Dataset**: Birk et al. (2023). "The VitaStress dataset for mental stress detection."
- **Configuration**: Based on best practices from `backup_code/matlabsettings.md`

## Contact & Support

For issues or questions:
1. Check RRest documentation: `experiments/shared/RRest/RRest.wiki/`
2. Validate configuration against `backup_code/matlabsettings.md`
3. Review previous discussions in conversation history

