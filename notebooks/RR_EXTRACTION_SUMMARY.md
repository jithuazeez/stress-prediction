# Respiratory Rate Extraction - Implementation Summary

## ✅ What Was Created

### 1. Main Notebook: `extract_respiratory_rate.ipynb`
**Location**: `/Users/jithuazeez/Documents/Msc/Dissertation/notebooks/`

**Purpose**: Extract respiratory rate features from PPG signals using RRest v3.0

**Key Components**:
- **RRestMatlabWrapper**: Python class wrapping Matlab Engine API
  - Initializes Matlab engine
  - Configures RRest with optimized settings (AM, FM, BW features)
  - Processes 60s sub-windows and aggregates to 120s windows
  
- **Extraction Pipeline**:
  - Loads raw PPG signals at 64 Hz
  - Splits each 120s window into two 60s sub-windows
  - Runs RRest on each sub-window
  - Aggregates results: mean, std, min, max, trend
  
- **Comprehensive Validation**:
  - Basic statistics and distribution analysis
  - Physiological validity checking (8-40 bpm range)
  - Missing data analysis
  - Quality metrics and coverage analysis
  - Per-subject analysis
  - Temporal pattern visualization

**Output**: 5 new respiratory features per 120s window
- `rr_mean`: Mean respiratory rate (bpm)
- `rr_std`: Respiratory rate variability (bpm)
- `rr_min`: Minimum RR in window (bpm)
- `rr_max`: Maximum RR in window (bpm)
- `rr_trend`: RR change from first to second 60s sub-window (bpm)

### 2. Documentation Files

#### `RR_EXTRACTION_README.md`
Comprehensive documentation including:
- Prerequisites and setup instructions
- Matlab Engine installation guide
- RRest configuration details
- Troubleshooting guide
- Performance optimization tips
- Integration instructions for ML pipeline
- References and citations

#### `RR_EXTRACTION_QUICKREF.md`
Quick reference card with:
- One-command startup
- Feature descriptions
- Configuration summary
- Common fixes
- Expected results
- Next steps

##  Configuration Used

Based on `backup_code/matlabsettings.md` (validated against RRest wiki):

### RRest Settings:
```matlab
% Features: AM (Amplitude), FM (Frequency), BW (Baseline Wander)
up.al.options.FMe = {'am', 'fm', 'bw'};

% Beat detection: IMS (optimized for PPG)
up.al.options.PDt = {'IMS'};

% Resampling: Cubic spline with bandpass
up.al.options.RS = {'cubB'};

% Estimation: Time-domain methods (5 techniques)
up.al.options.estimate_rr = {'CtO', 'CtA', 'PKS', 'ZeX', 'PZX'};

% Fusion: Smart Fusion (median of valid estimates)
up.al.sub_components.fus_mod = {'SFu'};

% Window: 60s (recommended for capturing RR dynamics)
up.paramSet.winLeng = 60;

% Range: 4-60 bpm (physiologically valid)
up.paramSet.rr_range = [4, 60];
```

### Processing Parameters:
```python
WINDOW_SIZE_SEC = 120      # Stress prediction window
RREST_WINDOW_SEC = 60      # RRest sub-window (2 per 120s window)
PPG_SAMPLING_RATE = 64.0   # Hz (native rate)
MIN_PPG_COVERAGE = 0.5     # Require 50% PPG data minimum
```

## 🔧 Technical Fixes Applied

During creation, two critical fixes were applied:

1. **Path Configuration**: Changed addpath to use `Algorithms/` subdirectory
   ```python
   self.eng.addpath(str(self.rrest_path / 'Algorithms'), nargout=0)
   ```

2. **Function Call**: Fixed RRest function name from `RRest_v3` to `RRest`
   ```matlab
   results = RRest(data, up);  % Correct v3.0 function name
   ```

## 📊 Expected Workflow

### Phase 1: Extraction (~45-90 minutes)
```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/notebooks
jupyter notebook extract_respiratory_rate.ipynb
# Run all cells (Kernel → Restart & Run All)
```

**Output**: `reports/rr_features_from_ppg.csv`

### Phase 2: Validation (automatic)
The notebook automatically generates:
- Statistical summaries
- Quality analysis plots
- Physiological validity checks
- Per-subject summaries
- Temporal pattern examples

### Phase 3: Integration
```python
# In experiments/classical_ml/feature_extraction.py
rr_df = pd.read_csv("../../reports/rr_features_from_ppg.csv")

# Merge with existing features
features = features.merge(
    rr_df[['subject_id', 'window_start', 'rr_mean', 'rr_std', 
           'rr_max', 'rr_min', 'rr_trend']],
    on=['subject_id', 'window_start'],
    how='left'
)
```

### Phase 4: Training & Evaluation
- Train Logistic Regression with new features
- Compare F1-score: Baseline vs. +RR features
- **Expected improvement**: +3-7% F1-score

## 🎯 Scientific Rationale

### Why RR from PPG?
- **Respiratory Sinus Arrhythmia (RSA)**: Heart rate modulates with breathing
- **Pulse Amplitude Modulation**: Breathing affects blood volume (PPG amplitude)
- **Baseline Wander**: Respiratory motion causes PPG baseline drift

### Why 60s Sub-windows?
- **Capture Dynamics**: Breathing rate can change within 120s window
- **Robust Estimates**: 60s = ~8-20 breaths (sufficient for estimation)
- **Feature Richness**: 2 estimates → compute mean, std, trend

### Why Time-Domain Methods?
- **Outperform Frequency-Domain**: Proven in Charlton 2016 study
- **Wearable PPG**: More robust to noise and artifacts
- **Computational Efficiency**: Faster than spectral methods

## 📈 Expected Results

Based on VitaStress dataset characteristics:

| Metric | Expected Value | Interpretation |
|--------|---------------|----------------|
| Total Windows | 3,000-5,000 | Depends on subject count & quality |
| Subjects with Data | 50-70 | Requires sufficient PPG coverage |
| Physiologically Valid | >85% | Within 8-40 bpm range |
| High Quality (>0.8) | >60% | Both sub-windows succeeded |
| Mean RR | 15-20 bpm | Typical for seated cognitive tasks |
| RR Range | 8-35 bpm | Covering rest to stress states |
| Normal Breathing (12-20) | 50-60% | Resting state |
| Elevated (20-30) | 30-40% | Stress response |
| Outliers (<8 or >40) | <15% | Motion artifacts or extreme states |

## 🔬 Validation Criteria

### ✅ Success Indicators:
1. **>85% physiologically valid** (8-40 bpm)
2. **>60% high quality** (quality score >0.8)
3. **>70% good PPG coverage** (>70% samples present)
4. **Clear stress differentiation** (elevated RR in stress windows)
5. **Low missing data** (<10% failed windows)

### ⚠️ Warning Signs:
1. Mean RR <10 or >25 bpm (unusual for dataset)
2. High failure rate (>30% failed subjects)
3. Low quality scores (<50% high quality)
4. No variation in RR (all same value = processing error)

## 📝 Next Steps

### Immediate (After Extraction):
1. ✅ Run notebook to extract RR features
2. ✅ Validate output quality (check plots and statistics)
3. ✅ Review per-subject summary for outliers

### Integration (Classical ML):
1. Update `experiments/classical_ml/feature_extraction.py`
2. Add RR features to feature set
3. Train Logistic Regression model
4. Compare metrics with baseline

### Advanced (Optional):
1. Create RR signal time-series for TCN
2. Experiment with different RRest configurations
3. Feature importance analysis (which RR feature matters most?)
4. Subject-specific RR thresholds

## 🙏 Acknowledgments

- **RRest Toolbox**: Charlton PH et al. (2016)
- **Configuration**: Based on best practices from Charlton 2016 paper
- **Validation Approach**: Inspired by HeartPy quality metrics
- **Integration Strategy**: Discussed in conversation history

## 📚 References

1. Charlton PH, et al. (2016). "Extraction of respiratory signals from the electrocardiogram and photoplethysmogram: technical and physiological determinants." *Physiological Measurement*, 37(5):610-638.

2. Birk J, et al. (2023). "The VitaStress dataset for mental stress detection."

3. RRest Toolbox: https://github.com/peterhcharlton/RRest

---

**Created**: January 1, 2026  
**Author**: AI Assistant  
**Status**: Ready for execution  
**Estimated Time**: 45-90 minutes extraction + validation

