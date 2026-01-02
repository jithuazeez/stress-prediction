# RR Extraction Quick Reference

## 🚀 Quick Start

```bash
# 1. Install Matlab Engine for Python (one-time setup)
cd /Applications/MATLAB_R20XXx.app/extern/engines/python
python setup.py install

# 2. Launch notebook
cd /Users/jithuazeez/Documents/Msc/Dissertation/notebooks
jupyter notebook extract_respiratory_rate.ipynb

# 3. Run all cells (Kernel → Restart & Run All)
# ⏱️ Estimated time: 45-90 minutes
```

## 📊 What Gets Extracted

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `rr_mean` | Mean respiratory rate in window | 12-25 bpm |
| `rr_std` | Variability of RR | 0-5 bpm |
| `rr_min` | Minimum RR in window | 8-20 bpm |
| `rr_max` | Maximum RR in window | 15-35 bpm |
| `rr_trend` | RR change (60s→60s) | -10 to +10 bpm |

## 🎯 Key Configuration

```python
WINDOW_SIZE_SEC = 120      # Stress prediction window
RREST_WINDOW_SEC = 60      # RRest analysis window (2 per 120s)
MIN_PPG_COVERAGE = 0.5     # Require 50% PPG data
PPG_SAMPLING_RATE = 64.0   # Hz

# RRest: AM + FM + BW features, Smart Fusion, Time-domain estimation
```

## 📁 Output Files

```
reports/
├── rr_features_from_ppg.csv          ← MAIN OUTPUT (for ML training)
├── rr_subject_summary.csv            ← Per-subject stats
├── rr_feature_distributions.png      ← 6-panel visualization
├── rr_quality_analysis.png           ← Quality validation
└── rr_temporal_example.png           ← Time-series example
```

## ✅ Validation Checklist

- [ ] **Physiologically valid**: >85% of windows in 8-40 bpm range
- [ ] **High quality**: >60% of windows with quality >0.8
- [ ] **Good coverage**: >70% of windows with PPG coverage >0.7
- [ ] **Reasonable mean**: Overall RR mean 15-20 bpm
- [ ] **Stress variation**: Clear difference between normal (12-20) and elevated (20-30) windows

## 🔧 Common Issues

| Issue | Quick Fix |
|-------|-----------|
| Matlab engine error | `python -c "import matlab.engine"` (check installation) |
| RRest not found | Verify path: `experiments/shared/RRest/RRest/RRest_v3.0/` |
| Many failed subjects | Check PPG coverage in `data_quality_analysis.ipynb` |
| Low quality scores | Increase `RREST_WINDOW_SEC` to 90s or decrease `MIN_PPG_COVERAGE` to 0.4 |
| Invalid RR values | Filter: `rr_df = rr_df[(rr_df['rr_mean'] >= 8) & (rr_df['rr_mean'] <= 40)]` |

## 🧪 Next Steps

### 1. Integrate with Classical ML
```python
# In experiments/classical_ml/feature_extraction.py
rr_df = pd.read_csv("reports/rr_features_from_ppg.csv")
# Add rr_mean, rr_std, rr_max, rr_min, rr_trend to feature set
```

### 2. Expected Performance Gain
- **Baseline F1**: ~0.65-0.70 (without RR)
- **With RR Features**: ~0.68-0.75 (+3-7% improvement)
- **Rationale**: RR is a strong stress biomarker (breathing rate ↑ during stress)

### 3. Optional: TCN Integration
- Create RR signal time-series at 4 Hz
- Add as 4th input channel to TCN
- More complex but potentially higher gains

## 📖 Full Documentation

See `RR_EXTRACTION_README.md` for:
- Detailed prerequisites
- Troubleshooting guide
- Performance optimization
- RRest configuration details
- References and citations

