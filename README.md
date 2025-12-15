# VitaStress Stress Prediction Project

**Proactive stress prediction from multimodal wearable physiological signals**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()
[![Phase](https://img.shields.io/badge/Phase-Feature%20Engineering-orange.svg)]()

---

## 🎯 Project Goal

Build a machine learning system that predicts **high stress/anxiety 5-10 minutes ahead** using physiological signals from the VitaStress wearable dataset. This enables proactive interventions before stress escalates.

---

## 🔬 Research Questions

1. **RQ1: Window Size Optimization** — How does input window size (30s-180s) affect prediction accuracy vs. latency?
2. **RQ2: Feature Importance** — Which physiological signals (HR/HRV, EDA, respiration, activity) best predict imminent stress?
3. **RQ3: Model Architecture** — How do baselines (LogReg, XGBoost) compare to deep learning (LSTM, Transformer)?

---

## 📊 Dataset: VitaStress

- **21 subjects** with naturalistic stress induction protocol
- **13 modalities per subject:** HR, HRV, EDA, respiration, activity, PPG, bioimpedance, temperature, etc.
- **Tasks:** Cognitive (mental arithmetic), physical (exercise), public speaking (TSST)
- **Sampling rates:** Variable (0.03-32 Hz depending on modality)

### Key Findings from EDA

| Modality | Sampling Rate | Missing Data | Status |
|----------|---------------|--------------|--------|
| **Accelerometer** | ~32 Hz | 0% | ✅ Features extracted |
| **Activity (HR/RR/SpO2)** | ~30s | 65-95% | ⚠️ High missingness |
| **Bioimpedance** | ~25 Hz | 50% + outliers | 🔄 Outlier removal needed |
| **PPG/RR Intervals** | Variable | Unknown | ⏳ Not yet processed |
| **EDA/Emography** | Variable | Unknown | ⏳ Not yet processed |

**Critical Issue:** Zeros in physiological features (HR, RR, SpO2, BP, bioz) represent **missing data**, not actual zero values!

---

## 📁 Project Structure

```
.
├── Datasets/
│   └── VitaStress/data/           # Raw data (21 subjects × 13 files each)
│
├── vitastress_draft.ipynb         # 🔍 EDA & prototyping (PRIMARY REFERENCE)
│   ├── ✅ Accelerometer EDA + feature extraction (TSFEL)
│   ├── ✅ Activity EDA + missing data analysis
│   ├── ✅ Bioimpedance EDA + outlier detection
│   ├── 🔄 PPG/EDA processing (in progress)
│   └── ⏳ Label alignment, multimodal fusion (pending)
│
├── src/                           # 🏗️ Production pipeline code
│   ├── data/
│   │   ├── vitastress_loader.py   # ✅ Load all modalities
│   │   └── preprocessing.py       # ✅ Cleaning, normalization
│   ├── features/
│   │   ├── hrv_features.py        # ✅ HRV: RMSSD, SDNN, pNN50, LF/HF
│   │   ├── eda_features.py        # ✅ EDA: tonic/phasic, SCR
│   │   ├── respiratory_features.py # ✅ Respiration rate, variability
│   │   ├── activity_features.py   # ✅ Movement, step count
│   │   ├── hr_features.py         # ✅ Heart rate statistics
│   │   ├── temperature_features.py # ✅ Skin temperature features
│   │   └── ppg_processing.py      # ✅ PPG signal processing
│   ├── models/
│   │   ├── baselines.py           # ⏳ LogReg, RF, XGBoost
│   │   ├── rnn.py                 # ⏳ LSTM/GRU
│   │   └── transformer.py         # ⏳ Self-attention model
│   ├── train.py                   # ⏳ Training loops
│   ├── evaluate.py                # ✅ Metrics, plots
│   └── utils.py                   # ✅ Helper functions
│
├── configs/                       # ⏳ YAML experiment configs
│   ├── rq1_window_sizes.yaml
│   ├── rq2_ablations.yaml
│   └── rq3_architectures.yaml
│
├── reports/                       # 📈 Results, figures, analysis
│   ├── evaluation_results.md
│   └── figures/
│
├── project_descritption.md        # 📋 Detailed project roadmap
├── .cursorrules                   # 📜 Coding standards & best practices
├── vitastress_paper.md            # 📄 Original dataset paper
├── timewindow-paper.md            # 📄 Optimal window size reference
└── README.md                      # 👈 You are here
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3.11 -m venv panic_prediction_env
source panic_prediction_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Explore the Data (Start Here!)

```bash
# Open the main EDA notebook
jupyter notebook vitastress_draft.ipynb
```

**Key Sections:**
- Cells 1-6: Dataset structure and imports
- Cells 7-28: Accelerometer data EDA + TSFEL feature extraction
- Cells 33-50: Activity data EDA + zero-as-missing correction
- Cells 51-73: Bioimpedance data EDA + outlier detection

### 3. Run Feature Extraction Pipeline (Once Ready)

```bash
# Load and process all modalities
python src/data/vitastress_loader.py

# Extract features for all subjects
python src/features/feature_extractor.py --window_size 120 --overlap 0.0
```

### 4. Train Baseline Models

```bash
# Train and evaluate baseline models
python src/train.py --config configs/baseline_config.yaml
```

---

## ✅ Current Progress (November 2024)

### Phase 1: EDA & Signal Processing ✅ **COMPLETED**

- ✅ **Accelerometer:** Loaded 3.4M samples (21 subjects), extracted 880 feature windows (120s), saved to `vitastress_acc_features.csv`
- ✅ **Activity:** Combined all subjects, identified 65-95% missing data in HR/RR/SpO2/BP (zeros = missing), saved cleaned dataset
- ✅ **Bioimpedance:** Combined all subjects, detected extreme outliers (up to 10M Ω), documented need for 100-5000 Ω filtering

### Phase 2: Feature Engineering 🔄 **IN PROGRESS**

- ✅ Accelerometer features extracted (time + frequency domain via TSFEL)
- ✅ Infrastructure set up (`src/` modules for loaders, feature extractors, models)
- 🔄 **Current work:** Bioimpedance outlier removal, PPG/EDA feature extraction
- ⏳ **Next:** Aggregate all modalities to 120s windows, multimodal fusion, label alignment

### Phase 3: Modeling ⏳ **PLANNED**

- ⏳ Baseline models (LogReg, RF, XGBoost)
- ⏳ Deep learning models (LSTM, Transformer)
- ⏳ RQ1/RQ2/RQ3 experiments

---

## 📖 Key Documents

- **[project_descritption.md](project_descritption.md)** — Complete project roadmap, milestones, research questions
- **[.cursorrules](.cursorrules)** — Coding standards, best practices, common pitfalls to avoid
- **[vitastress_draft.ipynb](vitastress_draft.ipynb)** — Primary reference for all data exploration and prototyping
- **[vitastress_paper.md](vitastress_paper.md)** — Original VitaStress dataset paper (ground truth for tasks, modalities)
- **[timewindow-paper.md](timewindow-paper.md)** — Literature on optimal window sizes for cognitive load

---

## 🛠️ Development Workflow

1. **Prototyping:** Use `vitastress_draft.ipynb` for exploration, visualization, testing hypotheses
2. **Production:** Refactor validated code into `src/` modules with docstrings and type hints
3. **Experiments:** Use config files in `configs/` to define hyperparameters and ablations
4. **Evaluation:** Generate reports in `reports/` with tables, figures, and analysis
5. **Documentation:** Update `project_descritption.md` with findings and progress

---

## ⚠️ Important Data Quality Notes

### Critical Issues

1. **Zeros = Missing Data** (not actual zeros!)
   - Features affected: HR, RR, SpO2, systolic/diastolic BP, bioimpedance
   - **Always replace zeros with `np.nan`** before analysis

2. **Bioimpedance Outliers**
   - Normal range: 500-2000 Ω
   - Dataset contains values up to 10,000,000 Ω (sensor errors)
   - **Filter to 100-5000 Ω** before feature extraction

3. **High Missingness in Activity Features**
   - HR: 65.6%, RR: 83.9%, SpO2: 81.4%, BP: 83-95%
   - **Use as supplementary features only**, rely on accelerometer, PPG, EDA as primary

4. **Quality Flags**
   - Activity data includes `bpm_q`, `resp_q`, `spo2_q`, `wearing` flags
   - **Always check quality=1** for reliable measurements

---

## 📊 Feature Extraction Strategy

### Window Parameters
- **Window size:** 120 seconds (based on cognitive load literature)
- **Overlap:** 0% (independent windows for predictive task)
- **Prediction horizon:** 5-10 minutes ahead
- **Temporal metadata:** Include `window_start_time`, `window_end_time`, `window_center_time`

### Feature Categories

| Modality | Time Domain | Frequency Domain | Special |
|----------|-------------|------------------|---------|
| **Accelerometer** | Mean, std, skewness, kurtosis, RMS, SMA | PSD, spectral entropy, FFT coefficients | TSFEL extraction |
| **HRV** | RMSSD, SDNN, pNN50, mean RR | LF/HF ratio, VLF, LF, HF power | From RR intervals |
| **EDA** | Mean, std, slope, rate-of-change | — | Tonic/phasic, SCR count |
| **Respiration** | Mean rate, std, breath-to-breath variability | — | From bioimpedance |
| **Activity** | Step count, movement intensity, posture | — | Direct from activity.csv |
| **Temperature** | Mean, std, slope | — | Skin temperature |

---

## 🎯 Next Steps (Immediate Priorities)

1. **Fix bioimpedance outliers** → Filter to 100-5000 Ω, re-analyze distribution
2. **Extract PPG features** → HRV from `rr_interval` files (RMSSD, SDNN, pNN50, LF/HF)
3. **Extract EDA features** → Tonic/phasic decomposition, SCR counting from `emography` files
4. **Aggregate to 120s windows** → Activity and bioimpedance modalities
5. **Multimodal fusion** → Time-based alignment using window center times
6. **Label extraction** → Parse `annotation.csv` files, define stress periods
7. **Label shifting** → Shift labels 5-10 min forward for proactive prediction

---

## 🤝 Contributing & Collaboration

This is a dissertation project. For questions or collaboration:
- See `project_descritption.md` for detailed technical specifications
- See `.cursorrules` for coding standards and best practices
- All EDA findings documented in `vitastress_draft.ipynb`

---

## 📚 References

- **VitaStress Dataset:** Schmidt, P., et al. "Introducing VitaStress: A Multi-Modal Wearable Dataset for Ambulatory Stress Monitoring." (See `vitastress_paper.md`)
- **TSFEL:** Time Series Feature Extraction Library — https://tsfel.readthedocs.io/
- **NeuroKit2:** Physiological signal processing — https://neuropsychology.github.io/NeuroKit/
- **Window Size Research:** See `timewindow-paper.md` for optimal window selection

---

## 📄 License

This project is for academic research purposes. VitaStress dataset usage subject to original dataset license terms.

---

**Last Updated:** November 24, 2024  
**Status:** Phase 2 (Feature Engineering) — 40% Complete  
**Next Milestone:** M2 completion (Feature extraction pipeline) — Target: Week 3

