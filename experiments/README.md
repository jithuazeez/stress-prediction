# VitaStress Multi-Model Stress Prediction Experiments

This directory contains 4 distinct modeling approaches for stress prediction from multimodal wearable sensor data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Note: If learn2learn fails on Python 3.11+, use:
L2L_USE_CYTHON=0 pip install learn2learn

# Run experiments in order
python 01_classical_ml/train.py
python 02_moment/train.py
python 03_ts2vec/pretrain.py
python 03_ts2vec/train_classifier.py
python 04_maml/train.py

# Compare all results
python compare_all.py
```

## Features

- **Progress bars**: All experiments show progress with `tqdm`
- **Colored logging**: Console output with colors for easy reading
- **Log files**: Each experiment saves detailed logs to `results/training.log`
- **LOSO tracking**: Per-fold metrics shown during training

## Directory Structure

```
experiments/
├── shared/                      # Shared utilities
│   ├── raw_loader.py           # Load raw signals from VitaStress
│   ├── alignment.py            # Align modalities to 1Hz
│   ├── windowing.py            # Create labeled windows
│   ├── evaluation.py           # Metrics and plotting
│   ├── config.py               # Configuration
│   └── logging_utils.py        # Progress bars and logging
│
├── 01_classical_ml/            # XGBoost, RF, LogReg
│   ├── feature_extraction.py   # Extract ~38 basic features
│   └── train.py                # LOSO training
│
├── 02_moment/                  # MOMENT Foundation Model
│   ├── model.py                # MOMENT wrapper
│   ├── dataset.py              # PyTorch dataset
│   └── train.py                # Fine-tuning script
│
├── 03_ts2vec/                  # TS2Vec Contrastive SSL
│   ├── dataset.py              # Data preparation
│   ├── pretrain.py             # Self-supervised pretraining
│   └── train_classifier.py     # Train classifier on embeddings
│
├── 04_maml/                    # MAML Meta-Learning
│   ├── model.py                # Base classifier
│   ├── meta_dataset.py         # Task/episode sampling
│   └── train.py                # Meta-training script
│
├── compare_all.py              # Compare all experiments
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Data Flow

```
Raw Data Files (per subject)
├── acc.csv (~32 Hz) ─────────────────────┐
├── emography.csv (~0.017 Hz) ────────────┤
├── heat_flux_sensor_temperature.csv (1Hz)├──► Alignment (1Hz) ──► Windowing ──► Features/Raw
├── ppg2_green_6.csv (~64 Hz) ────────────┤                                          │
└── annotation.csv ───────────────────────┘                                          │
                                                                                     ▼
                                                                    ┌────────────────┴────────────────┐
                                                                    │                                 │
                                                              Classical ML                    Deep Learning
                                                              (XGBoost/RF)                (MOMENT/TS2Vec/MAML)
```

## Experiments

### 1. Classical ML (01_classical_ml)

**Approach:** Extract ~38 statistical features, train XGBoost/RF with LOSO CV

**Features:**
- Accelerometer: magnitude stats, SMA, energy, ZCR (15 features)
- Temperature: skin_temp stats, trend (7 features)
- Heat flux: mean, std (2 features)
- EDA: mean, std, peaks, trend (9 features)
- PPG: signal mean, std, range (3 features)
- Core body temp (2 features)

### 2. MOMENT Foundation Model (02_moment)

**Approach:** Fine-tune pre-trained MOMENT transformer for classification

**Reference:** [MOMENT: A Family of Open Time-series Foundation Models](https://github.com/moment-timeseries-foundation-model/moment)

**Input:** 4-channel time series (acc_magnitude, skin_temp, eda, ppg_mean)

### 3. TS2Vec Contrastive SSL (03_ts2vec)

**Approach:** Self-supervised pretraining, then train classifier on embeddings

**Reference:** [TS2Vec: Towards Universal Representation of Time Series](https://github.com/zhihanyue/ts2vec)

**Steps:**
1. Pretrain encoder on ALL data (no labels)
2. Extract embeddings
3. Train SVM/MLP classifier

### 4. MAML Meta-Learning (04_maml)

**Approach:** Learn to quickly adapt to new subjects with few samples

**Reference:** [Model-Agnostic Meta-Learning](https://arxiv.org/pdf/1703.03400) | [learn2learn](https://github.com/learnables/learn2learn)

**Key idea:** Each subject = one task. Model learns to adapt with 5 samples.

## Evaluation

All experiments use **Leave-One-Subject-Out (LOSO)** cross-validation:
- Train on 20 subjects
- Test on 1 held-out subject
- Repeat for all 21 subjects

**Metrics:**
- AUROC (primary)
- PR-AUC (for imbalanced data)
- F1, Precision, Recall
- Accuracy

## Results

After running all experiments, use `compare_all.py` to generate:
- `comparison_results/model_comparison.csv` - Full comparison table
- `comparison_results/comparison_bar.png` - Bar chart
- `comparison_results/comparison_radar.png` - Radar chart
- `comparison_results/comparison_report.md` - Summary report

## Configuration

Edit `shared/config.py` to change:
- Data paths
- Window size (default: 120s)
- Prediction horizons (default: 3, 5, 10 minutes)
- Target label (default: `label_5min`)

