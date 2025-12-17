# VitaStress Multi-Model Stress Prediction Experiments

This directory contains 6 distinct modeling approaches for stress prediction from multimodal wearable sensor data.

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

# NEW: Subject-Aware SSL (pre-train then fine-tune)
python 05_subject_aware_ssl/pretrain.py --mode invariant --window_size 120
python 05_subject_aware_ssl/train.py --mode invariant --window_size 120 --horizon 3

# NEW: Multi-Rate Late Fusion
python 06_multirate_fusion/train.py --window_size 120 --horizon 3

# Run all configurations
python 05_subject_aware_ssl/run_all.py
python 06_multirate_fusion/run_all.py

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
├── 05_subject_aware_ssl/       # Subject-Aware Contrastive SSL (NEW)
│   ├── augmentations.py        # Temporal augmentations
│   ├── encoder.py              # 1D ResNet encoder
│   ├── losses.py               # InfoNCE, subject-invariant, subject-specific
│   ├── dataset.py              # Data loading at 8Hz
│   ├── config.py               # SSL configuration
│   ├── pretrain.py             # Self-supervised pre-training
│   ├── train.py                # Fine-tuning with LOSO
│   └── run_all.py              # Run all configurations
│
├── 06_multirate_fusion/        # Multi-Rate Late Fusion (NEW)
│   ├── encoders.py             # PPGEncoder, ACCEncoder, TempEncoder
│   ├── model.py                # MultiRateFusionModel
│   ├── dataset.py              # Native-rate data loading
│   ├── config.py               # Multi-rate configuration
│   ├── train.py                # Training with LOSO
│   └── run_all.py              # Run all configurations
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

### 5. Subject-Aware Contrastive SSL (05_subject_aware_ssl) - NEW

**Approach:** Self-supervised pre-training with subject awareness, then fine-tune

**Reference:** [Self-supervised learning of electrodermal activity representations](https://arxiv.org/abs/2301.06234) - Apple (2023)

**Three modes:**
- **Base SSL:** Standard InfoNCE contrastive loss
- **Subject-Invariant:** InfoNCE + Adversarial loss (for generalization)
- **Subject-Specific:** InfoNCE with same-subject negatives (for fine-tuning)

**Features:**
- 8Hz sampling rate (960 samples for 120s window)
- Conservative augmentations to preserve temporal patterns
- 1D ResNet encoder (~288K params)

### 6. Multi-Rate Late Fusion (06_multirate_fusion) - NEW

**Approach:** Process each modality at native sampling rate, then fuse

**Sampling rates:**
- PPG: 64 Hz (7680 samples for 120s)
- ACC: 32 Hz (3840 samples for 120s)
- Temp: 1 Hz (120 samples for 120s)

**Architecture:**
- Separate 1D CNN encoders per modality
- Late fusion of embeddings
- Classification head

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

