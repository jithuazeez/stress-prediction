# Experiment 07: TabPFN Foundation Model

## Overview

TabPFN is a foundation model for tabular classification, pre-trained on synthetic tabular datasets. Unlike traditional ML models, it requires **no hyperparameter tuning** and can make predictions in seconds.

This experiment applies TabPFN to the same engineered features used in classical ML (Experiment 01), enabling direct comparison between traditional ML and foundation model approaches.

## Key Features

- **Pre-trained**: Model trained on diverse synthetic tabular data
- **Zero hyperparameter tuning**: No grid search needed
- **Fast**: Predictions in seconds even on CPU
- **Handles imbalance**: Automatically handles class imbalance
- **Missing values**: Can handle NaN values directly (though we apply same preprocessing as classical ML for fair comparison)

## Installation

```bash
# Install TabPFN
pip install tabpfn

# Login to HuggingFace (required for model download)
huggingface-cli login

# Accept license at: https://huggingface.co/Prior-Labs/tabpfn_2_5
```

## Usage

```bash
# Run training
cd experiments/07_tabpfn
python train.py
```

The script will:
1. Load and preprocess raw sensor data (same as classical ML)
2. Extract ~65 features (accelerometer, temperature, HR/HRV, etc.)
3. Apply 2-stage quality filtering
4. Train TabPFN with LOSO cross-validation
5. Save results, predictions, and plots

## Comparison to Classical ML

| Aspect | Classical ML | TabPFN |
|--------|-------------|--------|
| Hyperparameter tuning | Required (nested CV) | Not needed |
| Scaling | Manual (StandardScaler) | Automatic |
| Class weights | Manual | Automatic |
| Training time per fold | ~30-60s | ~10-20s |
| Model size | Small (<1MB) | Large (~500MB) |
| Interpretability | High (feature importance) | Low (black-box) |

## Expected Performance

TabPFN should perform similarly to or better than XGBoost/Random Forest, especially on:
- Small datasets (LOSO with ~1000-2000 samples per fold)
- Imbalanced classes (~25% stress, 75% no stress)
- Mixed feature types (temporal, statistical, physiological)

## Architecture

```
Raw Sensor Data (PPG, ACC, Temp, etc.)
         ↓
    Feature Extraction
    (~65 features: ACC, Temp, HR/HRV)
         ↓
    Quality Filtering
    (Keep only high-quality HRV windows)
         ↓
    TabPFN Classifier
    (Pre-trained transformer)
         ↓
    Stress Prediction
```

## Features Used

Same as classical ML (Experiment 01):

1. **Accelerometer (41 features)**: Stress indicators + activity classification
   - Fidgeting, tremors, restlessness detection
   - Activity level (sitting vs. exercise)
   
2. **Temperature (7 features)**: Skin temperature statistics
   - Mean, std, min, max, range, slope, change

3. **Heat Flux (9 features)**: Thermal regulation
   - Heat flux + Core body temperature (CBT)

4. **HR/HRV from PPG (8 features)**: Cardiac measures
   - Heart rate: hr_bpm, hr_std
   - HRV time-domain: hrv_mean_rr, hrv_sdnn, hrv_rmssd, hrv_pnn50, hrv_pnn20, hrv_sdsd

## Limitations

- **Maximum 100 features**: We have ~65, so OK
- **Maximum 50,000 samples**: Our LOSO folds have ~1000-2000, so OK
- **Requires GPU for large datasets**: Optional for our size, but recommended
- **Large model download**: ~500MB on first use
- **Black-box**: Less interpretable than classical ML

## Results

Results are saved to `results/`:
- `tabpfn_metrics.json` - Overall metrics
- `tabpfn_fold_metrics.csv` - Per-fold metrics
- `tabpfn_predictions.csv` - All predictions
- `figures/` - ROC, PR curves, confusion matrix
- `training.log` - Detailed execution log

## Troubleshooting

### Error: "TabPFN not installed"
```bash
pip install tabpfn
```

### Error: "Access denied to model"
```bash
# Login to HuggingFace
huggingface-cli login

# Then accept license at:
# https://huggingface.co/Prior-Labs/tabpfn_2_5
```

### Error: "CUDA out of memory"
TabPFN will automatically fall back to CPU. You can also force CPU mode by setting `device="cpu"` in the training script.

### Warning: "Running on CPU will be slower"
This is expected. For our dataset size (~1000-2000 samples per fold), CPU should take ~10-30 seconds per fold, which is acceptable.

## References

- **Paper**: [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- **Nature Paper**: [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6)
- **GitHub**: https://github.com/PriorLabs/TabPFN
- **HuggingFace**: https://huggingface.co/Prior-Labs/tabpfn_2_5

## Citation

```bibtex
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 doi={10.1038/s41586-024-08328-6},
}
```

