# Temporal Convolutional Network (TCN) for Stress Classification

This experiment implements a Temporal Convolutional Network (TCN) for stress prediction from physiological signals.

## Overview

**Key Innovation**: TCN with on-the-fly feature extraction, combining the benefits of:
- Classical ML feature engineering (interpretable, domain-specific features)
- Deep learning architecture (automatic pattern learning, non-linear relationships)

## Architecture

### TCN Structure

Based on the architecture from [Unit8's article](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/) and the paper [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf).

```
Input (n_features, 1) 
  ↓
Temporal Block 1 (dilation=1)
  ├─ Conv1D → Chomp → ReLU → Dropout
  ├─ Conv1D → Chomp → ReLU → Dropout
  └─ Residual connection
  ↓
Temporal Block 2 (dilation=2)
  ├─ Conv1D → Chomp → ReLU → Dropout
  ├─ Conv1D → Chomp → ReLU → Dropout
  └─ Residual connection
  ↓
Temporal Block 3 (dilation=4)
  ├─ Conv1D → Chomp → ReLU → Dropout
  ├─ Conv1D → Chomp → ReLU → Dropout
  └─ Residual connection
  ↓
Global Average Pooling
  ↓
FC1 (128) → ReLU → Dropout
  ↓
FC2 (2) → Softmax
  ↓
Output (stress probabilities)
```

### Key Components

1. **Dilated Causal Convolutions**
   - Exponentially growing receptive field (dilation_base^level)
   - No future information leakage (causal padding)
   - Efficient long-range dependency modeling

2. **Residual Connections**
   - Enable training of deep networks
   - Gradient flow improvement
   - Feature reuse across layers

3. **Weight Normalization**
   - Reparameterization for better optimization
   - Faster convergence than batch normalization
   - More stable for small batches

4. **Regularization**
   - Spatial dropout after each convolution
   - Dropout in FC layers
   - Prevents overfitting

## Features

Unlike MOMENT which uses raw time series, TCN uses **extracted features** similar to classical ML:

### Feature Groups (~57-65 features, excluding HR/HRV):

1. **Accelerometer Features (41)**
   - Stress indicators: jerk, tremors, variability
   - Activity classification: stationary, walking, high activity
   - Statistical features: mean, std, min, max, range, IQR
   - Frequency domain: dominant frequency, power spectral density

2. **Temperature Features (7)**
   - Skin temperature: mean, std, min, max, range
   - Temporal: slope, change rate

3. **Heat Flux Features (9)**
   - Heat flux: mean, std, min, max, range, change
   - Core body temperature: mean, std, change

**Note**: HR/HRV features are excluded to match MOMENT's feature set, allowing fair comparison.

## Training

### LOSO Cross-Validation

Leave-One-Subject-Out (LOSO) ensures no data leakage:
- Train on N-1 subjects, test on 1 held-out subject
- Repeat for all subjects
- Evaluates generalization to unseen subjects

### Hyperparameters

**Model Architecture**:
- TCN channels: [64, 64, 64]
- Kernel size: 3
- Dilation base: 2
- Receptive field: 31 time steps
- Dropout: 0.3
- FC hidden dimension: 128

**Training**:
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 50 (with early stopping)
- Loss: Cross-entropy with class weights

**Threshold Selection**:
- Method: Geometric mean (G-Mean)
- Computed on training data per fold (no leakage)
- Maximizes sqrt(sensitivity × specificity)

## Usage

### Quick Start

```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/experiments/08_tcn
python train.py
```

### Custom Configuration

```python
from pathlib import Path
from shared.config import Config
from train import loso_cross_validation, load_all_windows
from shared.logging_utils import setup_logger
import torch

# Custom config
config = Config(
    window_size_sec=120,
    overlap_ratio=0.5,
    target_label="label_5min"
)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger("tcn_custom")
results_dir = Path("results_custom")

# Load data
windows_by_subject = load_all_windows(config, logger)

# Train with custom hyperparameters
results = loso_cross_validation(
    windows_by_subject,
    config,
    device,
    logger,
    n_epochs=100,
    batch_size=64,
    learning_rate=5e-4,
    tcn_channels=[128, 128, 128],
    kernel_size=5,
    dilation_base=2,
    dropout=0.4,
    fc_hidden_dim=256
)
```

## Results Structure

```
results/
├── training.log                 # Training logs
├── tcn_metrics.json            # Aggregate metrics
├── tcn_predictions.csv         # All predictions
├── tcn_fold_metrics.csv        # Per-fold metrics
├── checkpoints/
│   ├── best_model.pt           # Best model (highest G-Mean)
│   ├── fold_1_subject_*.pt     # Fold-specific checkpoints
│   └── ...
├── figures/
│   ├── tcn_roc.png            # ROC curve
│   ├── tcn_pr.png             # Precision-Recall curve
│   └── tcn_confusion_matrix.png
└── config/
    └── training_config.json    # Hyperparameters
```

## Key Metrics

Model evaluation uses:

1. **Threshold-Independent**:
   - AUROC: Area under ROC curve
   - PR-AUC: Area under Precision-Recall curve

2. **Classification Performance**:
   - G-Mean: √(sensitivity × specificity)
   - F1-Score: Harmonic mean of precision and recall
   - Balanced Accuracy: (sensitivity + specificity) / 2

3. **Clinical Relevance**:
   - Sensitivity (Recall): True positive rate
   - Specificity: True negative rate
   - Precision: Positive predictive value

## Comparison with Other Models

### vs. Classical ML
- **TCN Advantages**: 
  - Non-linear feature interactions
  - Automatic pattern learning
  - Better handling of complex relationships
- **Classical ML Advantages**:
  - More interpretable
  - Faster training
  - Less prone to overfitting with small data

### vs. MOMENT
- **TCN Advantages**:
  - Uses domain-specific features (interpretable)
  - Smaller model (fewer parameters)
  - Faster training
- **MOMENT Advantages**:
  - Pre-trained on large time series corpus
  - Can work with raw signals
  - Transfer learning from general patterns

### vs. MAML
- **TCN Advantages**:
  - Simpler architecture
  - Standard training (no meta-learning complexity)
  - More stable training
- **MAML Advantages**:
  - Explicit few-shot learning
  - Better generalization to new subjects
  - Meta-learned initialization

## References

1. [Temporal Convolutional Networks and Forecasting - Unit8](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)
2. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf) - Shaojie Bai, J. Zico Kolter, Vladlen Koltun (2018)
3. [VitaStress Dataset](https://github.com/eth-siplab/VitaStress)

## Implementation Details

### Feature Extraction
- On-the-fly extraction from windowed data
- Same pipeline as classical ML (excluding HR/HRV)
- Z-score normalization across training set

### Model Details
- Input shape: (batch, n_features, 1)
- Each feature is a "channel" for TCN
- Global pooling aggregates across temporal dimension
- Two FC layers for final classification

### Training Strategy
- Early stopping (patience=15 epochs)
- Class-weighted loss for imbalance
- Threshold optimized per fold on training data
- Model checkpointing (best + every 5 folds)

## Troubleshooting

### Import Errors
```python
# Add parent directories to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Memory Issues
- Reduce batch size: `batch_size=16`
- Reduce TCN channels: `tcn_channels=[32, 32, 32]`
- Reduce FC hidden dim: `fc_hidden_dim=64`

### Overfitting
- Increase dropout: `dropout=0.5`
- Reduce model capacity: fewer/smaller channels
- Early stopping (already implemented)

### Poor Performance
- Check feature normalization
- Verify class weights are computed correctly
- Try different threshold methods: "youden", "f1", "balanced"
- Increase model capacity if underfitting

## Citation

If you use this implementation, please cite:

```bibtex
@article{bai2018tcn,
  title={An empirical evaluation of generic convolutional and recurrent networks for sequence modeling},
  author={Bai, Shaojie and Kolter, J Zico and Koltun, Vladlen},
  journal={arXiv preprint arXiv:1803.01271},
  year={2018}
}

@article{vitastress2024,
  title={VitaStress: A Naturalistic Stress Detection Dataset using Wearable Sensors},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This code is for research purposes only.

