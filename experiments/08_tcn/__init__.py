"""
Temporal Convolutional Network (TCN) for stress classification.

This experiment uses TCN architecture with on-the-fly feature extraction
(similar to classical ML, excluding HR/HRV to match MOMENT's feature set).

References:
- https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
- https://arxiv.org/pdf/1803.01271.pdf

Modules:
- dataset.py: PyTorch dataset with feature extraction
- model.py: TCN architecture implementation
- train.py: LOSO cross-validation training script
"""

from .model import TCNClassifier, create_tcn_model
from .dataset import VitaStressTCNDataset, create_tcn_datasets

__all__ = [
    'TCNClassifier',
    'create_tcn_model',
    'VitaStressTCNDataset',
    'create_tcn_datasets',
]

