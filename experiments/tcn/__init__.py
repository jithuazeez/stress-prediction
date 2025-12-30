"""
Temporal Convolutional Network (TCN) for stress classification.

UPDATED: Now uses raw multivariate time series (matching MOMENT architecture).

This experiment uses TCN with 8 sensor channels as raw input sequences:
- acc_x, acc_y, acc_z: Accelerometer
- skin_temp, heatflux, cbt: Thermal sensors
- hr_bpm, rmssd: Heart rate and HRV

Architecture:
- Input: 8 channels × 120 timesteps
- TCN channels: [16, 16, 16, 16, 16, 16]
- Dilations: [1, 2, 4, 8, 16, 32] (receptive field = 127)
- Pooling: Last timestep (maintains causality)
- Subject-wise normalization

References:
- https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
- https://arxiv.org/pdf/1803.01271.pdf

Modules:
- dataset.py: PyTorch dataset with raw multivariate sequences
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

