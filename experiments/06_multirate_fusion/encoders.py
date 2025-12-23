"""
Modality-specific encoders for Multi-Rate Late Fusion.

Each encoder processes signals at their native sampling rate (Same 8 channels as MOMENT):
- ACCEncoder: 32Hz input, 3 channels (3840 samples for 120s) - acc_x, acc_y, acc_z
- PhysioEncoder: 1Hz input, 5 channels (120 samples for 120s) - skin_temp, heatflux, cbt, hr_bpm, rmssd

Note: PPG encoder is disabled. Using HR and HRV (RMSSD) instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# # PPGEncoder - DISABLED (using HR and HRV instead)
# class PPGEncoder(nn.Module):
#     """
#     1D CNN encoder for PPG signals at 64Hz.
#     
#     Designed to capture cardiac waveform patterns and HRV information.
#     Uses progressive downsampling to handle high-resolution input.
#     
#     Input: (batch, ppg_samples) where ppg_samples = 7680 for 120s at 64Hz
#     Output: (batch, embedding_dim)
#     """
#     
#     def __init__(
#         self,
#         input_samples: int = 7680,
#         embedding_dim: int = 128,
#         dropout: float = 0.2
#     ):
#         """
#         Initialize PPG encoder.
#         
#         Args:
#             input_samples: Expected input length (window_sec * 64)
#             embedding_dim: Output embedding dimension
#             dropout: Dropout rate
#         """
#         super().__init__()
#         
#         self.input_samples = input_samples
#         self.embedding_dim = embedding_dim
#         
#         # Conv layers with progressive downsampling
#         # 7680 -> 960 -> 240 -> 60 -> 15 -> pooled
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28)
#         self.bn1 = nn.BatchNorm1d(32)
#         
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14)
#         self.bn2 = nn.BatchNorm1d(64)
#         
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=6)
#         self.bn3 = nn.BatchNorm1d(128)
#         
#         self.conv4 = nn.Conv1d(128, 128, kernel_size=8, stride=4, padding=2)
#         self.bn4 = nn.BatchNorm1d(128)
#         
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(128, embedding_dim)
#     
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Encode PPG signal.
#         
#         Args:
#             x: Input tensor of shape (batch, ppg_samples)
#         
#         Returns:
#             Embedding of shape (batch, embedding_dim)
#         """
#         # Add channel dimension
#         x = x.unsqueeze(1)  # (batch, 1, ppg_samples)
#         
#         # Convolutional layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         
#         # Global pooling
#         x = self.pool(x)
#         x = x.squeeze(-1)  # (batch, 128)
#         
#         # Projection
#         x = self.dropout(x)
#         x = self.fc(x)
#         
#         return x


class ACCEncoder(nn.Module):
    """
    1D CNN encoder for 3-axis accelerometer at 32Hz.
    
    Processes 3 channels simultaneously to capture movement patterns.
    
    Input: (batch, 3, acc_samples) where acc_samples = 3840 for 120s at 32Hz
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_samples: int = 3840,
        n_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize ACC encoder.
        
        Args:
            input_samples: Expected input length per channel
            n_channels: Number of input channels (3 for xyz)
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_samples = input_samples
        self.embedding_dim = embedding_dim
        
        # Conv layers with downsampling
        # 3840 -> 960 -> 240 -> 60 -> pooled
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=32, stride=4, padding=14)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=4, padding=6)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode accelerometer signal.
        
        Args:
            x: Input tensor of shape (batch, 3, acc_samples)
        
        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        # x is already (batch, 3, acc_samples)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x)
        x = x.squeeze(-1)  # (batch, 128)
        
        # Projection
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class PhysioEncoder(nn.Module):
    """
    1D CNN encoder for multi-channel physiological signals at 1Hz.
    
    Processes 5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd
    These are slow-changing signals, so we use a lightweight CNN.
    
    Input: (batch, 5, physio_samples) where physio_samples = 120 for 120s at 1Hz
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_samples: int = 120,
        n_channels: int = 5,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize physiological encoder.
        
        Args:
            input_samples: Expected input length per channel
            n_channels: Number of input channels (5: skin_temp, heatflux, cbt, hr_bpm, rmssd)
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_samples = input_samples
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        
        # Lightweight CNN for 1Hz signals (120 samples)
        # 120 -> 60 -> 30 -> pooled
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=8, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode physiological signals.
        
        Args:
            x: Input tensor of shape (batch, 5, physio_samples)
        
        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        # x is already (batch, 5, physio_samples)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x)
        x = x.squeeze(-1)  # (batch, 128)
        
        # Projection
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_encoders(
    window_size_sec: int = 120,
    acc_rate: float = 32.0,
    physio_rate: float = 1.0,
    acc_dim: int = 128,
    physio_dim: int = 128,
    dropout: float = 0.2
) -> tuple:
    """
    Factory function to create all encoders.
    
    Args:
        window_size_sec: Window size in seconds
        acc_rate: ACC sampling rate (32Hz)
        physio_rate: Physiological signals sampling rate (1Hz)
        acc_dim: ACC embedding dimension
        physio_dim: Physiological embedding dimension
        dropout: Dropout rate
    
    Returns:
        Tuple of (ACCEncoder, PhysioEncoder)
    """
    acc_encoder = ACCEncoder(
        input_samples=int(window_size_sec * acc_rate),
        n_channels=3,
        embedding_dim=acc_dim,
        dropout=dropout
    )
    
    physio_encoder = PhysioEncoder(
        input_samples=int(window_size_sec * physio_rate),
        n_channels=5,
        embedding_dim=physio_dim,
        dropout=dropout
    )
    
    return acc_encoder, physio_encoder


if __name__ == "__main__":
    # Test encoders
    print("Testing Multi-Rate Encoders...")
    
    torch.manual_seed(42)
    batch_size = 4
    
    # Test configurations
    configs = [
        {"window": 120, "acc": 3840, "physio": 120},
        {"window": 60, "acc": 1920, "physio": 60},
    ]
    
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Window: {cfg['window']}s")
        print(f"{'='*50}")
        
        # Create encoders
        acc_enc, physio_enc = create_encoders(window_size_sec=cfg["window"])
        
        # Create dummy inputs
        acc_input = torch.randn(batch_size, 3, cfg["acc"])
        physio_input = torch.randn(batch_size, 5, cfg["physio"])
        
        # Forward pass
        acc_emb = acc_enc(acc_input)
        physio_emb = physio_enc(physio_input)
        
        print(f"\nInput shapes:")
        print(f"  ACC:    {acc_input.shape}")
        print(f"  Physio: {physio_input.shape}")
        
        print(f"\nOutput shapes:")
        print(f"  ACC:    {acc_emb.shape}")
        print(f"  Physio: {physio_emb.shape}")
        
        print(f"\nParameter counts:")
        print(f"  ACC:    {count_parameters(acc_enc):,}")
        print(f"  Physio: {count_parameters(physio_enc):,}")
        print(f"  Total:  {count_parameters(acc_enc) + count_parameters(physio_enc):,}")
    
    print("\nAll tests passed!")


