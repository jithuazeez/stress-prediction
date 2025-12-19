"""
Modality-specific encoders for Multi-Rate Late Fusion.

Each encoder processes signals at their native sampling rate:
- PPGEncoder: 64Hz input (7680 samples for 120s)
- ACCEncoder: 32Hz input, 3 channels (3840 samples for 120s)
- TempEncoder: 1Hz input (120 samples for 120s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PPGEncoder(nn.Module):
    """
    1D CNN encoder for PPG signals at 64Hz.
    
    Designed to capture cardiac waveform patterns and HRV information.
    Uses progressive downsampling to handle high-resolution input.
    
    Input: (batch, ppg_samples) where ppg_samples = 7680 for 120s at 64Hz
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_samples: int = 7680,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize PPG encoder.
        
        Args:
            input_samples: Expected input length (window_sec * 64)
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_samples = input_samples
        self.embedding_dim = embedding_dim
        
        # Conv layers with progressive downsampling
        # 7680 -> 960 -> 240 -> 60 -> 15 -> pooled
        self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=6)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, stride=4, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode PPG signal.
        
        Args:
            x: Input tensor of shape (batch, ppg_samples)
        
        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, ppg_samples)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.pool(x)
        x = x.squeeze(-1)  # (batch, 128)
        
        # Projection
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


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


class TempEncoder(nn.Module):
    """
    MLP encoder for temperature at 1Hz.
    
    Temperature is slow-changing, so a simple MLP is sufficient.
    
    Input: (batch, temp_samples) where temp_samples = 120 for 120s at 1Hz
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_samples: int = 120,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize temperature encoder.
        
        Args:
            input_samples: Expected input length
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_samples = input_samples
        self.embedding_dim = embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_samples, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode temperature signal.
        
        Args:
            x: Input tensor of shape (batch, temp_samples)
        
        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_encoders(
    window_size_sec: int = 120,
    ppg_rate: float = 64.0,
    acc_rate: float = 32.0,
    temp_rate: float = 1.0,
    ppg_dim: int = 128,
    acc_dim: int = 128,
    temp_dim: int = 64,
    dropout: float = 0.2
) -> tuple:
    """
    Factory function to create all encoders.
    
    Args:
        window_size_sec: Window size in seconds
        ppg_rate: PPG sampling rate
        acc_rate: ACC sampling rate
        temp_rate: Temperature sampling rate
        ppg_dim: PPG embedding dimension
        acc_dim: ACC embedding dimension
        temp_dim: Temperature embedding dimension
        dropout: Dropout rate
    
    Returns:
        Tuple of (PPGEncoder, ACCEncoder, TempEncoder)
    """
    ppg_encoder = PPGEncoder(
        input_samples=int(window_size_sec * ppg_rate),
        embedding_dim=ppg_dim,
        dropout=dropout
    )
    
    acc_encoder = ACCEncoder(
        input_samples=int(window_size_sec * acc_rate),
        embedding_dim=acc_dim,
        dropout=dropout
    )
    
    temp_encoder = TempEncoder(
        input_samples=int(window_size_sec * temp_rate),
        embedding_dim=temp_dim,
        dropout=dropout
    )
    
    return ppg_encoder, acc_encoder, temp_encoder


if __name__ == "__main__":
    # Test encoders
    print("Testing Multi-Rate Encoders...")
    
    torch.manual_seed(42)
    batch_size = 4
    
    # Test configurations
    configs = [
        {"window": 120, "ppg": 7680, "acc": 3840, "temp": 120},
        {"window": 60, "ppg": 3840, "acc": 1920, "temp": 60},
    ]
    
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Window: {cfg['window']}s")
        print(f"{'='*50}")
        
        # Create encoders
        ppg_enc, acc_enc, temp_enc = create_encoders(window_size_sec=cfg["window"])
        
        # Create dummy inputs
        ppg_input = torch.randn(batch_size, cfg["ppg"])
        acc_input = torch.randn(batch_size, 3, cfg["acc"])
        temp_input = torch.randn(batch_size, cfg["temp"])
        
        # Forward pass
        ppg_emb = ppg_enc(ppg_input)
        acc_emb = acc_enc(acc_input)
        temp_emb = temp_enc(temp_input)
        
        print(f"\nInput shapes:")
        print(f"  PPG:  {ppg_input.shape}")
        print(f"  ACC:  {acc_input.shape}")
        print(f"  Temp: {temp_input.shape}")
        
        print(f"\nOutput shapes:")
        print(f"  PPG:  {ppg_emb.shape}")
        print(f"  ACC:  {acc_emb.shape}")
        print(f"  Temp: {temp_emb.shape}")
        
        print(f"\nParameter counts:")
        print(f"  PPG:  {count_parameters(ppg_enc):,}")
        print(f"  ACC:  {count_parameters(acc_enc):,}")
        print(f"  Temp: {count_parameters(temp_enc):,}")
        print(f"  Total: {count_parameters(ppg_enc) + count_parameters(acc_enc) + count_parameters(temp_enc):,}")
    
    print("\nAll tests passed!")


