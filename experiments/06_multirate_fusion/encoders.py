"""
Modality-specific encoders for Multi-Rate Late Fusion.

UPDATED: Now uses TCN-based encoders instead of CNN encoders for better temporal modeling.

Each encoder processes signals at their native sampling rate (Same 8 channels as MOMENT):
- ACCEncoder: 32Hz input, 3 channels (3840 samples for 120s) - acc_x, acc_y, acc_z -> TCN-based
- PhysioEncoder: 1Hz input, 5 channels (120 samples for 120s) - skin_temp, heatflux, cbt, hr_bpm, rmssd -> TCN-based

Note: PPG encoder is disabled. Using HR and HRV (RMSSD) instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, List


# =============================================================================
# TCN Building Blocks (from experiments/tcn/model.py)
# =============================================================================

class Chomp1d(nn.Module):
    """
    Removes padding from the end of the sequence to ensure causality.
    
    Causal convolution requires padding on the left side only.
    PyTorch's Conv1d pads symmetrically, so we remove right padding.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, channels, seq_len)
        
        Returns:
            Tensor with chomp_size removed from the end
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Basic building block of TCN: two dilated causal convolutions with residual connection.
    
    Structure:
        Input -> Conv1 -> ReLU -> Dropout -> Conv2 -> ReLU -> Dropout -> (+) -> ReLU -> Output
                                                                           |
                                                                        Residual
    
    Features:
    - Weight normalization for better gradient flow
    - Spatial dropout for regularization
    - Residual connection (with 1x1 conv if needed for dimension matching)
    """
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2):
        """
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            dilation: Dilation factor for dilated convolution
            padding: Padding size
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Sequential composition of layers
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection (1x1 conv if channel dimensions differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, n_inputs, seq_len)
        
        Returns:
            Tensor of shape (batch, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN).
    
    Stack of temporal blocks with custom dilation pattern.
    
    Receptive field calculation:
    - receptive_field = 1 + sum((kernel_size - 1) * dilation for each layer)
    """
    def __init__(self,
                 num_inputs: int,
                 num_channels: List[int],
                 kernel_size: int = 3,
                 dilations: List[int] = None,
                 dropout: float = 0.2):
        """
        Args:
            num_inputs: Number of input channels
            num_channels: List of channel sizes for each level
            kernel_size: Size of convolutional kernel
            dilations: List of dilation factors
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8]  # Default for smaller sequences
        
        assert len(num_channels) == len(dilations), \
            f"num_channels ({len(num_channels)}) must match dilations ({len(dilations)})"
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding to ensure same output length
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.dilations = dilations
        self.kernel_size = kernel_size
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, num_inputs, seq_len)
        
        Returns:
            Tensor of shape (batch, num_channels[-1], seq_len)
        """
        return self.network(x)
    
    def get_receptive_field(self) -> int:
        """Calculate receptive field based on dilations."""
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


# =============================================================================
# TCN-Based Encoders for Multi-Rate Fusion
# =============================================================================

class ACCEncoder(nn.Module):
    """
    TCN-based encoder for 3-axis accelerometer at 32Hz.
    
    Processes 3 channels simultaneously to capture movement patterns.
    Uses TCN for better temporal modeling than CNN.
    
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
        Initialize TCN-based ACC encoder.
        
        Args:
            input_samples: Expected input length per channel
            n_channels: Number of input channels (3 for xyz)
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_samples = input_samples
        self.embedding_dim = embedding_dim
        
        # TCN backbone for high-rate signal (32Hz, ~3840 samples for 120s)
        # Need large receptive field to cover significant portion of 120s window
        # Dilations [1, 2, 4, 8, 16, 32, 64, 128] with kernel=7
        # RF = 1 + 6*(1+2+4+8+16+32+64+128) = 1 + 6*255 = 1531 samples
        # Coverage: 1531/32Hz ≈ 48 seconds ≈ 40% of window ✓
        tcn_channels = [16, 16, 32, 32, 64, 64, 128, 128]
        self.tcn = TemporalConvNet(
            num_inputs=n_channels,
            num_channels=tcn_channels,
            kernel_size=7,
            dilations=[1, 2, 4, 8, 16, 32, 64, 128],
            dropout=dropout
        )
        
        # Projection to embedding (from last TCN channel to embedding_dim)
        self.fc = nn.Linear(tcn_channels[-1], embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode accelerometer signal using TCN.
        
        Args:
            x: Input tensor of shape (batch, 3, acc_samples)
        
        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        # TCN processing
        tcn_out = self.tcn(x)  # (batch, 128, acc_samples)
        
        # Take last timestep (preserves causality)
        last_timestep = tcn_out[:, :, -1]  # (batch, 128)
        
        # Projection to embedding
        emb = self.dropout(last_timestep)
        emb = self.fc(emb)
        
        return emb


class PhysioEncoder(nn.Module):
    """
    TCN-based encoder for multi-channel physiological signals at 1Hz.
    
    Processes 5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd
    These are slow-changing signals, so we use a lightweight TCN.
    
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
        Initialize TCN-based physiological encoder.
        
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
        
        # TCN for 1Hz physiological signals (120 samples for 120s)
        # Need large RF to cover most of the 120-second window
        # Dilations [1, 2, 4, 8, 16, 32] with kernel=5
        # RF = 1 + 4*(1+2+4+8+16+32) = 1 + 4*63 = 253 samples
        # But input is only 120 samples, so RF effectively covers FULL window ✓
        # Note: When RF > input length, TCN sees entire sequence
        tcn_channels = [16, 32, 32, 64, 64, 128]
        tcn_dilations = [1, 2, 4, 8, 16, 32]
        
        self.tcn = TemporalConvNet(
            num_inputs=n_channels,
            num_channels=tcn_channels,
            kernel_size=5,
            dilations=tcn_dilations,
            dropout=dropout
        )
        
        # Projection to embedding (from last TCN channel to embedding_dim)
        self.fc = nn.Linear(tcn_channels[-1], embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode physiological signals using TCN.
        
        Args:
            x: Input tensor of shape (batch, 5, physio_samples)
        
        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        # TCN processing
        tcn_out = self.tcn(x)  # (batch, 128, physio_samples)
        
        # Take last timestep (preserves causality)
        last_timestep = tcn_out[:, :, -1]  # (batch, 128)
        
        # Projection to embedding
        emb = self.dropout(last_timestep)
        emb = self.fc(emb)
        
        return emb


# =============================================================================
# OLD CNN ENCODERS (COMMENTED OUT - kept for reference)
# =============================================================================

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


# class ACCEncoder(nn.Module):
#     """
#     OLD: 1D CNN encoder for 3-axis accelerometer at 32Hz.
#     REPLACED WITH: TCN-based encoder above
#     
#     Processes 3 channels simultaneously to capture movement patterns.
#     
#     Input: (batch, 3, acc_samples) where acc_samples = 3840 for 120s at 32Hz
#     Output: (batch, embedding_dim)
#     """
#     
#     def __init__(
#         self,
#         input_samples: int = 3840,
#         n_channels: int = 3,
#         embedding_dim: int = 128,
#         dropout: float = 0.2
#     ):
#         """
#         Initialize ACC encoder.
#         
#         Args:
#             input_samples: Expected input length per channel
#             n_channels: Number of input channels (3 for xyz)
#             embedding_dim: Output embedding dimension
#             dropout: Dropout rate
#         """
#         super().__init__()
#         
#         self.input_samples = input_samples
#         self.embedding_dim = embedding_dim
#         
#         # Conv layers with downsampling
#         # 3840 -> 960 -> 240 -> 60 -> pooled
#         self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=32, stride=4, padding=14)
#         self.bn1 = nn.BatchNorm1d(32)
#         
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=4, padding=6)
#         self.bn2 = nn.BatchNorm1d(64)
#         
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2)
#         self.bn3 = nn.BatchNorm1d(128)
#         
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(128, embedding_dim)
#     
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Encode accelerometer signal.
#         
#         Args:
#             x: Input tensor of shape (batch, 3, acc_samples)
#         
#         Returns:
#             Embedding of shape (batch, embedding_dim)
#         """
#         # x is already (batch, 3, acc_samples)
#         
#         # Convolutional layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
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


# class PhysioEncoder(nn.Module):
#     """
#     OLD: 1D CNN encoder for multi-channel physiological signals at 1Hz.
#     REPLACED WITH: TCN-based encoder above
#     
#     Processes 5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd
#     These are slow-changing signals, so we use a lightweight CNN.
#     
#     Input: (batch, 5, physio_samples) where physio_samples = 120 for 120s at 1Hz
#     Output: (batch, embedding_dim)
#     """
#     
#     def __init__(
#         self,
#         input_samples: int = 120,
#         n_channels: int = 5,
#         embedding_dim: int = 128,
#         dropout: float = 0.2
#     ):
#         """
#         Initialize physiological encoder.
#         
#         Args:
#             input_samples: Expected input length per channel
#             n_channels: Number of input channels (5: skin_temp, heatflux, cbt, hr_bpm, rmssd)
#             embedding_dim: Output embedding dimension
#             dropout: Dropout rate
#         """
#         super().__init__()
#         
#         self.input_samples = input_samples
#         self.n_channels = n_channels
#         self.embedding_dim = embedding_dim
#         
#         # Lightweight CNN for 1Hz signals (120 samples)
#         # 120 -> 60 -> 30 -> pooled
#         self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=8, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm1d(32)
#         
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(128, embedding_dim)
#     
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Encode physiological signals.
#         
#         Args:
#             x: Input tensor of shape (batch, 5, physio_samples)
#         
#         Returns:
#             Embedding of shape (batch, embedding_dim)
#         """
#         # x is already (batch, 5, physio_samples)
#         
#         # Convolutional layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
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


