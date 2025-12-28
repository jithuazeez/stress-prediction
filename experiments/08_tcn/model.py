"""
Temporal Convolutional Network (TCN) for stress classification.

UPDATED: Now matches MOMENT-style architecture with raw multivariate sequences.

Based on: https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
Paper: https://arxiv.org/pdf/1803.01271.pdf

Architecture:
- Dilated causal 1D convolutions with residual blocks
- Weight normalization and dropout for regularization
- Maintains same input/output length through causal padding
- Uses last timestep instead of global pooling for classification

Key features:
- Causal convolutions (no future information leakage)
- Custom dilation pattern [1, 2, 4, 8, 16, 32] for receptive field = 127
- Residual connections for deep networks
- Multivariate time series input (8 channels × 120 timesteps)

For our use case:
- Input: (batch, n_channels=8, seq_len=120) raw sensor data
- Output: (batch, num_classes) for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from typing import List, Tuple


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
    
    Example with kernel_size=3, dilations=[1, 2, 4, 8, 16, 32]:
    - receptive_field = 1 + 2*(1 + 2 + 4 + 8 + 16 + 32) = 1 + 2*63 = 127
    """
    def __init__(self,
                 num_inputs: int,
                 num_channels: List[int],
                 kernel_size: int = 3,
                 dilations: List[int] = None,
                 dropout: float = 0.2):
        """
        Args:
            num_inputs: Number of input channels (e.g., 8 for multivariate sensors)
            num_channels: List of channel sizes for each level
            kernel_size: Size of convolutional kernel
            dilations: List of dilation factors (default: [1, 2, 4, 8, 16, 32])
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]  # Custom pattern for RF=127
        
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


class TCNClassifier(nn.Module):
    """
    TCN-based binary classifier for stress prediction.
    
    Architecture:
        Input (n_channels, seq_len) -> TCN blocks -> Last timestep -> FC layers -> Output (num_classes)
    
    Updated design:
    - Uses raw multivariate sequences (8 channels × 120 timesteps)
    - Custom dilation pattern [1, 2, 4, 8, 16, 32] for receptive field = 127
    - Narrower channels [16, 16, 16, 16, 16, 16] to reduce parameters
    - Uses LAST TIMESTEP instead of global average pooling
    - Maintains temporal causality for real-time prediction
    """
    def __init__(self,
                 num_inputs: int = 8,
                 num_channels: List[int] = None,
                 num_classes: int = 2,
                 kernel_size: int = 3,
                 dilations: List[int] = None,
                 dropout: float = 0.2,
                 fc_hidden_dim: int = 128,
                 use_last_timestep: bool = True):
        """
        Args:
            num_inputs: Number of input channels (default 8 for sensors)
            num_channels: List of channel sizes for TCN blocks (default: [16]*6)
            num_classes: Number of output classes
            kernel_size: Size of convolutional kernel
            dilations: List of dilation factors (default: [1, 2, 4, 8, 16, 32])
            dropout: Dropout probability
            fc_hidden_dim: Hidden dimension for FC layers
            use_last_timestep: If True, use last timestep; else use global avg pooling
        """
        super(TCNClassifier, self).__init__()
        
        if num_channels is None:
            num_channels = [16, 16, 16, 16, 16, 16]  # Narrower for reduced parameters
        
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]  # Custom dilations for RF=127
        
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.use_last_timestep = use_last_timestep
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout
        )
        
        # Pooling strategy
        if not use_last_timestep:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(num_channels[-1], fc_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize FC layer weights."""
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, num_inputs, seq_len)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # TCN encoding
        # x: (batch, num_inputs, seq_len) -> (batch, num_channels[-1], seq_len)
        tcn_out = self.tcn(x)
        
        if self.use_last_timestep:
            # Use last timestep (maintains causality)
            # (batch, num_channels[-1], seq_len) -> (batch, num_channels[-1])
            pooled = tcn_out[:, :, -1]
        else:
            # Global average pooling
            # (batch, num_channels[-1], seq_len) -> (batch, num_channels[-1], 1)
            pooled = self.global_pool(tcn_out)
            # Flatten: (batch, num_channels[-1], 1) -> (batch, num_channels[-1])
            pooled = pooled.squeeze(-1)
        
        # Classification head
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field size of the TCN."""
        return self.tcn.get_receptive_field()


def create_tcn_model(num_inputs: int = 8,
                     num_classes: int = 2,
                     num_channels: List[int] = None,
                     kernel_size: int = 3,
                     dilations: List[int] = None,
                     dropout: float = 0.2,
                     fc_hidden_dim: int = 128,
                     use_last_timestep: bool = True) -> TCNClassifier:
    """
    Create a TCN classifier with default or custom architecture.
    
    Default configuration matches requirements:
    - num_inputs: 8 (multivariate sensors)
    - num_channels: [16, 16, 16, 16, 16, 16] (narrower than before)
    - dilations: [1, 2, 4, 8, 16, 32] (receptive field = 127)
    - use_last_timestep: True (instead of global pooling)
    
    Args:
        num_inputs: Number of input channels (default 8)
        num_classes: Number of output classes
        num_channels: List of channel sizes for TCN blocks (default: [16]*6)
        kernel_size: Size of convolutional kernel
        dilations: List of dilation factors (default: [1, 2, 4, 8, 16, 32])
        dropout: Dropout probability
        fc_hidden_dim: Hidden dimension for FC layers
        use_last_timestep: If True, use last timestep; else use global avg pooling
    
    Returns:
        TCNClassifier model
    """
    if num_channels is None:
        num_channels = [16, 16, 16, 16, 16, 16]
    
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]
    
    model = TCNClassifier(
        num_inputs=num_inputs,
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        fc_hidden_dim=fc_hidden_dim,
        use_last_timestep=use_last_timestep
    )
    
    return model


if __name__ == "__main__":
    # Test TCN model
    print("Testing TCN model...")
    
    # Create model with new architecture
    num_channels_list = 8  # 8 sensor channels
    model = create_tcn_model(
        num_inputs=num_channels_list,
        num_classes=2,
        num_channels=[16, 16, 16, 16, 16, 16],
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
        dropout=0.2,
        use_last_timestep=True
    )
    
    print(f"\nModel architecture:")
    print(f"  Input channels: {num_channels_list}")
    print(f"  TCN channels: {model.num_channels}")
    print(f"  Dilations: [1, 2, 4, 8, 16, 32]")
    print(f"  Output classes: {model.num_classes}")
    print(f"  Receptive field: {model.get_receptive_field()}")
    print(f"  Pooling: {'Last timestep' if model.use_last_timestep else 'Global average'}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 16
    seq_len = 120  # 120 seconds at 1Hz
    x = torch.randn(batch_size, num_channels_list, seq_len)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Output (first 3 samples):")
    print(logits[:3])
    
    print("\n✅ TCN model working!")

