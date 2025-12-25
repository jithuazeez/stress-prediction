"""
Temporal Convolutional Network (TCN) for stress classification.

Based on: https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
Paper: https://arxiv.org/pdf/1803.01271.pdf

Architecture:
- Dilated causal 1D convolutions with residual blocks
- Weight normalization and dropout for regularization
- Maintains same input/output length through causal padding

Key features:
- Causal convolutions (no future information leakage)
- Exponentially growing receptive field via dilations
- Residual connections for deep networks
- Flexible for both univariate and multivariate time series

For our use case:
- Input: (batch, n_features, seq_len) where seq_len=1 for feature vectors
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
    
    Stack of temporal blocks with exponentially growing dilations.
    
    Receptive field grows exponentially:
    - receptive_field = 1 + 2 * (kernel_size - 1) * (dilation_base^num_levels - 1) / (dilation_base - 1)
    
    Example with kernel_size=3, dilation_base=2, num_levels=4:
    - Level 0: dilation=1, receptive_field=3
    - Level 1: dilation=2, receptive_field=7
    - Level 2: dilation=4, receptive_field=15
    - Level 3: dilation=8, receptive_field=31
    """
    def __init__(self,
                 num_inputs: int,
                 num_channels: List[int],
                 kernel_size: int = 3,
                 dilation_base: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            num_inputs: Number of input channels (features)
            num_channels: List of channel sizes for each level
            kernel_size: Size of convolutional kernel
            dilation_base: Base for exponential dilation growth
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding to ensure same output length
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, num_inputs, seq_len)
        
        Returns:
            Tensor of shape (batch, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNClassifier(nn.Module):
    """
    TCN-based binary classifier for stress prediction.
    
    Architecture:
        Input (n_features, seq_len) -> TCN blocks -> Global pooling -> FC layers -> Output (num_classes)
    
    For feature-based input (seq_len=1):
    - Each feature is treated as a separate channel
    - TCN learns temporal patterns (though with seq_len=1, mainly feature interactions)
    - Global pooling reduces to per-sample representation
    - FC layers perform final classification
    """
    def __init__(self,
                 num_inputs: int,
                 num_channels: List[int] = [64, 64, 64],
                 num_classes: int = 2,
                 kernel_size: int = 3,
                 dilation_base: int = 2,
                 dropout: float = 0.2,
                 fc_hidden_dim: int = 128):
        """
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes for TCN blocks
            num_classes: Number of output classes
            kernel_size: Size of convolutional kernel
            dilation_base: Base for exponential dilation
            dropout: Dropout probability
            fc_hidden_dim: Hidden dimension for FC layers
        """
        super(TCNClassifier, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout
        )
        
        # Global pooling to aggregate temporal information
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
        
        # Global pooling
        # (batch, num_channels[-1], seq_len) -> (batch, num_channels[-1], 1)
        pooled = self.global_pool(tcn_out)
        
        # Flatten
        # (batch, num_channels[-1], 1) -> (batch, num_channels[-1])
        pooled = pooled.squeeze(-1)
        
        # Classification head
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    def get_receptive_field(self, kernel_size: int, dilation_base: int) -> int:
        """
        Calculate the receptive field size of the TCN.
        
        Args:
            kernel_size: Kernel size used
            dilation_base: Dilation base used
        
        Returns:
            Receptive field size
        """
        num_levels = len(self.num_channels)
        receptive_field = 1 + 2 * (kernel_size - 1) * (dilation_base**num_levels - 1) / (dilation_base - 1)
        return int(receptive_field)


def create_tcn_model(num_inputs: int,
                     num_classes: int = 2,
                     num_channels: List[int] = None,
                     kernel_size: int = 3,
                     dilation_base: int = 2,
                     dropout: float = 0.2,
                     fc_hidden_dim: int = 128) -> TCNClassifier:
    """
    Create a TCN classifier with default or custom architecture.
    
    Args:
        num_inputs: Number of input features
        num_classes: Number of output classes
        num_channels: List of channel sizes for TCN blocks (default: [64, 64, 64])
        kernel_size: Size of convolutional kernel
        dilation_base: Base for exponential dilation
        dropout: Dropout probability
        fc_hidden_dim: Hidden dimension for FC layers
    
    Returns:
        TCNClassifier model
    """
    if num_channels is None:
        num_channels = [64, 64, 64]
    
    model = TCNClassifier(
        num_inputs=num_inputs,
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        dilation_base=dilation_base,
        dropout=dropout,
        fc_hidden_dim=fc_hidden_dim
    )
    
    return model


if __name__ == "__main__":
    # Test TCN model
    print("Testing TCN model...")
    
    # Create model
    num_features = 57  # Approx number of features (acc + temp + heatflux)
    model = create_tcn_model(
        num_inputs=num_features,
        num_classes=2,
        num_channels=[32, 32, 32],
        kernel_size=3,
        dilation_base=2,
        dropout=0.2
    )
    
    print(f"\nModel architecture:")
    print(f"  Input features: {num_features}")
    print(f"  TCN channels: {model.num_channels}")
    print(f"  Output classes: {model.num_classes}")
    print(f"  Receptive field: {model.get_receptive_field(3, 2)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 16
    seq_len = 1  # For feature vectors
    x = torch.randn(batch_size, num_features, seq_len)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Output (first 3 samples):")
    print(logits[:3])
    
    print("\n✅ TCN model working!")

