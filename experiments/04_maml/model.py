"""
Base model for MAML meta-learning.

Defines a simple MLP classifier that can be wrapped with MAML.
Following the architecture from the original MAML paper.

References:
- https://arxiv.org/pdf/1703.03400
- https://github.com/cbfinn/maml
- https://github.com/learnables/learn2learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressClassifier(nn.Module):
    """
    Simple MLP classifier for stress prediction.
    
    Designed to work with MAML - uses few parameters for fast adaptation.
    
    Architecture follows the MAML paper guidelines:
    - Small network to enable fast adaptation
    - ReLU activations
    - No batch normalization (interferes with MAML)
    """
    
    def __init__(self, 
                 input_dim: int = 38,
                 hidden_dims: list = [64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.2):
        """
        Initialize classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
        
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class ConvStressClassifier(nn.Module):
    """
    1D CNN classifier for time series input.
    
    Alternative to MLP that works directly on raw signals.
    Inspired by the CNN architecture in the MAML paper.
    """
    
    def __init__(self,
                 n_channels: int = 4,
                 seq_len: int = 120,
                 num_classes: int = 2):
        """
        Initialize CNN classifier.
        
        Args:
            n_channels: Number of input channels
            seq_len: Sequence length
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        
        # Calculate flattened dimension
        # After 3 pooling layers: seq_len / 8
        flat_dim = 64 * (seq_len // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
        
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten and FC
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def create_maml_model(model_type: str = "mlp",
                      input_dim: int = 38,
                      n_channels: int = 4,
                      seq_len: int = 120,
                      num_classes: int = 2) -> nn.Module:
    """
    Factory function to create MAML-compatible model.
    
    Args:
        model_type: "mlp" or "cnn"
        input_dim: Input dimension for MLP
        n_channels: Number of channels for CNN
        seq_len: Sequence length for CNN
        num_classes: Number of output classes
    
    Returns:
        Model instance
    """
    if model_type == "mlp":
        return StressClassifier(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_classes=num_classes,
            dropout=0.0  # No dropout for MAML
        )
    elif model_type == "cnn":
        return ConvStressClassifier(
            n_channels=n_channels,
            seq_len=seq_len,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing MAML models...")
    
    # Test MLP
    mlp = StressClassifier(input_dim=38, num_classes=2)
    x_mlp = torch.randn(4, 38)
    out_mlp = mlp(x_mlp)
    print(f"MLP output shape: {out_mlp.shape}")
    
    # Test CNN
    cnn = ConvStressClassifier(n_channels=4, seq_len=120, num_classes=2)
    x_cnn = torch.randn(4, 4, 120)
    out_cnn = cnn(x_cnn)
    print(f"CNN output shape: {out_cnn.shape}")
    
    # Count parameters
    mlp_params = sum(p.numel() for p in mlp.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())
    print(f"\nMLP parameters: {mlp_params:,}")
    print(f"CNN parameters: {cnn_params:,}")

