"""
MAML-compatible models for stress prediction.

Following the MAML paper (Finn et al., 2017) guidelines:
- Small networks for fast adaptation (few parameters)
- NO batch normalization (interferes with per-task adaptation)
- ReLU activations
- Designed to work with learn2learn's MAML wrapper

References:
- https://arxiv.org/pdf/1703.03400 (MAML paper)
- https://github.com/learnables/learn2learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressClassifier(nn.Module):
    """
    MLP classifier for stress prediction with MAML.
    
    Architecture based on MAML paper guidelines:
    - 2 hidden layers with 64 units each
    - ReLU activations
    - NO batch normalization (critical for MAML)
    - NO dropout during meta-training (can interfere)
    
    Input: Extracted statistical features (~61 features)
    Output: 2-class logits (no stress, stress)
    """
    
    def __init__(self, 
                 input_dim: int = 61,
                 hidden_dim: int = 64,
                 n_classes: int = 2):
        """
        Initialize classifier.
        
        Args:
            input_dim: Number of input features (default 61 from BasicFeatureExtractor)
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes (2 for binary stress classification)
        """
        super().__init__()
        
        # Store for cloning
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Simple 2-layer MLP (MAML paper recommends small networks)
        # NO BatchNorm - it interferes with per-task adaptation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
        
        Returns:
            Logits tensor of shape (batch, n_classes)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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
    1D CNN classifier for time series input with MAML.
    
    Designed for raw signal input (8 channels × 120 timesteps).
    Uses 4 conv layers following the MAML paper's image classifier architecture.
    
    8 channels: acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd
    (Same as MOMENT and SSL experiments for consistency)
    
    NO batch normalization (critical for MAML).
    """
    
    def __init__(self,
                 n_channels: int = 8,
                 seq_len: int = 120,
                 n_classes: int = 2,
                 hidden_filters: int = 32):
        """
        Initialize CNN classifier.
        
        Args:
            n_channels: Number of input channels (8: acc_x, acc_y, acc_z, 
                       skin_temp, heatflux, cbt, hr_bpm, rmssd)
            seq_len: Sequence length (120 for 2-min window at 1Hz)
            n_classes: Number of output classes
            hidden_filters: Number of filters in conv layers
        """
        super().__init__()
        
        # Store for cloning
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.hidden_filters = hidden_filters
        
        # 4 conv blocks (similar to MAML paper's 4-layer convnet)
        # Kernel size 3, stride 2 for downsampling (no max pooling)
        # NO BatchNorm - critical for MAML
        self.conv1 = nn.Conv1d(n_channels, hidden_filters, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_filters, hidden_filters, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_filters, hidden_filters, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(hidden_filters, hidden_filters, kernel_size=3, stride=2, padding=1)
        
        # Calculate output dimension after 4 stride-2 convs
        # 120 -> 60 -> 30 -> 15 -> 8 (approximately)
        self.flat_dim = hidden_filters * (seq_len // 16 + 1)
        
        # Final classifier
        self.fc = nn.Linear(self.flat_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
        
        Returns:
            Logits tensor of shape (batch, n_classes)
        """
        # Conv layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_maml_model(model_type: str = "mlp",
                      input_dim: int = 61,
                      n_channels: int = 8,
                      seq_len: int = 120,
                      n_classes: int = 2,
                      hidden_dim: int = 64) -> nn.Module:
    """
    Factory function to create MAML-compatible model.
    
    Args:
        model_type: "mlp" (uses extracted features) or "cnn" (uses raw signals)
        input_dim: Input dimension for MLP (default 61 features)
        n_channels: Number of channels for CNN (default 8: acc_x, acc_y, acc_z,
                   skin_temp, heatflux, cbt, hr_bpm, rmssd - same as MOMENT/SSL)
        seq_len: Sequence length for CNN
        n_classes: Number of output classes
        hidden_dim: Hidden dimension for MLP
    
    Returns:
        Model instance (without learn2learn wrapper - add that in training)
    """
    if model_type == "mlp":
        return StressClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes
        )
    elif model_type == "cnn":
        return ConvStressClassifier(
            n_channels=n_channels,
            seq_len=seq_len,
            n_classes=n_classes,
            hidden_filters=hidden_dim // 2  # Smaller for CNN
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'cnn'")


if __name__ == "__main__":
    print("Testing MAML-compatible models...")
    print("=" * 60)
    
    # Test MLP with extracted features
    print("\n1. MLP with extracted features (61 dims):")
    mlp = StressClassifier(input_dim=61, hidden_dim=64, n_classes=2)
    x_mlp = torch.randn(8, 61)  # batch of 8
    out_mlp = mlp(x_mlp)
    print(f"   Input shape: {x_mlp.shape}")
    print(f"   Output shape: {out_mlp.shape}")
    
    # Test CNN with raw signals
    print("\n2. CNN with raw signals (8 channels × 120 timesteps):")
    cnn = ConvStressClassifier(n_channels=8, seq_len=120, n_classes=2)
    x_cnn = torch.randn(8, 8, 120)  # batch of 8, 8 channels
    out_cnn = cnn(x_cnn)
    print(f"   Input shape: {x_cnn.shape}")
    print(f"   Output shape: {out_cnn.shape}")
    
    # Count parameters
    mlp_params = sum(p.numel() for p in mlp.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())
    print(f"\n3. Parameter counts (smaller = faster adaptation):")
    print(f"   MLP parameters: {mlp_params:,}")
    print(f"   CNN parameters: {cnn_params:,}")
    
    # Test gradient flow (important for MAML)
    print("\n4. Testing gradient flow for MAML:")
    mlp.train()
    x = torch.randn(4, 61, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1])
    
    logits = mlp(x)
    loss = F.cross_entropy(logits, y)
    
    # Compute gradients with create_graph=True (needed for MAML)
    grads = torch.autograd.grad(loss, mlp.parameters(), create_graph=True)
    print(f"   Number of gradient tensors: {len(grads)}")
    print(f"   Gradients have grad_fn (for 2nd order): {all(g.grad_fn is not None for g in grads)}")
    
    print("\n✅ Models are MAML-compatible!")
    print("   - No BatchNorm (which would interfere with per-task adaptation)")
    print("   - Small parameter count for fast adaptation")
    print("   - Gradients support second-order derivatives")
