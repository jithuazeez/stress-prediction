"""
1D ResNet Encoder for Subject-Aware Contrastive SSL.

Based on the Apple paper architecture with ~288K parameters.
Designed for physiological signals at 8Hz sampling rate.

Architecture:
    Input → Conv1D → [ResBlock] × 3 → GlobalPool → Embedding
    
Each ResBlock has skip connections to preserve temporal information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional downsampling.
    
    Structure:
        x → Conv1D → BN → ELU → Conv1D → BN → (+x) → ELU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        downsample: bool = False
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            stride: Stride for first convolution (for downsampling)
            downsample: Whether to apply downsampling via MaxPool
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection with optional channel projection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        # Optional downsampling
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output tensor
        """
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual
        out = out + identity
        out = F.elu(out)
        
        # Optional downsampling
        if self.downsample:
            out = self.pool(out)
        
        return out


class SSLEncoder(nn.Module):
    """
    1D ResNet Encoder for contrastive self-supervised learning.
    
    Architecture follows the Apple paper design (~288K params):
    - Input: (batch, n_channels, time) at 8Hz
    - Conv1D stem with large kernel (captures ~1.6s patterns)
    - 3 ResBlocks with progressive channel expansion and downsampling
    - Global average pooling
    - Output: 256-d embedding
    
    For 120s window at 8Hz: 960 timesteps → 256-d embedding
    For 60s window at 8Hz: 480 timesteps → 256-d embedding
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        embedding_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        kernel_sizes: Tuple[int, ...] = (13, 11, 9, 7),
        dropout: float = 0.1
    ):
        """
        Initialize encoder.
        
        Args:
            input_channels: Number of input channels (default 3 for multimodal)
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden channel dimensions for ResBlocks
            kernel_sizes: Kernel sizes for stem and ResBlocks
            dropout: Dropout rate before final projection
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        
        # Stem: Initial convolution with large kernel
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels, hidden_dims[0],
                kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, bias=False
            ),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ELU()
        )
        
        # ResBlocks with progressive downsampling
        self.resblock1 = ResBlock1D(
            hidden_dims[0], hidden_dims[0],
            kernel_size=kernel_sizes[1], downsample=False
        )
        
        self.resblock2 = ResBlock1D(
            hidden_dims[0], hidden_dims[1],
            kernel_size=kernel_sizes[2], downsample=True  # 4x downsample
        )
        
        self.resblock3 = ResBlock1D(
            hidden_dims[1], hidden_dims[2],
            kernel_size=kernel_sizes[3], downsample=True  # 4x downsample
        )
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dims[-1], embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get embeddings.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Stem
        h = self.stem(x)  # (batch, 32, time)
        
        # ResBlocks
        h = self.resblock1(h)  # (batch, 32, time)
        h = self.resblock2(h)  # (batch, 64, time/4)
        h = self.resblock3(h)  # (batch, 128, time/16)
        
        # Global pooling
        h = self.pool(h)  # (batch, 128, 1)
        h = h.squeeze(-1)  # (batch, 128)
        
        # Projection
        h = self.dropout(h)
        embedding = self.fc(h)  # (batch, embedding_dim)
        
        return embedding
    
    def get_intermediate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both intermediate representation and final embedding.
        
        Useful for adversarial training where we need pre-projection features.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Tuple of (intermediate, embedding) where:
            - intermediate: Pre-pooling features (batch, 128, time//16)
            - embedding: Final embedding (batch, embedding_dim)
        """
        # Stem
        h = self.stem(x)
        
        # ResBlocks
        h = self.resblock1(h)
        h = self.resblock2(h)
        h = self.resblock3(h)
        
        intermediate = h  # Before pooling
        
        # Pooling and projection
        h = self.pool(h)
        h = h.squeeze(-1)
        h = self.dropout(h)
        embedding = self.fc(h)
        
        return intermediate, embedding


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning (F in the paper).
    
    Maps embeddings to a lower-dimensional space for contrastive loss.
    This is discarded after pre-training.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input dimension (encoder embedding dim)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension for contrastive loss
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embedding to contrastive space.
        
        Args:
            x: Embedding tensor of shape (batch, input_dim)
        
        Returns:
            Projected tensor of shape (batch, output_dim)
        """
        return self.net(x)


class SubjectClassifier(nn.Module):
    """
    Subject classifier for adversarial training (subject-invariant mode).
    
    Tries to predict subject identity from embeddings.
    The encoder is trained to CONFUSE this classifier.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        n_subjects: int = 21
    ):
        """
        Initialize subject classifier.
        
        Args:
            input_dim: Input dimension (encoder embedding dim)
            n_subjects: Number of subjects to classify
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_subjects)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict subject identity.
        
        Args:
            x: Embedding tensor of shape (batch, input_dim)
        
        Returns:
            Logits tensor of shape (batch, n_subjects)
        """
        return self.net(x)


class ClassificationHead(nn.Module):
    """
    Classification head for downstream stress prediction.
    
    Added on top of pre-trained encoder for fine-tuning.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input dimension (encoder embedding dim)
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits.
        
        Args:
            x: Embedding tensor of shape (batch, input_dim)
        
        Returns:
            Logits tensor of shape (batch, n_classes)
        """
        return self.net(x)


class SSLModel(nn.Module):
    """
    Complete SSL model combining encoder and projection head.
    
    For pre-training:
        model = SSLModel(encoder, ProjectionHead())
        z = model(x)  # Projected embedding for contrastive loss
    
    For fine-tuning:
        model = SSLModel(encoder, ClassificationHead())
        logits = model(x)  # Class predictions
    """
    
    def __init__(
        self,
        encoder: SSLEncoder,
        head: nn.Module
    ):
        """
        Initialize SSL model.
        
        Args:
            encoder: Pre-trained or fresh encoder
            head: Projection head (pre-training) or classification head (fine-tuning)
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and head.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Head output (projected embedding or logits)
        """
        embedding = self.encoder(x)
        return self.head(embedding)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder embedding without head.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        return self.encoder(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_ssl_encoder(
    input_channels: int = 3,
    embedding_dim: int = 256,
    small: bool = False
) -> SSLEncoder:
    """
    Factory function to create SSL encoder.
    
    Args:
        input_channels: Number of input channels
        embedding_dim: Output embedding dimension
        small: Use smaller model (fewer parameters)
    
    Returns:
        Configured SSLEncoder instance
    """
    if small:
        # Smaller model (~100K params)
        return SSLEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            hidden_dims=(16, 32, 64),
            kernel_sizes=(9, 7, 5, 3)
        )
    else:
        # Standard model (~288K params)
        return SSLEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            hidden_dims=(32, 64, 128),
            kernel_sizes=(13, 11, 9, 7)
        )


if __name__ == "__main__":
    # Test encoder
    print("Testing SSL Encoder...")
    
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        {"name": "120s @ 8Hz", "batch": 4, "channels": 3, "time": 960},
        {"name": "60s @ 8Hz", "batch": 4, "channels": 3, "time": 480},
    ]
    
    encoder = create_ssl_encoder(input_channels=3, embedding_dim=256)
    n_params = count_parameters(encoder)
    print(f"\nEncoder parameters: {n_params:,}")
    
    for cfg in configs:
        x = torch.randn(cfg["batch"], cfg["channels"], cfg["time"])
        embedding = encoder(x)
        print(f"\n{cfg['name']}:")
        print(f"  Input:  {x.shape}")
        print(f"  Output: {embedding.shape}")
    
    # Test with projection head
    print("\n\nTesting full SSL model with projection head...")
    proj_head = ProjectionHead(input_dim=256, hidden_dim=128, output_dim=64)
    model = SSLModel(encoder, proj_head)
    
    x = torch.randn(8, 3, 960)
    z = model(x)
    print(f"Input:  {x.shape}")
    print(f"Projected: {z.shape}")
    
    # Test subject classifier
    print("\nTesting subject classifier...")
    subject_clf = SubjectClassifier(input_dim=256, n_subjects=21)
    embedding = encoder(x)
    subject_logits = subject_clf(embedding)
    print(f"Embedding: {embedding.shape}")
    print(f"Subject logits: {subject_logits.shape}")
    
    # Test classification head
    print("\nTesting classification head...")
    cls_head = ClassificationHead(input_dim=256, n_classes=2)
    cls_model = SSLModel(encoder, cls_head)
    logits = cls_model(x)
    print(f"Class logits: {logits.shape}")
    
    # Parameter counts
    print("\n\nParameter counts:")
    print(f"  Encoder:         {count_parameters(encoder):,}")
    print(f"  Projection head: {count_parameters(proj_head):,}")
    print(f"  Subject clf:     {count_parameters(subject_clf):,}")
    print(f"  Class head:      {count_parameters(cls_head):,}")
    print(f"  Total SSL:       {count_parameters(model):,}")
    
    print("\nAll tests passed!")
