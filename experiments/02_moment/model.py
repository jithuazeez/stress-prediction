"""
MOMENT model wrapper for stress classification.

Wraps the MOMENT foundation model pipeline for easy use in training.

References:
- https://github.com/moment-timeseries-foundation-model/moment
- Paper: "MOMENT: A Family of Open Time-series Foundation Models" (ICML 2024)
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn

# Try to import momentfm
try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False
    print("Warning: momentfm not installed. Install with: pip install momentfm")


class MOMENTClassifier(nn.Module):
    """
    MOMENT-based classifier for stress prediction.
    
    Uses the pre-trained MOMENT model and adds a classification head.
    """
    
    def __init__(self,
                 n_channels: int = 4,
                 num_classes: int = 2,
                 model_name: str = "AutonLab/MOMENT-1-large",
                 freeze_backbone: bool = False):
        """
        Initialize MOMENT classifier.
        
        Args:
            n_channels: Number of input channels (default 4)
            num_classes: Number of output classes (default 2: stress/no-stress)
            model_name: HuggingFace model name
            freeze_backbone: Whether to freeze the MOMENT backbone
        """
        super().__init__()
        
        if not MOMENT_AVAILABLE:
            raise ImportError("momentfm not installed. Install with: pip install momentfm")
        
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Initialize MOMENT pipeline for classification
        self.moment = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                "task_name": "classification",
                "n_channels": n_channels,
                "num_class": num_classes
            }
        )
        self.moment.init()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze the MOMENT backbone parameters."""
        # Manually freeze encoder and embedder parameters
        # Keep head (classification layer) trainable
        for name, param in self.moment.named_parameters():
            # Don't freeze head/classification parameters
            if 'head' not in name.lower() and 'class' not in name.lower():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
        
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Use task-specific method with keyword argument
        output = self.moment.classify(x_enc=x)
        return output.logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            x: Input tensor
        
        Returns:
            Predicted class labels
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted probabilities.
        
        Args:
            x: Input tensor
        
        Returns:
            Probability tensor of shape (batch, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


class SimpleMOMENTClassifier(nn.Module):
    """
    Simple fallback classifier using MOMENT embeddings.
    
    If direct classification doesn't work well, use MOMENT for
    embedding extraction and train a simple classifier on top.
    """
    
    def __init__(self,
                 n_channels: int = 4,
                 num_classes: int = 2,
                 embedding_dim: int = 1024,  # MOMENT-1-large outputs 1024-dim embeddings
                 hidden_dim: int = 128,
                 model_name: str = "AutonLab/MOMENT-1-large"):
        """
        Initialize simple classifier.
        
        Args:
            n_channels: Number of input channels
            num_classes: Number of output classes
            embedding_dim: MOMENT embedding dimension (1024 for MOMENT-1-large)
            hidden_dim: Hidden layer dimension
            model_name: HuggingFace model name
        """
        super().__init__()
        
        if not MOMENT_AVAILABLE:
            raise ImportError("momentfm not installed")
        
        self.n_channels = n_channels
        self.num_classes = num_classes
        
        # MOMENT for embedding
        self.moment = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"}
        )
        self.moment.init()
        
        # Freeze MOMENT using named_parameters directly
        for param in self.moment.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
        
        Returns:
            Logits tensor
        """
        # Get embeddings from MOMENT using task-specific method with keyword arg
        with torch.no_grad():
            output = self.moment.embed(x_enc=x)
            embeddings = output.embeddings  # Shape: (batch, embedding_dim)
        
        # Classify
        return self.classifier(embeddings)


def create_moment_model(n_channels: int = 4,
                        num_classes: int = 2,
                        use_simple: bool = False,
                        freeze_backbone: bool = True) -> nn.Module:
    """
    Factory function to create MOMENT classifier.
    
    Args:
        n_channels: Number of input channels
        num_classes: Number of output classes
        use_simple: Whether to use simple embedding-based classifier
        freeze_backbone: Whether to freeze MOMENT backbone
    
    Returns:
        MOMENT classifier model
    """
    if use_simple:
        return SimpleMOMENTClassifier(n_channels, num_classes)
    else:
        # logger.info(f"Creating MOMENTClassifier with n_channels={n_channels}, num_classes={num_classes}, freeze_backbone={freeze_backbone}")
        return MOMENTClassifier(n_channels, num_classes, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    print("Testing MOMENT model wrapper...")
    
    if not MOMENT_AVAILABLE:
        print("MOMENT not available. Install with: pip install momentfm")
    else:
        # Create model
        model = MOMENTClassifier(n_channels=4, num_classes=2, freeze_backbone=True)
        
        # Test forward pass
        x = torch.randn(2, 4, 512)  # batch=2, channels=4, seq_len=512
        
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        proba = model.predict_proba(x)
        print(f"Probabilities shape: {proba.shape}")
        print(f"Probabilities sum: {proba.sum(dim=-1)}")

