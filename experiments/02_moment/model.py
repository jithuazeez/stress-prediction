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
    Supports selective layer unfreezing for fine-tuning.
    
    Unfreezing Strategies:
    - freeze_backbone=True, unfreeze_last_n_blocks=0: Only train head (safest)
    - freeze_backbone=True, unfreeze_last_n_blocks=2: Train head + last 2 transformer blocks (recommended)
    - freeze_backbone=True, unfreeze_last_n_blocks=4: Train head + last 4 blocks (more data needed)
    - freeze_backbone=False: Full fine-tuning (risk of overfitting with small data)
    """
    
    def __init__(self,
                 n_channels: int = 4,
                 num_classes: int = 2,
                 model_name: str = "AutonLab/MOMENT-1-large",
                 freeze_backbone: bool = False,
                 unfreeze_last_n_blocks: int = 0):
        """
        Initialize MOMENT classifier.
        
        Args:
            n_channels: Number of input channels (default 4)
            num_classes: Number of output classes (default 2: stress/no-stress)
            model_name: HuggingFace model name
            freeze_backbone: Whether to freeze the MOMENT backbone initially
            unfreeze_last_n_blocks: Number of transformer blocks to unfreeze from the end
                                   (only applies when freeze_backbone=True)
                                   - 0: Only train classification head
                                   - 2: Train head + last 2 transformer blocks (RECOMMENDED)
                                   - 4: Train head + last 4 blocks
                                   - 12: Equivalent to full fine-tuning
        """
        super().__init__()
        
        if not MOMENT_AVAILABLE:
            raise ImportError("momentfm not installed. Install with: pip install momentfm")
        
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        
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
        
        # Apply freezing strategy
        if freeze_backbone:
            self._freeze_with_selective_unfreezing(unfreeze_last_n_blocks)
    
    def _freeze_with_selective_unfreezing(self, unfreeze_last_n: int = 0):
        """
        Freeze backbone with selective unfreezing of last N transformer blocks.
        
        MOMENT-1-large has 12 transformer blocks (numbered 0-11).
        Earlier blocks learn generic time series features.
        Later blocks learn more task-specific features.
        
        Args:
            unfreeze_last_n: Number of blocks to keep unfrozen from the end
        """
        # First, freeze everything
        for param in self.moment.parameters():
            param.requires_grad = False
        
        # Track what we're unfreezing
        unfrozen_params = []
        frozen_params = []
        
        for name, param in self.moment.named_parameters():
            should_unfreeze = False
            
            # Always unfreeze classification head
            if any(x in name.lower() for x in ['head', 'class', 'classifier']):
                should_unfreeze = True
            
            # Unfreeze last N transformer blocks
            # Block names typically contain "encoder.layers.X" or "blocks.X"
            if unfreeze_last_n > 0:
                for block_idx in range(12 - unfreeze_last_n, 12):
                    if f'.{block_idx}.' in name or f'layers.{block_idx}' in name or f'blocks.{block_idx}' in name:
                        should_unfreeze = True
                        break
            
            if should_unfreeze:
                param.requires_grad = True
                unfrozen_params.append(name)
            else:
                frozen_params.append(name)
        
        # Store for logging
        self._unfrozen_params = unfrozen_params
        self._frozen_params = frozen_params
    
    def get_trainable_params_info(self) -> dict:
        """Get information about trainable vs frozen parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_pct": 100 * trainable_params / total_params,
            "unfrozen_layers": getattr(self, '_unfrozen_params', []),
        }
    
    def _freeze_backbone(self):
        """Legacy method - freeze the MOMENT backbone parameters."""
        self._freeze_with_selective_unfreezing(unfreeze_last_n=0)
    
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
                        freeze_backbone: bool = True,
                        unfreeze_last_n_blocks: int = 0) -> nn.Module:
    """
    Factory function to create MOMENT classifier.
    
    Args:
        n_channels: Number of input channels
        num_classes: Number of output classes
        use_simple: Whether to use simple embedding-based classifier
        freeze_backbone: Whether to freeze MOMENT backbone initially
        unfreeze_last_n_blocks: Number of transformer blocks to unfreeze
                               (RECOMMENDED: 2 for ~1000 samples)
    
    Unfreezing Guide:
        - 0 blocks: ~2K trainable params, lowest risk (current default)
        - 2 blocks: ~5M trainable params, good balance (RECOMMENDED)
        - 4 blocks: ~10M trainable params, needs more data
        - All (12): ~125M trainable params, high overfit risk
    
    Returns:
        MOMENT classifier model
    """
    if use_simple:
        return SimpleMOMENTClassifier(n_channels, num_classes)
    else:
        return MOMENTClassifier(
            n_channels=n_channels, 
            num_classes=num_classes, 
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )


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

