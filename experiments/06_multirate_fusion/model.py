"""
Multi-Rate Late Fusion Model.

Combines embeddings from modality-specific encoders using late fusion:
1. Each encoder processes its signal at native rate
2. Embeddings are concatenated
3. Fusion layer combines information
4. Classifier predicts stress

Architecture:
    PPG (64Hz) → PPGEncoder → 128-d
    ACC (32Hz) → ACCEncoder → 128-d  →  Concat (320-d) → Fusion → Classifier
    Temp (1Hz) → TempEncoder → 64-d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from encoders import PPGEncoder, ACCEncoder, TempEncoder, create_encoders, count_parameters
from config import MultiRateConfig


class FusionLayer(nn.Module):
    """
    Fusion layer that combines modality embeddings.
    
    Input: Concatenated embeddings (batch, total_embed_dim)
    Output: Fused representation (batch, output_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 320,
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize fusion layer.
        
        Args:
            input_dim: Total dimension after concatenation
            output_dim: Fused representation dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuse concatenated embeddings.
        
        Args:
            x: Concatenated embeddings (batch, input_dim)
        
        Returns:
            Fused representation (batch, output_dim)
        """
        return self.net(x)


class Classifier(nn.Module):
    """
    Classification head for stress prediction.
    
    Input: Fused representation (batch, input_dim)
    Output: Class logits (batch, n_classes)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        n_classes: int = 2
    ):
        """
        Initialize classifier.
        
        Args:
            input_dim: Input dimension
            n_classes: Number of output classes
        """
        super().__init__()
        
        self.fc = nn.Linear(input_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits.
        
        Args:
            x: Fused representation (batch, input_dim)
        
        Returns:
            Logits (batch, n_classes)
        """
        return self.fc(x)


class MultiRateFusionModel(nn.Module):
    """
    Complete Multi-Rate Late Fusion model.
    
    Processes each modality at native rate with separate encoders,
    then fuses embeddings for classification.
    """
    
    def __init__(
        self,
        config: MultiRateConfig,
        ppg_encoder: Optional[PPGEncoder] = None,
        acc_encoder: Optional[ACCEncoder] = None,
        temp_encoder: Optional[TempEncoder] = None
    ):
        """
        Initialize fusion model.
        
        Args:
            config: Multi-rate configuration
            ppg_encoder: Optional pre-created PPG encoder
            acc_encoder: Optional pre-created ACC encoder
            temp_encoder: Optional pre-created Temp encoder
        """
        super().__init__()
        
        self.config = config
        
        # Create or use provided encoders
        if ppg_encoder is None or acc_encoder is None or temp_encoder is None:
            ppg_enc, acc_enc, temp_enc = create_encoders(
                window_size_sec=config.window_size_sec,
                ppg_rate=config.ppg_sample_rate,
                acc_rate=config.acc_sample_rate,
                temp_rate=config.temp_sample_rate,
                ppg_dim=config.ppg_embedding_dim,
                acc_dim=config.acc_embedding_dim,
                temp_dim=config.temp_embedding_dim,
                dropout=config.dropout
            )
            self.ppg_encoder = ppg_enc
            self.acc_encoder = acc_enc
            self.temp_encoder = temp_enc
        else:
            self.ppg_encoder = ppg_encoder
            self.acc_encoder = acc_encoder
            self.temp_encoder = temp_encoder
        
        # Total embedding dimension after concatenation
        total_embed_dim = (
            config.ppg_embedding_dim +
            config.acc_embedding_dim +
            config.temp_embedding_dim
        )
        
        # Fusion and classifier
        self.fusion = FusionLayer(
            input_dim=total_embed_dim,
            output_dim=config.fusion_dim,
            dropout=config.dropout
        )
        
        self.classifier = Classifier(
            input_dim=config.fusion_dim,
            n_classes=2
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through all encoders and fusion.
        
        Args:
            batch: Dictionary with 'ppg', 'acc', 'temp' tensors
        
        Returns:
            Class logits (batch, 2)
        """
        # Encode each modality
        ppg_emb = self.ppg_encoder(batch["ppg"])
        acc_emb = self.acc_encoder(batch["acc"])
        temp_emb = self.temp_encoder(batch["temp"])
        
        # Concatenate embeddings
        combined = torch.cat([ppg_emb, acc_emb, temp_emb], dim=-1)
        
        # Fuse and classify
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits
    
    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get individual and combined embeddings.
        
        Args:
            batch: Dictionary with 'ppg', 'acc', 'temp' tensors
        
        Returns:
            Dictionary with embeddings for each modality and fused
        """
        ppg_emb = self.ppg_encoder(batch["ppg"])
        acc_emb = self.acc_encoder(batch["acc"])
        temp_emb = self.temp_encoder(batch["temp"])
        
        combined = torch.cat([ppg_emb, acc_emb, temp_emb], dim=-1)
        fused = self.fusion(combined)
        
        return {
            "ppg": ppg_emb,
            "acc": acc_emb,
            "temp": temp_emb,
            "combined": combined,
            "fused": fused
        }
    
    def predict_proba(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            batch: Dictionary with 'ppg', 'acc', 'temp' tensors
        
        Returns:
            Probabilities (batch, 2)
        """
        logits = self.forward(batch)
        return F.softmax(logits, dim=-1)


def create_model(config: MultiRateConfig) -> MultiRateFusionModel:
    """
    Factory function to create Multi-Rate Fusion model.
    
    Args:
        config: Multi-rate configuration
    
    Returns:
        Configured MultiRateFusionModel
    """
    return MultiRateFusionModel(config)


if __name__ == "__main__":
    # Test model
    print("Testing Multi-Rate Fusion Model...")
    
    from config import DEFAULT_MULTIRATE_CONFIG
    
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        (120, "120s window"),
        (60, "60s window")
    ]
    
    for window_size, name in configs:
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        
        config = DEFAULT_MULTIRATE_CONFIG.get_config_for_run(window_size, 3)
        model = create_model(config)
        
        # Create dummy batch
        batch_size = 4
        batch = {
            "ppg": torch.randn(batch_size, config.ppg_samples_per_window),
            "acc": torch.randn(batch_size, 3, config.acc_samples_per_window),
            "temp": torch.randn(batch_size, config.temp_samples_per_window),
            "label": torch.randint(0, 2, (batch_size,)),
            "subject_id": torch.randint(0, 5, (batch_size,))
        }
        
        print(f"\nInput shapes:")
        print(f"  PPG:  {batch['ppg'].shape}")
        print(f"  ACC:  {batch['acc'].shape}")
        print(f"  Temp: {batch['temp'].shape}")
        
        # Forward pass
        logits = model(batch)
        probs = model.predict_proba(batch)
        embeddings = model.get_embeddings(batch)
        
        print(f"\nOutput shapes:")
        print(f"  Logits: {logits.shape}")
        print(f"  Probs:  {probs.shape}")
        
        print(f"\nEmbedding shapes:")
        for k, v in embeddings.items():
            print(f"  {k}: {v.shape}")
        
        # Parameter counts
        print(f"\nParameter counts:")
        print(f"  PPG Encoder:  {count_parameters(model.ppg_encoder):,}")
        print(f"  ACC Encoder:  {count_parameters(model.acc_encoder):,}")
        print(f"  Temp Encoder: {count_parameters(model.temp_encoder):,}")
        print(f"  Fusion:       {count_parameters(model.fusion):,}")
        print(f"  Classifier:   {count_parameters(model.classifier):,}")
        print(f"  Total:        {count_parameters(model):,}")
    
    print("\nAll tests passed!")

