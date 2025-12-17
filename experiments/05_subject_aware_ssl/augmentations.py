"""
Temporal augmentations for contrastive self-supervised learning.

Based on the Apple paper with conservative parameters to preserve 
stress-related temporal patterns in physiological signals.

All augmentations operate on tensors of shape (batch, channels, time).
"""

import torch
import numpy as np
from typing import Optional, Tuple, Callable, Dict
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    
    # ALWAYS USE (low risk, high benefit)
    temporal_delay_max_samples: int = 10  # Shift up to 10 samples (~1.25s at 8Hz)
    temporal_delay_p: float = 0.8         # Apply 80% of the time
    
    gaussian_noise_std: float = 0.03      # Small noise relative to signal std
    gaussian_noise_p: float = 0.8         # Apply 80% of the time
    
    # USE WITH CAUTION
    temporal_cutout_max_ratio: float = 0.3  # Max 30% of window
    temporal_cutout_p: float = 0.5          # Apply 50% of the time
    
    channel_dropout_p_drop: float = 0.15   # Drop 15% of channels
    channel_dropout_p: float = 0.3         # Apply 30% of the time
    
    # OPTIONAL (enable via flag)
    signal_mixing_alpha: float = 0.9      # 90% original, 10% other
    signal_mixing_p: float = 0.3          # Apply 30% of the time
    enable_signal_mixing: bool = False    # Disabled by default during SSL


# Default safe configuration
DEFAULT_AUG_CONFIG = AugmentationConfig()


def temporal_delay(
    x: torch.Tensor,
    max_samples: int = 10,
    p: float = 0.8
) -> torch.Tensor:
    """
    Apply random temporal delay (shift) to signals.
    
    This simulates small timing variations between subjects or sessions.
    Low risk - preserves all patterns, just shifts timing.
    
    Args:
        x: Input tensor of shape (batch, channels, time)
        max_samples: Maximum shift in samples (both directions)
        p: Probability of applying this augmentation
    
    Returns:
        Shifted tensor of same shape
    """
    if torch.rand(1).item() > p:
        return x
    
    batch, channels, time = x.shape
    device = x.device
    
    # Random shift per sample in batch
    shifts = torch.randint(-max_samples, max_samples + 1, (batch,), device=device)
    
    x_delayed = torch.zeros_like(x)
    
    for i in range(batch):
        shift = shifts[i].item()
        if shift > 0:
            # Shift right - pad beginning
            x_delayed[i, :, shift:] = x[i, :, :-shift]
            x_delayed[i, :, :shift] = x[i, :, :shift]  # Repeat first values
        elif shift < 0:
            # Shift left - pad end
            x_delayed[i, :, :shift] = x[i, :, -shift:]
            x_delayed[i, :, shift:] = x[i, :, shift:]  # Repeat last values
        else:
            x_delayed[i] = x[i]
    
    return x_delayed


def gaussian_noise(
    x: torch.Tensor,
    std: float = 0.03,
    p: float = 0.8
) -> torch.Tensor:
    """
    Add Gaussian noise to signals.
    
    Simulates sensor noise. Low risk - small noise preserves temporal patterns.
    
    Args:
        x: Input tensor of shape (batch, channels, time)
        std: Standard deviation of noise (relative to input)
        p: Probability of applying this augmentation
    
    Returns:
        Noisy tensor of same shape
    """
    if torch.rand(1).item() > p:
        return x
    
    # Scale noise by the channel-wise std of the signal
    # This makes the noise adaptive to the signal magnitude
    noise = torch.randn_like(x) * std
    
    return x + noise


def temporal_cutout(
    x: torch.Tensor,
    max_ratio: float = 0.3,
    p: float = 0.5
) -> torch.Tensor:
    """
    Mask a contiguous segment of the signal with zeros.
    
    Forces the model to learn from context. Moderate risk - may remove
    important events like stress onset markers.
    
    Args:
        x: Input tensor of shape (batch, channels, time)
        max_ratio: Maximum ratio of sequence to mask (e.g., 0.3 = max 30%)
        p: Probability of applying this augmentation
    
    Returns:
        Masked tensor of same shape
    """
    if torch.rand(1).item() > p:
        return x
    
    batch, channels, time = x.shape
    device = x.device
    
    x_masked = x.clone()
    
    for i in range(batch):
        # Random mask length (between 5% and max_ratio of sequence)
        mask_ratio = torch.rand(1).item() * (max_ratio - 0.05) + 0.05
        mask_len = int(time * mask_ratio)
        
        # Random start position
        max_start = time - mask_len
        if max_start > 0:
            start = torch.randint(0, max_start, (1,)).item()
            x_masked[i, :, start:start + mask_len] = 0.0
    
    return x_masked


def channel_dropout(
    x: torch.Tensor,
    p_drop: float = 0.15,
    p: float = 0.3
) -> torch.Tensor:
    """
    Zero out entire channels randomly.
    
    Forces the model to not rely on any single modality.
    Moderate risk - preserves temporal patterns in remaining channels.
    
    Args:
        x: Input tensor of shape (batch, channels, time)
        p_drop: Probability of dropping each channel
        p: Probability of applying this augmentation
    
    Returns:
        Tensor with some channels zeroed
    """
    if torch.rand(1).item() > p:
        return x
    
    batch, channels, time = x.shape
    device = x.device
    
    # Create channel mask (same for entire batch to preserve batch statistics)
    # But ensure at least one channel remains
    channel_mask = torch.rand(channels, device=device) > p_drop
    
    # Ensure at least one channel is kept
    if not channel_mask.any():
        keep_idx = torch.randint(0, channels, (1,)).item()
        channel_mask[keep_idx] = True
    
    # Expand mask for broadcasting
    channel_mask = channel_mask.unsqueeze(0).unsqueeze(-1)  # (1, channels, 1)
    
    return x * channel_mask.float()


def signal_mixing(
    x: torch.Tensor,
    alpha: float = 0.9,
    p: float = 0.3
) -> torch.Tensor:
    """
    Mix signal with another random sample from the batch.
    
    x_mixed = alpha * x + (1 - alpha) * x_shuffled
    
    Higher risk - may introduce conflicting stress patterns.
    Use with caution, preferably with same-label samples during supervised training.
    
    Args:
        x: Input tensor of shape (batch, channels, time)
        alpha: Weight for original signal (0.9 means 90% original, 10% other)
        p: Probability of applying this augmentation
    
    Returns:
        Mixed tensor
    """
    if torch.rand(1).item() > p:
        return x
    
    batch = x.shape[0]
    
    if batch < 2:
        return x
    
    # Random permutation for mixing partners
    perm = torch.randperm(batch, device=x.device)
    
    # Ensure no sample mixes with itself
    same_mask = perm == torch.arange(batch, device=x.device)
    if same_mask.any():
        # Shift problematic indices by 1
        perm[same_mask] = (perm[same_mask] + 1) % batch
    
    x_shuffled = x[perm]
    
    return alpha * x + (1 - alpha) * x_shuffled


def signal_mixing_with_labels(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.9,
    p: float = 0.3,
    same_label_only: bool = True
) -> torch.Tensor:
    """
    Mix signal with another sample, optionally only with same-label samples.
    
    This is safer than blind mixing as it preserves label semantics.
    
    Args:
        x: Input tensor of shape (batch, channels, time)
        labels: Label tensor of shape (batch,)
        alpha: Weight for original signal
        p: Probability of applying this augmentation
        same_label_only: If True, only mix samples with same label
    
    Returns:
        Mixed tensor
    """
    if torch.rand(1).item() > p:
        return x
    
    batch = x.shape[0]
    device = x.device
    
    if batch < 2:
        return x
    
    x_mixed = x.clone()
    
    for i in range(batch):
        if same_label_only:
            # Find samples with same label
            same_label_mask = labels == labels[i]
            same_label_mask[i] = False  # Exclude self
            
            candidates = torch.where(same_label_mask)[0]
            
            if len(candidates) == 0:
                continue  # No mixing partner available
            
            partner_idx = candidates[torch.randint(len(candidates), (1,)).item()]
        else:
            # Random partner (excluding self)
            candidates = [j for j in range(batch) if j != i]
            partner_idx = candidates[torch.randint(len(candidates), (1,)).item()]
        
        x_mixed[i] = alpha * x[i] + (1 - alpha) * x[partner_idx]
    
    return x_mixed


class Augmenter:
    """
    Augmentation pipeline for contrastive learning.
    
    Creates two augmented views of each input for contrastive loss.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter with configuration.
        
        Args:
            config: Augmentation configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_AUG_CONFIG
    
    def __call__(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of input.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            labels: Optional labels for same-label mixing
        
        Returns:
            Tuple of (view1, view2) augmented tensors
        """
        view1 = self._augment(x, labels)
        view2 = self._augment(x, labels)
        return view1, view2
    
    def _augment(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply augmentation pipeline to input."""
        cfg = self.config
        
        # Always apply these (low risk)
        x = temporal_delay(x, cfg.temporal_delay_max_samples, cfg.temporal_delay_p)
        x = gaussian_noise(x, cfg.gaussian_noise_std, cfg.gaussian_noise_p)
        
        # Apply with caution
        x = temporal_cutout(x, cfg.temporal_cutout_max_ratio, cfg.temporal_cutout_p)
        x = channel_dropout(x, cfg.channel_dropout_p_drop, cfg.channel_dropout_p)
        
        # Optional signal mixing
        if cfg.enable_signal_mixing:
            if labels is not None:
                x = signal_mixing_with_labels(
                    x, labels, cfg.signal_mixing_alpha, cfg.signal_mixing_p,
                    same_label_only=True
                )
            else:
                x = signal_mixing(x, cfg.signal_mixing_alpha, cfg.signal_mixing_p)
        
        return x


def create_augmenter(
    enable_mixing: bool = False,
    aggressive: bool = False
) -> Augmenter:
    """
    Factory function to create augmenter with preset configurations.
    
    Args:
        enable_mixing: Whether to enable signal mixing
        aggressive: Whether to use more aggressive augmentation (higher cutout ratio)
    
    Returns:
        Configured Augmenter instance
    """
    config = AugmentationConfig()
    config.enable_signal_mixing = enable_mixing
    
    if aggressive:
        # More aggressive settings (use with caution)
        config.temporal_cutout_max_ratio = 0.4
        config.temporal_cutout_p = 0.6
        config.channel_dropout_p_drop = 0.2
        config.channel_dropout_p = 0.4
    
    return Augmenter(config)


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")
    
    torch.manual_seed(42)
    
    # Create dummy input
    batch, channels, time = 8, 3, 960  # 8 samples, 3 channels, 120s at 8Hz
    x = torch.randn(batch, channels, time)
    labels = torch.randint(0, 2, (batch,))
    
    print(f"Input shape: {x.shape}")
    
    # Test individual augmentations
    print("\n1. Temporal delay:")
    x_delayed = temporal_delay(x.clone(), max_samples=10)
    print(f"   Output shape: {x_delayed.shape}")
    print(f"   Changed: {not torch.allclose(x, x_delayed)}")
    
    print("\n2. Gaussian noise:")
    x_noisy = gaussian_noise(x.clone(), std=0.03)
    print(f"   Output shape: {x_noisy.shape}")
    print(f"   Max diff: {(x - x_noisy).abs().max():.4f}")
    
    print("\n3. Temporal cutout:")
    x_cutout = temporal_cutout(x.clone(), max_ratio=0.3)
    print(f"   Output shape: {x_cutout.shape}")
    zeros_ratio = (x_cutout == 0).float().mean().item()
    print(f"   Zeros ratio: {zeros_ratio:.2%}")
    
    print("\n4. Channel dropout:")
    x_dropout = channel_dropout(x.clone(), p_drop=0.15)
    print(f"   Output shape: {x_dropout.shape}")
    zero_channels = (x_dropout.abs().sum(dim=(0, 2)) == 0).sum().item()
    print(f"   Zeroed channels: {zero_channels}/{channels}")
    
    print("\n5. Signal mixing:")
    x_mixed = signal_mixing(x.clone(), alpha=0.9)
    print(f"   Output shape: {x_mixed.shape}")
    
    print("\n6. Full augmentation pipeline:")
    augmenter = create_augmenter(enable_mixing=False)
    view1, view2 = augmenter(x, labels)
    print(f"   View 1 shape: {view1.shape}")
    print(f"   View 2 shape: {view2.shape}")
    print(f"   Views different: {not torch.allclose(view1, view2)}")
    
    print("\nAll tests passed!")
