"""
Self-supervised pretraining script for TS2Vec.

Pretrains encoder on all data using contrastive learning (no labels).

References:
- https://github.com/zhihanyue/ts2vec
- https://arxiv.org/pdf/2106.10466
"""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import time
import pickle

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_to_1hz
from shared.windowing import create_labeled_windows, parse_stress_events
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import (
    setup_logger, log_experiment_start, log_experiment_end
)

from dataset import VitaStressTS2VecDataset


class TS2VecEncoder(torch.nn.Module):
    """
    Simple 1D CNN encoder for TS2Vec-style contrastive learning.
    
    Architecture:
    - 3 convolutional layers with dilated convolutions
    - Captures multi-scale temporal patterns
    """
    
    def __init__(self, 
                 input_channels: int = 4,
                 output_dim: int = 64,
                 hidden_dim: int = 64):
        super().__init__()
        
        # Temporal convolutions with increasing dilation
        self.conv1 = torch.nn.Conv1d(input_channels, hidden_dim, kernel_size=3, 
                                      padding=1, dilation=1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, 
                                      padding=2, dilation=2)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, 
                                      padding=4, dilation=4)
        
        # Output projection
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Encoded representation of shape (batch, time, output_dim)
        """
        # x: (batch, channels, time)
        h = F.gelu(self.conv1(x))
        h = self.dropout(h)
        
        h = F.gelu(self.conv2(h))
        h = self.dropout(h)
        
        h = F.gelu(self.conv3(h))
        
        # h: (batch, hidden_dim, time) -> (batch, time, hidden_dim)
        h = h.permute(0, 2, 1)
        
        # Project to output dimension
        out = self.fc(h)  # (batch, time, output_dim)
        
        return out


def hierarchical_contrastive_loss(z1: torch.Tensor, 
                                   z2: torch.Tensor, 
                                   temporal_unit: int = 0,
                                   temperature: float = 0.5) -> torch.Tensor:
    """
    Hierarchical contrastive loss from TS2Vec paper.
    
    Computes instance-level and temporal-level contrastive objectives.
    
    Args:
        z1, z2: Encoded representations from two augmented views (batch, time, dim)
        temporal_unit: Minimum temporal aggregation unit (0 for timestamp level)
        temperature: Softmax temperature
    
    Returns:
        Combined loss
    """
    batch_size, seq_len, dim = z1.shape
    
    # Instance-level contrast (compare entire sequences)
    # Pool over time dimension
    z1_pooled = z1.mean(dim=1)  # (batch, dim)
    z2_pooled = z2.mean(dim=1)  # (batch, dim)
    
    # Normalize
    z1_pooled = F.normalize(z1_pooled, dim=-1)
    z2_pooled = F.normalize(z2_pooled, dim=-1)
    
    # Similarity matrix
    sim = torch.matmul(z1_pooled, z2_pooled.T) / temperature  # (batch, batch)
    
    # InfoNCE loss
    labels = torch.arange(batch_size, device=z1.device)
    loss_instance = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    
    # Temporal-level contrast
    # Normalize along feature dimension
    z1_norm = F.normalize(z1, dim=-1)  # (batch, time, dim)
    z2_norm = F.normalize(z2, dim=-1)
    
    # Compute similarity between corresponding timestamps
    # For simplicity, use aligned timestamps
    sim_temporal = (z1_norm * z2_norm).sum(dim=-1)  # (batch, time)
    
    # Positive pairs: same timestamp
    # Negative pairs: different timestamps in same sequence
    loss_temporal = -sim_temporal.mean() + 1.0  # Simple alignment loss
    
    return loss_instance + 0.5 * loss_temporal


def augment_data(x: torch.Tensor, 
                 mask_ratio: float = 0.3,
                 jitter_std: float = 0.1) -> torch.Tensor:
    """
    Apply data augmentation for contrastive learning.
    
    Augmentations:
    - Random timestamp masking
    - Gaussian jitter
    
    Args:
        x: Input tensor (batch, channels, time)
        mask_ratio: Ratio of timestamps to mask
        jitter_std: Standard deviation of Gaussian noise
    
    Returns:
        Augmented tensor
    """
    batch, channels, time = x.shape
    device = x.device
    
    # Clone to avoid modifying original
    x_aug = x.clone()
    
    # Random masking
    mask = torch.rand(batch, 1, time, device=device) > mask_ratio
    x_aug = x_aug * mask.float()
    
    # Gaussian jitter
    noise = torch.randn_like(x_aug) * jitter_std
    x_aug = x_aug + noise
    
    return x_aug


def pretrain_ts2vec(windows: List[Dict],
                    config: Config,
                    device: torch.device,
                    logger,
                    n_epochs: int = 50,
                    batch_size: int = 32,
                    learning_rate: float = 1e-3,
                    output_dim: int = 64) -> TS2VecEncoder:
    """
    Pretrain TS2Vec encoder using contrastive learning.
    
    Args:
        windows: List of window dictionaries with aligned data
        config: Configuration object
        device: Torch device
        logger: Logger instance
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dim: Output embedding dimension
    
    Returns:
        Pretrained encoder
    """
    logger.info(f"Pretraining TS2Vec encoder...")
    logger.info(f"  Windows: {len(windows)}")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Output dim: {output_dim}")
    
    # Create dataset
    dataset = VitaStressTS2VecDataset(windows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=True)
    
    logger.info(f"  Batches per epoch: {len(dataloader)}")
    
    # Create encoder
    encoder = TS2VecEncoder(
        input_channels=config.n_channels,
        output_dim=output_dim,
        hidden_dim=64
    ).to(device)
    
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"  Encoder parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    start_time = time.time()
    
    # Training loop
    epoch_pbar = tqdm(range(n_epochs), desc="Pretraining", unit="epoch")
    
    for epoch in epoch_pbar:
        encoder.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, _ in dataloader:  # Ignore labels during pretraining
            x = x.to(device)
            
            # Create two augmented views
            x1 = augment_data(x, mask_ratio=0.3, jitter_std=0.1)
            x2 = augment_data(x, mask_ratio=0.3, jitter_std=0.1)
            
            # Encode both views
            z1 = encoder(x1)
            z2 = encoder(x2)
            
            # Compute contrastive loss
            loss = hierarchical_contrastive_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / max(n_batches, 1)
        epoch_pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})
        
        # Log periodically
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}")
    
    epoch_pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Pretraining completed in {elapsed/60:.1f} minutes")
    
    return encoder


def load_all_windows(config: Config, logger) -> List[Dict]:
    """Load all windows from all subjects."""
    subjects = get_all_subjects(config.data_path)
    all_windows = []
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Loading data", unit="subject")
    successful = 0
    
    for subject_folder in pbar:
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            continue
        
        aligned = align_to_1hz(signals, start, end)
        if aligned is None or len(aligned) == 0:
            continue
        
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=config.window_size_sec,
            overlap_ratio=config.overlap_ratio,
            horizons_minutes=config.horizons_minutes,
            skip_first_minutes=config.skip_first_minutes
        )
        
        if windows:
            for w in windows:
                w["subject_id"] = subject_id
            all_windows.extend(windows)
            successful += 1
            pbar.set_postfix({"OK": successful, "Windows": len(all_windows)})
    
    pbar.close()
    logger.info(f"Loaded {len(all_windows)} windows from {successful} subjects")
    
    return all_windows


def main():
    """Main pretraining pipeline."""
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("ts2vec_pretrain", log_file=results_dir / "pretrain.log")
    
    log_experiment_start(logger, "TS2VEC SELF-SUPERVISED PRETRAINING")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load all data
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Loading data")
    logger.info("-"*50)
    
    all_windows = load_all_windows(config, logger)
    
    if not all_windows:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Total windows for pretraining: {len(all_windows)}")
    
    # Pretrain encoder
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Contrastive pretraining")
    logger.info("-"*50)
    
    encoder = pretrain_ts2vec(
        all_windows,
        config,
        device,
        logger,
        n_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        output_dim=64
    )
    
    # Save encoder
    encoder_path = results_dir / "ts2vec_encoder.pt"
    torch.save(encoder.state_dict(), encoder_path)
    logger.info(f"Saved encoder to: {encoder_path}")
    
    # Save config for reproducibility
    config_path = results_dir / "pretrain_config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump({
            "n_channels": config.n_channels,
            "window_size_sec": config.window_size_sec,
            "output_dim": 64,
            "hidden_dim": 64
        }, f)
    logger.info(f"Saved config to: {config_path}")
    
    log_experiment_end(logger, "TS2VEC SELF-SUPERVISED PRETRAINING")
    logger.info(f"All outputs saved to: {results_dir}")


if __name__ == "__main__":
    main()
