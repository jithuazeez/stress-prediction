"""
Self-supervised pre-training script for Subject-Aware SSL.

Supports three training modes:
1. Base SSL: Standard InfoNCE contrastive loss
2. Subject-Invariant: InfoNCE + Adversarial loss (for generalization)
3. Subject-Specific: InfoNCE with same-subject negatives (for fine-tuning init)

Usage:
    python pretrain.py --mode invariant --window_size 120 --epochs 100
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import json
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import get_all_subjects
from shared.logging_utils import setup_logger, log_experiment_start, log_experiment_end

from config import SSLConfig, SSLMode, DEFAULT_SSL_CONFIG, get_run_name
from dataset import load_windows_at_8hz, create_ssl_dataloaders, SSLDataset
from encoder import (
    SSLEncoder, ProjectionHead, SubjectClassifier, SSLModel,
    create_ssl_encoder, count_parameters
)
from losses import (
    BaseLoss, SubjectInvariantLoss, SubjectSpecificLoss,
    create_loss_fn
)
from augmentations import Augmenter, AugmentationConfig, create_augmenter


def pretrain_epoch(
    model: SSLModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    augmenter: Augmenter,
    config: SSLConfig,
    device: torch.device,
    subject_classifier: Optional[nn.Module] = None,
    clf_optimizer: Optional[optim.Optimizer] = None
) -> Dict[str, float]:
    """
    Run one pre-training epoch.
    
    Args:
        model: SSL model (encoder + projection head)
        dataloader: Training data loader
        loss_fn: Loss function module
        optimizer: Optimizer for encoder and projection head
        augmenter: Data augmentation pipeline
        config: SSL configuration
        device: Torch device
        subject_classifier: Subject classifier (for invariant mode)
        clf_optimizer: Optimizer for subject classifier
    
    Returns:
        Dictionary of epoch metrics
    """
    model.train()
    if subject_classifier is not None:
        subject_classifier.train()
    
    total_loss = 0.0
    total_contrastive = 0.0
    total_classifier = 0.0
    total_adversarial = 0.0
    n_batches = 0
    
    for x, labels, subject_ids in dataloader:
        x = x.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)
        
        # Create two augmented views
        view1, view2 = augmenter(x, labels)
        
        # Forward pass through encoder
        embedding1 = model.encoder(view1)
        embedding2 = model.encoder(view2)
        
        # Project embeddings
        z1 = model.head(embedding1)
        z2 = model.head(embedding2)
        
        # Compute loss based on mode
        if config.ssl_mode == SSLMode.SUBJECT_INVARIANT:
            # Use embeddings for adversarial training
            embeddings = embedding1  # Use view1 embeddings
            losses = loss_fn(z1, z2, embeddings, subject_ids)
            
            # IMPORTANT: Do encoder backward FIRST (before classifier update)
            optimizer.zero_grad()
            losses["total"].backward()
            
            # Now do a SEPARATE forward pass for classifier update
            # This avoids the in-place modification error
            if subject_classifier is not None and clf_optimizer is not None:
                clf_optimizer.zero_grad()
                # Fresh forward pass with detached embeddings
                with torch.no_grad():
                    emb_detached = model.encoder(view1)
                clf_logits = subject_classifier(emb_detached)
                clf_loss = nn.functional.cross_entropy(clf_logits, subject_ids)
                clf_loss.backward()
                clf_optimizer.step()
                total_classifier += clf_loss.item()
            
            total_adversarial += losses["adversarial"].item()
        else:
            # Base or Subject-Specific mode
            losses = loss_fn(z1, z2, subject_ids=subject_ids)
            
            # Backward pass for encoder
            optimizer.zero_grad()
            losses["total"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses["total"].item()
        total_contrastive += losses["contrastive"].item()
        n_batches += 1
    
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "contrastive_loss": total_contrastive / max(n_batches, 1)
    }
    
    if config.ssl_mode == SSLMode.SUBJECT_INVARIANT:
        metrics["classifier_loss"] = total_classifier / max(n_batches, 1)
        metrics["adversarial_loss"] = total_adversarial / max(n_batches, 1)
    
    return metrics


def pretrain(
    config: SSLConfig,
    windows: list,
    subject_to_idx: Dict[str, int],
    device: torch.device,
    logger
) -> SSLEncoder:
    """
    Pre-train encoder using contrastive learning.
    
    Args:
        config: SSL configuration
        windows: List of window dictionaries
        subject_to_idx: Subject ID to index mapping
        device: Torch device
        logger: Logger instance
    
    Returns:
        Pre-trained encoder
    """
    n_subjects = len(subject_to_idx)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PRE-TRAINING: {config.ssl_mode.value.upper()} MODE")
    logger.info(f"{'='*60}")
    logger.info(f"Windows: {len(windows)}")
    logger.info(f"Subjects: {n_subjects}")
    logger.info(f"Window size: {config.window_size_sec}s")
    logger.info(f"Samples per window: {config.samples_per_window}")
    logger.info(f"Epochs: {config.pretrain_epochs}")
    logger.info(f"Batch size: {config.pretrain_batch_size}")
    logger.info(f"Learning rate: {config.pretrain_lr}")
    
    # Create dataset and dataloader
    dataset = SSLDataset(windows, config, subject_to_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    
    # Create encoder and projection head
    encoder = create_ssl_encoder(
        input_channels=config.n_channels,
        embedding_dim=config.embedding_dim
    ).to(device)
    
    projection_head = ProjectionHead(
        input_dim=config.embedding_dim,
        hidden_dim=128,
        output_dim=config.projection_dim
    ).to(device)
    
    model = SSLModel(encoder, projection_head)
    
    logger.info(f"\nModel parameters:")
    logger.info(f"  Encoder: {count_parameters(encoder):,}")
    logger.info(f"  Projection: {count_parameters(projection_head):,}")
    logger.info(f"  Total: {count_parameters(model):,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.pretrain_epochs,
        eta_min=config.pretrain_lr * 0.01
    )
    
    # Create subject classifier and loss function based on mode
    subject_classifier = None
    clf_optimizer = None
    
    if config.ssl_mode == SSLMode.SUBJECT_INVARIANT:
        subject_classifier = SubjectClassifier(
            input_dim=config.embedding_dim,
            n_subjects=n_subjects
        ).to(device)
        
        clf_optimizer = optim.Adam(
            subject_classifier.parameters(),
            lr=config.pretrain_lr * 5  # Classifier can learn faster
        )
        
        loss_fn = SubjectInvariantLoss(
            subject_classifier,
            lambda_adv=config.ssl_adversarial_lambda,
            temperature=config.temperature
        )
        
        logger.info(f"\nSubject-Invariant settings:")
        logger.info(f"  Lambda: {config.ssl_adversarial_lambda}")
        logger.info(f"  Classifier params: {count_parameters(subject_classifier):,}")
    
    elif config.ssl_mode == SSLMode.SUBJECT_SPECIFIC:
        loss_fn = SubjectSpecificLoss(temperature=config.temperature)
        logger.info(f"\nSubject-Specific: Using same-subject negatives only")
    
    else:  # BASE
        loss_fn = BaseLoss(temperature=config.temperature)
        logger.info(f"\nBase SSL: Using all samples as negatives")
    
    # Create augmenter
    aug_config = AugmentationConfig()
    aug_config.enable_signal_mixing = config.enable_signal_mixing
    augmenter = Augmenter(aug_config)
    
    logger.info(f"\nAugmentation settings:")
    logger.info(f"  Temporal delay: max {aug_config.temporal_delay_max_samples} samples")
    logger.info(f"  Gaussian noise std: {aug_config.gaussian_noise_std}")
    logger.info(f"  Temporal cutout: max {aug_config.temporal_cutout_max_ratio*100:.0f}%")
    logger.info(f"  Channel dropout: {aug_config.channel_dropout_p_drop*100:.0f}%")
    logger.info(f"  Signal mixing: {'enabled' if aug_config.enable_signal_mixing else 'disabled'}")
    
    # Training loop with early stopping
    logger.info(f"\n{'-'*50}")
    logger.info("Starting pre-training...")
    logger.info(f"{'-'*50}")
    
    pretrain_patience = getattr(config, "pretrain_patience", 30)
    logger.info(f"Early stopping: DISABLED (will train for all {config.pretrain_epochs} epochs)")
    
    start_time = time.time()
    best_loss = float("inf")
    patience_counter = 0
    best_encoder_state = None
    
    pbar = tqdm(range(config.pretrain_epochs), desc="Pre-training", unit="epoch")
    
    for epoch in pbar:
        metrics = pretrain_epoch(
            model, dataloader, loss_fn, optimizer, augmenter,
            config, device, subject_classifier, clf_optimizer
        )
        
        scheduler.step()
        
        # Update progress bar
        pbar_dict = {"loss": f"{metrics['loss']:.4f}", "best": f"{best_loss:.4f}"}
        if config.ssl_mode == SSLMode.SUBJECT_INVARIANT:
            pbar_dict["clf"] = f"{metrics['classifier_loss']:.4f}"
        pbar.set_postfix(pbar_dict)
        
        # Log periodically
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{config.pretrain_epochs}: "
                f"Loss={metrics['loss']:.4f}, "
                f"Best={best_loss:.4f}, "
                f"Contrastive={metrics['contrastive_loss']:.4f}, "
                f"LR={scheduler.get_last_lr()[0]:.6f}"
            )
        
        # Check for improvement and save best state
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            patience_counter = 0
            # Save best encoder state
            best_encoder_state = encoder.state_dict().copy()
        else:
            patience_counter += 1
        
        # Early stopping disabled - let model train for full epochs
        # if patience_counter >= pretrain_patience:
        #     logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
        #     logger.info(f"No improvement for {pretrain_patience} epochs")
        #     break
    
    pbar.close()
    
    # Restore best encoder state
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        logger.info("Restored best encoder state")
    
    elapsed = time.time() - start_time
    final_epoch = epoch + 1
    logger.info(f"\nPre-training completed in {elapsed/60:.1f} minutes ({final_epoch} epochs)")
    logger.info(f"Best loss: {best_loss:.4f}")
    
    return encoder


def save_pretrained(
    encoder: SSLEncoder,
    config: SSLConfig,
    logger
) -> Path:
    """
    Save pre-trained encoder and config.
    
    Args:
        encoder: Pre-trained encoder
        config: SSL configuration
        logger: Logger instance
    
    Returns:
        Path to saved encoder
    """
    results_dir = config.results_path / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save encoder
    run_name = get_run_name(config)
    encoder_path = results_dir / f"pretrain_encoder_{run_name}.pt"
    torch.save(encoder.state_dict(), encoder_path)
    logger.info(f"Saved encoder to: {encoder_path}")
    
    # Save config
    config_dict = {
        "ssl_mode": config.ssl_mode.value,
        "window_size_sec": config.window_size_sec,
        "ssl_sample_rate": config.ssl_sample_rate,
        "n_channels": config.n_channels,
        "embedding_dim": config.embedding_dim,
        "projection_dim": config.projection_dim,
        "pretrain_epochs": config.pretrain_epochs,
        "pretrain_batch_size": config.pretrain_batch_size,
        "pretrain_lr": config.pretrain_lr,
        "ssl_adversarial_lambda": config.ssl_adversarial_lambda,
        "temperature": config.temperature
    }
    
    config_path = results_dir / f"pretrain_config_{run_name}.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config to: {config_path}")
    
    return encoder_path


def main():
    """Main pre-training pipeline."""
    parser = argparse.ArgumentParser(description="SSL Pre-training")
    parser.add_argument(
        "--mode", type=str, default="invariant",
        choices=["base", "invariant", "specific"],
        help="SSL training mode"
    )
    parser.add_argument(
        "--window_size", type=int, default=120,
        help="Window size in seconds (60 or 120)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of pre-training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=0.5,
        help="Lambda for adversarial loss (invariant mode)"
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience (stop if no improvement for N epochs)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = DEFAULT_SSL_CONFIG
    config.ssl_mode = SSLMode(args.mode)
    config.window_size_sec = args.window_size
    config.pretrain_epochs = args.epochs
    config.pretrain_batch_size = args.batch_size
    config.pretrain_lr = args.lr
    config.ssl_adversarial_lambda = args.lambda_adv
    config.pretrain_patience = args.patience
    
    # Setup directories
    config.results_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = config.results_path / f"pretrain_{get_run_name(config)}.log"
    logger = setup_logger("ssl_pretrain", log_file=log_file)
    
    log_experiment_start(logger, "SUBJECT-AWARE SSL PRE-TRAINING")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Load data
    logger.info(f"\n{'-'*50}")
    logger.info("Loading data at 8Hz...")
    logger.info(f"{'-'*50}")
    
    windows, subject_to_idx = load_windows_at_8hz(config)
    
    if not windows:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(windows)} windows from {len(subject_to_idx)} subjects")
    
    # Pre-train
    encoder = pretrain(config, windows, subject_to_idx, device, logger)
    
    # Save
    encoder_path = save_pretrained(encoder, config, logger)
    
    log_experiment_end(logger, "SUBJECT-AWARE SSL PRE-TRAINING")
    logger.info(f"\nPre-trained encoder saved to: {encoder_path}")


if __name__ == "__main__":
    main()
