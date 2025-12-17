"""
Configuration for Subject-Aware SSL experiment.

Defines experiment parameters, sampling rates, and training settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from enum import Enum


class SSLMode(Enum):
    """SSL training mode."""
    BASE = "base"                    # Plain contrastive, no subject awareness
    SUBJECT_INVARIANT = "invariant"  # Contrastive + Adversarial (combined)
    SUBJECT_SPECIFIC = "specific"    # Contrastive with same-subject negatives only


@dataclass
class SSLConfig:
    """Configuration for Subject-Aware SSL experiment."""
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    # data_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data")
    # results_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/experiments/05_subject_aware_ssl/results")
    # data_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data")
    # data_path: Path = Path("/kaggle/input/vitastress/VitaStress/data")
    data_path: Path = Path("/kaggle/input/vitastess2/VitaStress/data")
    # results_base_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/experiments")
    results_path: Path = Path("/kaggle/working/stress-prediction/experiments/05_subject_aware_ssl/results") 
    # ==========================================================================
    # Data Parameters
    # ==========================================================================
    # Sampling rate for SSL (8Hz provides balance between resolution and efficiency)
    ssl_sample_rate: float = 8.0
    
    # Window sizes to test
    window_sizes_sec: List[int] = field(default_factory=lambda: [60, 120])
    
    # Default window size
    window_size_sec: int = 120
    
    # Overlap ratio for windowing
    overlap_ratio: float = 0.5
    
    # Skip first N minutes (sensor settling)
    skip_first_minutes: int = 1
    
    # Prediction horizons (in minutes)
    horizons_minutes: List[int] = field(default_factory=lambda: [3, 5])
    
    # Default target label
    target_label: str = "label_3min"
    
    # Number of input channels
    # At 8Hz: acc_magnitude (from 32Hz), skin_temp (from 1Hz), ppg_mean (from 64Hz)
    n_channels: int = 3
    
    # Feature names for each channel
    feature_names: List[str] = field(default_factory=lambda: [
        "acc_magnitude",
        "skin_temp",
        "ppg_mean"
    ])
    
    # ==========================================================================
    # SSL Training Mode
    # ==========================================================================
    ssl_mode: SSLMode = SSLMode.SUBJECT_INVARIANT
    
    # Lambda for adversarial loss (only used in SUBJECT_INVARIANT mode)
    # Paper recommends 0.5-1.0 for ~20 subjects
    ssl_adversarial_lambda: float = 0.5
    
    # Temperature for InfoNCE loss
    temperature: float = 0.1
    
    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    # Encoder embedding dimension
    embedding_dim: int = 256
    
    # Projection head output dimension (for contrastive loss)
    projection_dim: int = 64
    
    # Encoder hidden dimensions
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    
    # ==========================================================================
    # Pre-training Parameters
    # ==========================================================================
    pretrain_epochs: int = 800
    pretrain_batch_size: int = 128
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-4
    pretrain_patience: int = 30  # Early stopping: stop if no improvement for N epochs
    
    # ==========================================================================
    # Fine-tuning Parameters
    # ==========================================================================
    finetune_epochs: int = 50
    finetune_batch_size: int = 32
    finetune_lr: float = 1e-4  # Lower LR for fine-tuning
    finetune_encoder_lr_factor: float = 0.1  # Encoder gets lr * factor
    finetune_weight_decay: float = 1e-4
    
    # Early stopping patience
    patience: int = 15
    
    # ==========================================================================
    # Augmentation Settings
    # ==========================================================================
    # Enable signal mixing during pre-training
    enable_signal_mixing: bool = False
    
    # ==========================================================================
    # Stress Event Definitions (from shared config)
    # ==========================================================================
    emotional_stress_start_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Start",
        "Public Speaking Start"
    ])
    
    emotional_stress_stop_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Stop",
        "Public Speaking Stop"
    ])
    
    stress_start_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Start",
        "Public Speaking Start"
    ])
    
    stress_stop_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Stop",
        "Public Speaking Stop"
    ])
    
    baseline_events: List[str] = field(default_factory=lambda: [
        "Baseline Start",
        "Rest: Start"
    ])
    
    # ==========================================================================
    # Reproducibility
    # ==========================================================================
    random_seed: int = 42
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_path = Path(self.data_path)
        self.results_path = Path(self.results_path)
    
    @property
    def samples_per_window(self) -> int:
        """Number of samples per window at SSL sample rate."""
        return int(self.window_size_sec * self.ssl_sample_rate)
    
    def get_config_for_run(
        self,
        window_size_sec: int,
        horizon_minutes: int,
        ssl_mode: SSLMode
    ) -> "SSLConfig":
        """
        Create a config copy for a specific run configuration.
        
        Args:
            window_size_sec: Window size in seconds
            horizon_minutes: Prediction horizon in minutes
            ssl_mode: SSL training mode
        
        Returns:
            Config with updated parameters
        """
        config = SSLConfig(
            data_path=self.data_path,
            results_path=self.results_path,
            ssl_sample_rate=self.ssl_sample_rate,
            window_sizes_sec=self.window_sizes_sec,
            window_size_sec=window_size_sec,
            overlap_ratio=self.overlap_ratio,
            skip_first_minutes=self.skip_first_minutes,
            horizons_minutes=self.horizons_minutes,
            target_label=f"label_{horizon_minutes}min",
            n_channels=self.n_channels,
            feature_names=self.feature_names,
            ssl_mode=ssl_mode,
            ssl_adversarial_lambda=self.ssl_adversarial_lambda,
            temperature=self.temperature,
            embedding_dim=self.embedding_dim,
            projection_dim=self.projection_dim,
            encoder_hidden_dims=self.encoder_hidden_dims,
            pretrain_epochs=self.pretrain_epochs,
            pretrain_batch_size=self.pretrain_batch_size,
            pretrain_lr=self.pretrain_lr,
            pretrain_weight_decay=self.pretrain_weight_decay,
            finetune_epochs=self.finetune_epochs,
            finetune_batch_size=self.finetune_batch_size,
            finetune_lr=self.finetune_lr,
            finetune_encoder_lr_factor=self.finetune_encoder_lr_factor,
            finetune_weight_decay=self.finetune_weight_decay,
            patience=self.patience,
            enable_signal_mixing=self.enable_signal_mixing,
            emotional_stress_start_events=self.emotional_stress_start_events,
            emotional_stress_stop_events=self.emotional_stress_stop_events,
            stress_start_events=self.stress_start_events,
            stress_stop_events=self.stress_stop_events,
            baseline_events=self.baseline_events,
            random_seed=self.random_seed
        )
        return config


# Default configuration
DEFAULT_SSL_CONFIG = SSLConfig()


def get_run_name(config: SSLConfig) -> str:
    """Generate run name from config."""
    return f"ssl_{config.ssl_mode.value}_{config.window_size_sec}s"


if __name__ == "__main__":
    # Test config
    config = SSLConfig()
    
    print("SSL Config:")
    print(f"  Data path: {config.data_path}")
    print(f"  SSL sample rate: {config.ssl_sample_rate} Hz")
    print(f"  Window size: {config.window_size_sec}s")
    print(f"  Samples per window: {config.samples_per_window}")
    print(f"  SSL mode: {config.ssl_mode.value}")
    print(f"  Adversarial lambda: {config.ssl_adversarial_lambda}")
    print(f"  Channels: {config.n_channels}")
    print(f"  Features: {config.feature_names}")
    
    # Test run config
    run_config = config.get_config_for_run(60, 5, SSLMode.SUBJECT_SPECIFIC)
    print(f"\nRun config:")
    print(f"  Window: {run_config.window_size_sec}s")
    print(f"  Target: {run_config.target_label}")
    print(f"  Mode: {run_config.ssl_mode.value}")
    print(f"  Run name: {get_run_name(run_config)}")
