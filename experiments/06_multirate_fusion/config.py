"""
Configuration for Multi-Rate Late Fusion experiment.

Processes signals at their native sampling rates:
- PPG: 64 Hz (7680 samples for 120s window)
- ACC: 32 Hz (3840 samples for 120s window)
- Temp: 1 Hz (120 samples for 120s window)
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class MultiRateConfig:
    """Configuration for Multi-Rate Late Fusion experiment."""
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    data_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data")
    results_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/experiments/06_multirate_fusion/results")
    
    # ==========================================================================
    # Native Sampling Rates
    # ==========================================================================
    ppg_sample_rate: float = 64.0   # PPG at 64 Hz
    acc_sample_rate: float = 32.0   # Accelerometer at 32 Hz
    temp_sample_rate: float = 1.0   # Temperature at 1 Hz
    
    # ==========================================================================
    # Window Parameters
    # ==========================================================================
    window_sizes_sec: List[int] = field(default_factory=lambda: [60, 120])
    window_size_sec: int = 120
    overlap_ratio: float = 0.5
    skip_first_minutes: int = 1
    
    # ==========================================================================
    # Prediction Horizons
    # ==========================================================================
    horizons_minutes: List[int] = field(default_factory=lambda: [3, 5])
    target_label: str = "label_3min"
    
    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    # PPG Encoder output
    ppg_embedding_dim: int = 128
    
    # ACC Encoder output
    acc_embedding_dim: int = 128
    
    # Temp Encoder output
    temp_embedding_dim: int = 64
    
    # Fusion layer dimension
    fusion_dim: int = 128
    
    # Dropout
    dropout: float = 0.3
    
    # ==========================================================================
    # Training Parameters
    # ==========================================================================
    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    
    # ==========================================================================
    # Stress Event Definitions
    # ==========================================================================
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
    def ppg_samples_per_window(self) -> int:
        """Number of PPG samples per window."""
        return int(self.window_size_sec * self.ppg_sample_rate)
    
    @property
    def acc_samples_per_window(self) -> int:
        """Number of ACC samples per window."""
        return int(self.window_size_sec * self.acc_sample_rate)
    
    @property
    def temp_samples_per_window(self) -> int:
        """Number of temperature samples per window."""
        return int(self.window_size_sec * self.temp_sample_rate)
    
    @property
    def total_embedding_dim(self) -> int:
        """Total dimension after concatenating all embeddings."""
        return self.ppg_embedding_dim + self.acc_embedding_dim + self.temp_embedding_dim
    
    def get_config_for_run(
        self,
        window_size_sec: int,
        horizon_minutes: int
    ) -> "MultiRateConfig":
        """Create config copy for specific run."""
        config = MultiRateConfig(
            data_path=self.data_path,
            results_path=self.results_path,
            ppg_sample_rate=self.ppg_sample_rate,
            acc_sample_rate=self.acc_sample_rate,
            temp_sample_rate=self.temp_sample_rate,
            window_sizes_sec=self.window_sizes_sec,
            window_size_sec=window_size_sec,
            overlap_ratio=self.overlap_ratio,
            skip_first_minutes=self.skip_first_minutes,
            horizons_minutes=self.horizons_minutes,
            target_label=f"label_{horizon_minutes}min",
            ppg_embedding_dim=self.ppg_embedding_dim,
            acc_embedding_dim=self.acc_embedding_dim,
            temp_embedding_dim=self.temp_embedding_dim,
            fusion_dim=self.fusion_dim,
            dropout=self.dropout,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            stress_start_events=self.stress_start_events,
            stress_stop_events=self.stress_stop_events,
            baseline_events=self.baseline_events,
            random_seed=self.random_seed
        )
        return config


# Default configuration
DEFAULT_MULTIRATE_CONFIG = MultiRateConfig()


def get_run_name(config: MultiRateConfig) -> str:
    """Generate run name from config."""
    return f"multirate_{config.window_size_sec}s_{config.target_label}"


if __name__ == "__main__":
    config = MultiRateConfig()
    
    print("Multi-Rate Config:")
    print(f"  Window size: {config.window_size_sec}s")
    print(f"  PPG: {config.ppg_sample_rate}Hz -> {config.ppg_samples_per_window} samples")
    print(f"  ACC: {config.acc_sample_rate}Hz -> {config.acc_samples_per_window} samples")
    print(f"  Temp: {config.temp_sample_rate}Hz -> {config.temp_samples_per_window} samples")
    print(f"  Total embedding: {config.total_embedding_dim}")
