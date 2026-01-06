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
    # data_path: Path = Path("/kaggle/input/vitastress/VitaStress/data")
    # data_path: Path = Path("/kaggle/input/vitastess2/VitaStress/data")
    results_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/experiments")
    # results_path: Path = Path("/kaggle/working/stress-prediction/experiments")
    # ==========================================================================
    # Native Sampling Rates (Same channels as MOMENT but at native rates)
    # ==========================================================================
    # ppg_sample_rate: float = 64.0   # PPG at 64 Hz (DISABLED - using HR/HRV instead)
    acc_sample_rate: float = 32.0   # Accelerometer at 32 Hz (3 channels: x, y, z)
    physio_sample_rate: float = 1.0   # Physiological signals at 1 Hz (5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd)
    
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
    # ACC Encoder (3 channels at 32Hz: acc_x, acc_y, acc_z)
    acc_embedding_dim: int = 128
    
    # Physio Encoder (5 channels at 1Hz: skin_temp, heatflux, cbt, hr_bpm, rmssd)
    physio_embedding_dim: int = 128
    
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
    def acc_samples_per_window(self) -> int:
        """Number of ACC samples per window (3 channels: x, y, z)."""
        return int(self.window_size_sec * self.acc_sample_rate)
    
    @property
    def physio_samples_per_window(self) -> int:
        """Number of physiological samples per window (5 channels at 1Hz)."""
        return int(self.window_size_sec * self.physio_sample_rate)
    
    @property
    def total_embedding_dim(self) -> int:
        """Total dimension after concatenating all embeddings."""
        return self.acc_embedding_dim + self.physio_embedding_dim
    
    def get_config_for_run(
        self,
        window_size_sec: int,
        horizon_minutes: int
    ) -> "MultiRateConfig":
        """Create config copy for specific run."""
        config = MultiRateConfig(
            data_path=self.data_path,
            results_path=self.results_path,
            acc_sample_rate=self.acc_sample_rate,
            physio_sample_rate=self.physio_sample_rate,
            window_sizes_sec=self.window_sizes_sec,
            window_size_sec=window_size_sec,
            overlap_ratio=self.overlap_ratio,
            skip_first_minutes=self.skip_first_minutes,
            horizons_minutes=self.horizons_minutes,
            target_label=f"label_{horizon_minutes}min",
            acc_embedding_dim=self.acc_embedding_dim,
            physio_embedding_dim=self.physio_embedding_dim,
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
    print(f"  ACC: {config.acc_sample_rate}Hz (3 channels) -> {config.acc_samples_per_window} samples")
    print(f"  Physio: {config.physio_sample_rate}Hz (5 channels) -> {config.physio_samples_per_window} samples")
    print(f"  Total embedding: {config.total_embedding_dim}")

