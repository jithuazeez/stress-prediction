"""
Shared configuration for all experiments.

Defines paths, window sizes, prediction horizons, and other common parameters.

IMPORTANT: This experiment focuses on EMOTIONAL/MENTAL stress detection.
Physical stress (exercise) is labeled as NO STRESS to help the model learn
to distinguish between emotional stress and physical exertion.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration for VitaStress experiments."""
    
    # Paths
    data_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data")
    # data_path: Path = Path("/kaggle/input/vitastress/VitaStress/data")
    results_base_path: Path = Path("/Users/jithuazeez/Documents/Msc/Dissertation/experiments")
    # results_base_path: Path = Path("/kaggle/working/stress-prediction/experiments")
    # Window parameters
    window_size_sec: int = 120  # 120 second windows
    overlap_ratio: float = 0.5  # 50% overlap (doubles sample count)
    skip_first_minutes: int = 1  # Skip first minute (sensor settling)
    
    # NOTE: Overlapping windows are safe with LOSO because:
    # - All windows from the same subject stay in the same split (train OR test)
    # - No data leakage between subjects
    # - Overlap only increases samples WITHIN each subject's data
    
    # Prediction horizons (in minutes)
    horizons_minutes: List[int] = field(default_factory=lambda: [3, 5, 10])
    
    # Target label for experiments
    target_label: str = "label_3min"
    
    # Sampling rates (from VitaStress dataset)
    # NOTE: These are used as defaults but should be calculated from data
    acc_sample_rate: float = 32.0  # Accelerometer ~32 Hz
    ppg_sample_rate: float = 64.0  # PPG ~64 Hz
    heatflux_sample_rate: float = 1.0  # Heat flux 1 Hz
    eda_sample_rate: float = 0.017  # Emography ~1 sample/minute
    
    # Target sample rate for alignment
    target_sample_rate: float = 1.0  # Align everything to 1 Hz
    
    # ==========================================================================
    # STRESS EVENT DEFINITIONS
    # ==========================================================================
    
    # EMOTIONAL/MENTAL stress events - these ARE labeled as STRESS (1)
    # These involve cognitive load, social stress, anxiety, etc.
    emotional_stress_start_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Start",       # Mental arithmetic, cognitive tasks
        "Public Speaking Start"   # Social stress, anxiety
    ])
    
    emotional_stress_stop_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Stop",
        "Public Speaking Stop"
    ])
    
    # PHYSICAL stress events - these are NOT labeled as stress (0)
    # These are exercise/physical activity, not emotional stress
    physical_stress_start_events: List[str] = field(default_factory=lambda: [
        "Physical: Start"
    ])
    
    physical_stress_stop_events: List[str] = field(default_factory=lambda: [
        "Physical: Stop"
    ])
    
    # ALL experiment events (for context detection, NOT for labeling)
    # NOTE: Physical stress is NOT labeled as stress (1) - only emotional stress is!
    # This list is used to detect when user is in ANY experiment (for context)
    all_experiment_start_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Start",
        "Physical: Start",
        "Public Speaking Start"
    ])
    
    all_experiment_stop_events: List[str] = field(default_factory=lambda: [
        "Cognitive: Stop",
        "Physical: Stop",
        "Public Speaking Stop"
    ])
    
    # For backwards compatibility (deprecated - use emotional_stress_start_events for labeling)
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
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Number of channels for deep learning models
    # HR from HeartPy replaces raw PPG (meaningful at 1Hz)
    # EDA disabled due to low sampling rate (~0.017 Hz)
    n_channels: int = 3  # acc_magnitude, skin_temp, hr_bpm
    # n_channels: int = 3  # acc_magnitude, skin_temp, eda_stress_skin (EDA disabled)
    # n_channels: int = 4  # acc_magnitude, skin_temp, eda_stress_skin, ppg_mean (PPG disabled)
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_path = Path(self.data_path)
        self.results_base_path = Path(self.results_base_path)


# Default configuration instance
DEFAULT_CONFIG = Config()
