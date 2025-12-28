"""
PyTorch Dataset for TCN model with raw multivariate time series.

UPDATED: Now matches MOMENT's approach - uses raw sensor channels instead of extracted features.
This allows TCN to learn temporal patterns directly from the raw sequences.

Uses same 8 channels as MOMENT:
- acc_x, acc_y, acc_z: Accelerometer axes (capturing movement patterns)
- skin_temp: Skin temperature
- heatflux: Heat flux (thermal energy transfer)
- cbt: Core body temperature
- hr_bpm: Heart rate from PPG (at 1Hz)
- rmssd: HRV metric (rolling RMSSD at 1Hz)

Architecture:
- Input: (batch, 8 channels, 120 timesteps)
- Each sensor is a channel (not extracted features)
- Subject-wise normalization for physiological consistency
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config

logger = logging.getLogger(__name__)


class VitaStressTCNDataset(Dataset):
    """
    Dataset for TCN model with raw multivariate time series.
    
    Uses same channels as MOMENT (8 channels × 120 timesteps):
    - acc_x, acc_y, acc_z: Accelerometer (3D movement)
    - skin_temp: Skin temperature
    - heatflux: Heat flux
    - cbt: Core body temperature  
    - hr_bpm: Heart rate (from PPG)
    - rmssd: HRV metric (rolling RMSSD)
    
    Output shape: (n_channels=8, seq_len=120)
    """
    
    # Channel names matching MOMENT
    CHANNELS = [
        "acc_x",
        "acc_y",
        "acc_z",
        "skin_temp",
        "heatflux",
        "cbt",
        "hr_bpm",
        "rmssd"
    ]
    
    def __init__(self, 
                 windows: List[Dict],
                 label_col: str = "label_5min",
                 seq_len: int = 120,
                 normalize: bool = True,
                 normalization_mode: str = "subject"):
        """
        Initialize dataset.
        
        Args:
            windows: List of window dictionaries from create_labeled_windows()
            label_col: Label column to use
            seq_len: Expected sequence length (default 120 for 120s windows at 1Hz)
            normalize: Whether to normalize each channel
            normalization_mode: How to normalize. Options:
                - "subject": Use subject-level mean/std (recommended)
                - "window": Use per-window mean/std
        """
        self.windows = windows
        self.label_col = label_col
        self.seq_len = seq_len
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.n_channels = len(self.CHANNELS)
        
        # Pre-process all windows
        self.samples = []
        self.labels = []
        self.subject_ids = []
        
        logger.info(f"Processing {len(windows)} windows into raw sequences...")
        
        for i, window in enumerate(windows):
            x = self._prepare_window(window)
            if x is not None:
                self.samples.append(x)
                self.labels.append(window.get(label_col, 0))
                self.subject_ids.append(window.get("subject_id", "unknown"))
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i+1}/{len(windows)} windows")
        
        logger.info(f"Dataset created: {len(self.samples)} samples with shape ({self.n_channels}, {self.seq_len})")
    
    def _prepare_window(self, window: Dict) -> Optional[np.ndarray]:
        """
        Prepare a single window as raw multivariate time series.
        
        Extracts 8 channels from window data and normalizes using subject-level stats.
        
        Args:
            window: Window dictionary with 'window_data' DataFrame
                   and optionally 'subject_stats' for subject-wise normalization
        
        Returns:
            Array of shape (n_channels, seq_len) or None if invalid
        """
        df = window.get("window_data")
        if df is None or len(df) == 0:
            return None
        
        # Get subject-level stats if available
        subject_stats = window.get("subject_stats", {})
        use_subject_stats = (self.normalization_mode == "subject" and len(subject_stats) > 0)
        
        channels = []
        
        for channel_name in self.CHANNELS:
            if channel_name in df.columns:
                values = df[channel_name].values
            else:
                # Try to find column with similar name
                found = False
                for col in df.columns:
                    if channel_name.lower() in col.lower():
                        values = df[col].values
                        found = True
                        break
                
                if not found:
                    # Fill with zeros if channel not found
                    values = np.zeros(len(df))
            
            # Handle NaN values
            values = np.nan_to_num(values, nan=0.0)
            
            # Pad or truncate to seq_len
            if len(values) < self.seq_len:
                # Pad with zeros
                values = np.pad(values, (0, self.seq_len - len(values)), mode='constant')
            elif len(values) > self.seq_len:
                # Truncate
                values = values[:self.seq_len]
            
            # Normalize if requested
            if self.normalize:
                if use_subject_stats and channel_name in subject_stats:
                    # Subject-wise normalization (recommended)
                    mean = subject_stats[channel_name]["mean"]
                    std = subject_stats[channel_name]["std"]
                else:
                    # Fall back to per-window normalization
                    mean = np.mean(values)
                    std = np.std(values)
                
                if std > 0:
                    values = (values - mean) / std
                else:
                    values = values - mean
            
            channels.append(values)
        
        # Stack channels: shape (n_channels, seq_len)
        x = np.stack(channels, axis=0)
        
        return x.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Tuple of (x, y) where:
            - x: Tensor of shape (n_channels, seq_len) for TCN
            - y: Tensor of shape () (scalar label)
        """
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y
    
    def get_subject_ids(self) -> np.ndarray:
        """Get array of subject IDs for each sample."""
        return np.array(self.subject_ids)
    
    def get_n_channels(self) -> int:
        """Get number of channels."""
        return self.n_channels
    
    def get_seq_len(self) -> int:
        """Get sequence length."""
        return self.seq_len


def create_tcn_datasets(windows_by_subject: Dict[str, List[Dict]],
                        test_subject: str,
                        label_col: str = "label_5min",
                        seq_len: int = 120) -> Tuple[VitaStressTCNDataset, VitaStressTCNDataset]:
    """
    Create train and test datasets for LOSO cross-validation.
    
    Args:
        windows_by_subject: Dictionary mapping subject_id to list of windows
        test_subject: Subject ID to hold out for testing
        label_col: Label column to use
        seq_len: Sequence length (default 120)
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_windows = []
    test_windows = []
    
    for subject_id, windows in windows_by_subject.items():
        # Add subject_id to each window
        for w in windows:
            w["subject_id"] = subject_id
        
        if subject_id == test_subject:
            test_windows.extend(windows)
        else:
            train_windows.extend(windows)
    
    train_dataset = VitaStressTCNDataset(train_windows, label_col, seq_len, 
                                         normalize=True, normalization_mode="subject")
    test_dataset = VitaStressTCNDataset(test_windows, label_col, seq_len,
                                        normalize=True, normalization_mode="subject")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test dataset creation
    print("Testing TCN dataset with raw multivariate time series...")
    
    # Create dummy windows
    np.random.seed(42)
    
    dummy_windows = []
    for i in range(10):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=120, freq="1S"),
            "acc_x": np.random.normal(0, 0.5, 120),
            "acc_y": np.random.normal(0, 0.5, 120),
            "acc_z": np.random.normal(1, 0.3, 120),
            "skin_temp": np.random.normal(32, 1, 120),
            "heatflux": np.random.normal(70, 30, 120),
            "cbt": np.random.normal(37.2, 0.1, 120),
            "hr_bpm": np.random.normal(75, 10, 120),
            "rmssd": np.random.normal(50, 15, 120),
        })
        
        dummy_windows.append({
            "window_data": df,
            "label_5min": np.random.randint(0, 2),
            "subject_id": f"subject_{i % 3}"
        })
    
    dataset = VitaStressTCNDataset(dummy_windows)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of channels: {dataset.get_n_channels()}")
    print(f"Sequence length: {dataset.get_seq_len()}")
    print(f"Channels: {dataset.CHANNELS}")
    
    x, y = dataset[0]
    print(f"\nSample shape: {x.shape}")  # Should be (8, 120)
    print(f"Label shape: {y.shape}")
    
    print("\n✅ TCN dataset working!")

