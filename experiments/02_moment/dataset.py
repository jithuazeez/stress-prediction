"""
PyTorch Dataset for MOMENT foundation model.

Prepares multivariate time series data for MOMENT classification.
MOMENT expects input shape: [batch, n_channels, seq_len] where seq_len = 512

We use 3 channels:
- acc_magnitude: Accelerometer magnitude
- skin_temp: Skin temperature
- eda_stress_skin: EDA/skin conductance

Note: PPG is excluded because downsampling from 64Hz to 1Hz destroys
the cardiac waveform, making it useless for stress detection.

References:
- https://github.com/moment-timeseries-foundation-model/moment
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.signal import resample

# Add shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config


class VitaStressMOMENTDataset(Dataset):
    """
    Dataset for MOMENT foundation model.
    
    Converts aligned 1Hz windows to MOMENT-compatible format.
    Resamples to 512 samples per window (MOMENT's expected length).
    """
    
    # Channel names in order
    # PPG excluded: 1Hz downsampling destroys cardiac waveform (64Hz -> 1Hz loses heartbeats)
    CHANNELS = ["acc_magnitude", "skin_temp", "eda_stress_skin"]  # 3 channels
    # CHANNELS = ["acc_magnitude", "skin_temp", "eda_stress_skin", "ppg_mean"]  # 4 channels (PPG disabled)
    
    def __init__(self, 
                 windows: List[Dict],
                 label_col: str = "label_5min",
                 seq_len: int = 512,
                 normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            windows: List of window dictionaries from create_labeled_windows()
            label_col: Label column to use
            seq_len: Target sequence length (MOMENT uses 512)
            normalize: Whether to normalize each channel
        """
        self.windows = windows
        self.label_col = label_col
        self.seq_len = seq_len
        self.normalize = normalize
        self.n_channels = len(self.CHANNELS)
        
        # Pre-process all windows
        self.samples = []
        self.labels = []
        self.subject_ids = []
        
        for window in windows:
            x = self._prepare_window(window)
            if x is not None:
                self.samples.append(x)
                self.labels.append(window.get(label_col, 0))
                self.subject_ids.append(window.get("subject_id", "unknown"))
    
    def _prepare_window(self, window: Dict) -> Optional[np.ndarray]:
        """
        Prepare a single window for MOMENT.
        
        Args:
            window: Window dictionary with 'window_data' DataFrame
        
        Returns:
            Array of shape (n_channels, seq_len) or None if invalid
        """
        df = window.get("window_data")
        if df is None or len(df) == 0:
            return None
        
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
            
            # Resample to seq_len using scipy
            if len(values) != self.seq_len:
                values = resample(values, self.seq_len)
            
            # Normalize if requested
            if self.normalize:
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
            - x: Tensor of shape (n_channels, seq_len)
            - y: Tensor of shape () (scalar label)
        """
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    def get_subject_ids(self) -> np.ndarray:
        """Get array of subject IDs for each sample."""
        return np.array(self.subject_ids)


def create_moment_datasets(windows_by_subject: Dict[str, List[Dict]],
                           test_subject: str,
                           label_col: str = "label_5min",
                           seq_len: int = 512) -> Tuple[VitaStressMOMENTDataset, VitaStressMOMENTDataset]:
    """
    Create train and test datasets for LOSO cross-validation.
    
    Args:
        windows_by_subject: Dictionary mapping subject_id to list of windows
        test_subject: Subject ID to hold out for testing
        label_col: Label column to use
        seq_len: Target sequence length
    
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
    
    train_dataset = VitaStressMOMENTDataset(train_windows, label_col, seq_len)
    test_dataset = VitaStressMOMENTDataset(test_windows, label_col, seq_len)
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test dataset creation
    import pandas as pd
    
    print("Testing MOMENT dataset...")
    
    # Create dummy windows
    np.random.seed(42)
    
    dummy_windows = []
    for i in range(10):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=120, freq="1S"),
            "acc_magnitude": np.random.normal(1, 0.1, 120),
            "skin_temp": np.random.normal(32, 1, 120),
            "eda_stress_skin": np.random.normal(2, 0.5, 120),
            "ppg_mean": np.random.normal(30000, 1000, 120),
        })
        
        dummy_windows.append({
            "window_data": df,
            "label_5min": np.random.randint(0, 2),
            "subject_id": f"subject_{i % 3}"
        })
    
    dataset = VitaStressMOMENTDataset(dummy_windows)
    
    print(f"Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Sample shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Channels: {dataset.CHANNELS}")

