"""
Dataset utilities for Subject-Aware SSL.

Loads and prepares data at 8Hz sampling rate for contrastive learning.
Handles subject ID tracking for subject-aware contrastive sampling.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset, DataLoader

# Add shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.windowing import parse_stress_events, create_labeled_windows

from config import SSLConfig, DEFAULT_SSL_CONFIG


class SSLDataset(Dataset):
    """
    PyTorch Dataset for SSL pre-training and fine-tuning.
    
    Returns samples in format (x, label, subject_id) where:
    - x: Tensor of shape (n_channels, seq_len) at 8Hz
    - label: Binary stress label (0 or 1)
    - subject_id: Integer subject index
    """
    
    def __init__(
        self,
        windows: List[Dict],
        config: SSLConfig,
        subject_to_idx: Optional[Dict[str, int]] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            windows: List of window dictionaries with 8Hz data
            config: SSL configuration
            subject_to_idx: Mapping from subject ID to integer index
            normalize: Whether to normalize features
        """
        self.config = config
        self.normalize = normalize
        
        # Build subject mapping if not provided
        if subject_to_idx is None:
            unique_subjects = sorted(set(w.get("subject_id", "unknown") for w in windows))
            subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
        self.subject_to_idx = subject_to_idx
        
        # Process windows
        self.samples = []
        self.labels = []
        self.subject_ids = []
        self.subject_str_ids = []
        
        for window in windows:
            x = self._process_window(window)
            if x is None:
                continue
            
            self.samples.append(x)
            self.labels.append(window.get(config.target_label, 0))
            
            subject_str = window.get("subject_id", "unknown")
            self.subject_str_ids.append(subject_str)
            self.subject_ids.append(subject_to_idx.get(subject_str, 0))
        
        self.samples = np.stack(self.samples, axis=0).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids, dtype=np.int64)
        
        # Normalize per-channel if requested
        if normalize and len(self.samples) > 0:
            self._normalize()
    
    def _process_window(self, window: Dict) -> Optional[np.ndarray]:
        """
        Process a window dictionary to extract features at 8Hz.
        
        Args:
            window: Window dictionary with 'window_data_8hz' or 'window_data'
        
        Returns:
            Feature array of shape (n_channels, seq_len) or None if invalid
        """
        # Try 8Hz data first
        df = window.get("window_data_8hz", window.get("window_data"))
        
        if df is None or len(df) == 0:
            return None
        
        features = []
        seq_len = self.config.samples_per_window
        
        for feature_name in self.config.feature_names:
            if feature_name in df.columns:
                values = df[feature_name].values
            else:
                # Try to find similar column
                found = False
                for col in df.columns:
                    if feature_name.lower() in col.lower():
                        values = df[col].values
                        found = True
                        break
                if not found:
                    values = np.zeros(len(df))
            
            # Handle NaN
            values = np.nan_to_num(values, nan=0.0)
            
            # Resample/pad to expected length if needed
            if len(values) != seq_len:
                if len(values) > seq_len:
                    values = values[:seq_len]
                else:
                    values = np.pad(values, (0, seq_len - len(values)), mode="edge")
            
            features.append(values)
        
        # Stack: shape (n_channels, seq_len)
        return np.stack(features, axis=0)
    
    def _normalize(self):
        """Normalize samples per-channel using z-score."""
        # Compute mean and std per channel (ensure float32 to avoid promotion to float64)
        mean = self.samples.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = self.samples.std(axis=(0, 2), keepdims=True).astype(np.float32)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        self.samples = ((self.samples - mean) / std).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get sample by index.
        
        Returns:
            Tuple of (x, label, subject_id)
        """
        x = torch.from_numpy(self.samples[idx])
        label = torch.tensor(self.labels[idx])
        subject_id = torch.tensor(self.subject_ids[idx])
        
        return x, label, subject_id
    
    @property
    def n_subjects(self) -> int:
        """Number of unique subjects in dataset."""
        return len(self.subject_to_idx)


def resample_signal(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    target_rate: float
) -> np.ndarray:
    """
    Resample signal to target rate using interpolation.
    
    Args:
        values: Signal values
        timestamps: Original timestamps
        target_start: Start time for resampling
        target_end: End time for resampling
        target_rate: Target sampling rate in Hz
    
    Returns:
        Resampled signal array
    """
    # Convert timestamps to seconds from start
    t_original = (timestamps - target_start).dt.total_seconds().values
    
    # Create target time points
    duration = (target_end - target_start).total_seconds()
    n_samples = int(duration * target_rate)
    t_target = np.linspace(0, duration, n_samples)
    
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 2:
        return np.zeros(n_samples)
    
    # Interpolate
    try:
        interp_func = interp1d(
            t_original[valid_mask],
            values[valid_mask],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        resampled = interp_func(t_target)
    except Exception:
        resampled = np.zeros(n_samples)
    
    return resampled


def align_to_8hz(
    signals: Dict[str, Optional[pd.DataFrame]],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    target_rate: float = 8.0
) -> pd.DataFrame:
    """
    Align all signals to 8Hz time grid.
    
    Handles different source sampling rates:
    - ACC: 32Hz -> 8Hz (downsample) - now extracts acc_x, acc_y, acc_z separately
    - PPG: 64Hz -> 8Hz (downsample)
    - Heatflux (skin_temp, heatflux, cbt): 1Hz -> 8Hz (upsample)
    
    Args:
        signals: Dictionary of signal DataFrames
        start_time: Start time for alignment
        end_time: End time for alignment
        target_rate: Target sampling rate (default 8Hz)
    
    Returns:
        DataFrame with aligned signals at 8Hz
    """
    duration = (end_time - start_time).total_seconds()
    n_samples = int(duration * target_rate)
    
    # Create target time index
    time_index = pd.date_range(start=start_time, end=end_time, periods=n_samples)
    
    aligned = pd.DataFrame({"timestamp": time_index})
    
    # Process accelerometer (32Hz -> 8Hz) - extract X, Y, Z separately
    acc_df = signals.get("acc")
    if acc_df is not None and len(acc_df) > 0:
        # Extract each axis separately for better directional info
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in acc_df.columns:
                aligned[col] = resample_signal(
                    acc_df[col].values, acc_df["timestamp"], 
                    start_time, end_time, target_rate
                )
            else:
                aligned[col] = 0.0
    else:
        aligned["acc_x"] = 0.0
        aligned["acc_y"] = 0.0
        aligned["acc_z"] = 0.0
    
    # Process heatflux data (1Hz -> 8Hz) - includes skin_temp, heatflux, cbt
    hf_df = signals.get("heatflux")
    if hf_df is not None and len(hf_df) > 0:
        # Skin temperature
        if "skin_temp" in hf_df.columns:
            aligned["skin_temp"] = resample_signal(
                hf_df["skin_temp"].values,
                hf_df["timestamp"],
                start_time, end_time, target_rate
            )
        else:
            aligned["skin_temp"] = 0.0
        
        # Heat flux - thermal energy transfer rate
        if "heatflux" in hf_df.columns:
            aligned["heatflux"] = resample_signal(
                hf_df["heatflux"].values,
                hf_df["timestamp"],
                start_time, end_time, target_rate
            )
        else:
            aligned["heatflux"] = 0.0
        
        # Core body temperature
        if "cbt" in hf_df.columns:
            aligned["cbt"] = resample_signal(
                hf_df["cbt"].values,
                hf_df["timestamp"],
                start_time, end_time, target_rate
            )
        else:
            aligned["cbt"] = 0.0
    else:
        aligned["skin_temp"] = 0.0
        aligned["heatflux"] = 0.0
        aligned["cbt"] = 0.0
    
    # Process PPG (64Hz -> 8Hz)
    ppg_df = signals.get("ppg")
    if ppg_df is not None and len(ppg_df) > 0:
        if "value" in ppg_df.columns:
            aligned["ppg_mean"] = resample_signal(
                ppg_df["value"].values,
                ppg_df["timestamp"],
                start_time, end_time, target_rate
            )
        else:
            aligned["ppg_mean"] = 0.0
    else:
        aligned["ppg_mean"] = 0.0
    
    return aligned


def load_windows_at_8hz(
    config: SSLConfig,
    subjects: Optional[List[Path]] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Load all windows from subjects at 8Hz sampling rate.
    
    Args:
        config: SSL configuration
        subjects: List of subject folders (default: all subjects)
    
    Returns:
        Tuple of (windows, subject_to_idx)
    """
    from tqdm.auto import tqdm
    
    if subjects is None:
        subjects = get_all_subjects(config.data_path)
    
    all_windows = []
    subject_to_idx = {}
    
    for idx, subject_folder in enumerate(tqdm(subjects, desc="Loading subjects")):
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        subject_to_idx[subject_id] = idx
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            continue
        
        # Align to 8Hz
        aligned = align_to_8hz(signals, start, end, config.ssl_sample_rate)
        
        if aligned is None or len(aligned) == 0:
            continue
        
        # Parse stress events
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        # Create windows
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=config.window_size_sec,
            overlap_ratio=config.overlap_ratio,
            horizons_minutes=config.horizons_minutes,
            skip_first_minutes=config.skip_first_minutes
        )
        
        # Add subject ID and 8Hz data
        for w in windows:
            w["subject_id"] = subject_id
            w["window_data_8hz"] = w.get("window_data")
        
        all_windows.extend(windows)
    
    return all_windows, subject_to_idx


def create_ssl_dataloaders(
    windows: List[Dict],
    config: SSLConfig,
    subject_to_idx: Dict[str, int],
    batch_size: Optional[int] = None,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader for SSL training.
    
    Args:
        windows: List of window dictionaries
        config: SSL configuration
        subject_to_idx: Subject ID to index mapping
        batch_size: Batch size (default from config)
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    if batch_size is None:
        batch_size = config.pretrain_batch_size
    
    dataset = SSLDataset(windows, config, subject_to_idx)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    return dataloader


def create_loso_splits(
    windows: List[Dict],
    config: SSLConfig,
    subject_to_idx: Dict[str, int],
    test_subject_id: str
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/test splits for LOSO cross-validation.
    
    Args:
        windows: All windows
        config: SSL configuration
        subject_to_idx: Subject ID to index mapping
        test_subject_id: Subject to hold out for testing
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_windows = [w for w in windows if w.get("subject_id") != test_subject_id]
    test_windows = [w for w in windows if w.get("subject_id") == test_subject_id]
    
    train_dataset = SSLDataset(train_windows, config, subject_to_idx)
    test_dataset = SSLDataset(test_windows, config, subject_to_idx, normalize=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.finetune_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.finetune_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing SSL Dataset...")
    
    config = DEFAULT_SSL_CONFIG
    
    # Load a subset of data
    subjects = get_all_subjects(config.data_path)[:3]
    print(f"Testing with {len(subjects)} subjects")
    
    windows, subject_to_idx = load_windows_at_8hz(config, subjects)
    print(f"Loaded {len(windows)} windows")
    print(f"Subject mapping: {subject_to_idx}")
    
    if windows:
        # Create dataset
        dataset = SSLDataset(windows, config, subject_to_idx)
        print(f"\nDataset size: {len(dataset)}")
        print(f"N subjects: {dataset.n_subjects}")
        print(f"Label distribution: {np.bincount(dataset.labels)}")
        
        # Test sample
        x, label, subject_id = dataset[0]
        print(f"\nSample shapes:")
        print(f"  x: {x.shape}")
        print(f"  label: {label}")
        print(f"  subject_id: {subject_id}")
        
        # Test dataloader
        dataloader = create_ssl_dataloaders(windows, config, subject_to_idx, batch_size=8)
        
        for batch_x, batch_y, batch_s in dataloader:
            print(f"\nBatch shapes:")
            print(f"  x: {batch_x.shape}")
            print(f"  y: {batch_y.shape}")
            print(f"  subject_ids: {batch_s.shape}")
            break
    
    print("\nAll tests passed!")
