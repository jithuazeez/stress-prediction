"""
Dataset for Multi-Rate Late Fusion.

Loads signals at their native sampling rates:
- PPG: 64 Hz
- ACC: 32 Hz (3 channels)
- Temp: 1 Hz

Each sample contains separately shaped arrays for each modality.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset, DataLoader

# Add shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.windowing import parse_stress_events, create_labeled_windows

from config import MultiRateConfig, DEFAULT_MULTIRATE_CONFIG


def extract_signal_at_native_rate(
    df: pd.DataFrame,
    value_col: str,
    timestamps: pd.DatetimeIndex,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    target_rate: float
) -> np.ndarray:
    """
    Extract signal at its native rate within a time window.
    
    Args:
        df: DataFrame with signal data
        value_col: Column name for signal values
        timestamps: Timestamp column from df
        target_start: Window start time
        target_end: Window end time
        target_rate: Expected sampling rate
    
    Returns:
        Signal array at native rate
    """
    # Filter to window
    mask = (timestamps >= target_start) & (timestamps < target_end)
    window_df = df[mask]
    
    if len(window_df) == 0:
        n_expected = int((target_end - target_start).total_seconds() * target_rate)
        return np.zeros(n_expected)
    
    values = window_df[value_col].values
    window_timestamps = timestamps[mask]
    
    # Convert to seconds from start
    t_data = (window_timestamps - target_start).total_seconds().values
    
    # Create target time grid
    duration = (target_end - target_start).total_seconds()
    n_samples = int(duration * target_rate)
    t_target = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Handle NaN
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 2:
        return np.zeros(n_samples)
    
    # Interpolate to regular grid
    try:
        interp_func = interp1d(
            t_data[valid_mask],
            values[valid_mask],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        resampled = interp_func(t_target)
    except Exception:
        resampled = np.zeros(n_samples)
    
    return resampled.astype(np.float32)


def load_multirate_window(
    signals: Dict[str, Optional[pd.DataFrame]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    config: MultiRateConfig
) -> Dict[str, np.ndarray]:
    """
    Load a single window with signals at native rates.
    
    Args:
        signals: Dictionary of signal DataFrames
        window_start: Window start time
        window_end: Window end time
        config: Multi-rate configuration
    
    Returns:
        Dictionary with:
        - ppg: (n_ppg_samples,) at 64Hz
        - acc: (3, n_acc_samples) at 32Hz
        - temp: (n_temp_samples,) at 1Hz
    """
    result = {}
    
    # PPG at 64Hz
    ppg_df = signals.get("ppg")
    if ppg_df is not None and "value" in ppg_df.columns:
        result["ppg"] = extract_signal_at_native_rate(
            ppg_df, "value", ppg_df["timestamp"],
            window_start, window_end, config.ppg_sample_rate
        )
    else:
        result["ppg"] = np.zeros(config.ppg_samples_per_window, dtype=np.float32)
    
    # ACC at 32Hz (3 channels)
    acc_df = signals.get("acc")
    if acc_df is not None:
        acc_channels = []
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in acc_df.columns:
                channel = extract_signal_at_native_rate(
                    acc_df, col, acc_df["timestamp"],
                    window_start, window_end, config.acc_sample_rate
                )
            else:
                channel = np.zeros(config.acc_samples_per_window, dtype=np.float32)
            acc_channels.append(channel)
        result["acc"] = np.stack(acc_channels, axis=0)
    else:
        result["acc"] = np.zeros((3, config.acc_samples_per_window), dtype=np.float32)
    
    # Temperature at 1Hz
    hf_df = signals.get("heatflux")
    if hf_df is not None and "skin_temp" in hf_df.columns:
        result["temp"] = extract_signal_at_native_rate(
            hf_df, "skin_temp", hf_df["timestamp"],
            window_start, window_end, config.temp_sample_rate
        )
    else:
        result["temp"] = np.zeros(config.temp_samples_per_window, dtype=np.float32)
    
    return result


class MultiRateDataset(Dataset):
    """
    PyTorch Dataset for Multi-Rate Late Fusion.
    
    Returns samples as dictionary of modality tensors:
    - ppg: (ppg_samples,) at 64Hz
    - acc: (3, acc_samples) at 32Hz
    - temp: (temp_samples,) at 1Hz
    - label: Binary label
    - subject_id: Integer subject index
    """
    
    def __init__(
        self,
        windows: List[Dict],
        config: MultiRateConfig,
        subject_to_idx: Optional[Dict[str, int]] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            windows: List of window dictionaries with native-rate data
            config: Multi-rate configuration
            subject_to_idx: Subject ID to index mapping
            normalize: Whether to normalize features
        """
        self.config = config
        self.normalize = normalize
        
        # Build subject mapping
        if subject_to_idx is None:
            unique_subjects = sorted(set(w.get("subject_id", "unknown") for w in windows))
            subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
        self.subject_to_idx = subject_to_idx
        
        # Process windows
        self.ppg_data = []
        self.acc_data = []
        self.temp_data = []
        self.labels = []
        self.subject_ids = []
        
        for window in windows:
            multirate = window.get("multirate_data")
            if multirate is None:
                continue
            
            ppg = multirate.get("ppg", np.zeros(config.ppg_samples_per_window))
            acc = multirate.get("acc", np.zeros((3, config.acc_samples_per_window)))
            temp = multirate.get("temp", np.zeros(config.temp_samples_per_window))
            
            # Ensure correct shapes
            if len(ppg) != config.ppg_samples_per_window:
                ppg = self._pad_or_truncate(ppg, config.ppg_samples_per_window)
            if acc.shape[1] != config.acc_samples_per_window:
                acc = np.stack([
                    self._pad_or_truncate(acc[i], config.acc_samples_per_window)
                    for i in range(3)
                ])
            if len(temp) != config.temp_samples_per_window:
                temp = self._pad_or_truncate(temp, config.temp_samples_per_window)
            
            self.ppg_data.append(ppg)
            self.acc_data.append(acc)
            self.temp_data.append(temp)
            self.labels.append(window.get(config.target_label, 0))
            
            subject_str = window.get("subject_id", "unknown")
            self.subject_ids.append(subject_to_idx.get(subject_str, 0))
        
        # Convert to arrays
        self.ppg_data = np.stack(self.ppg_data).astype(np.float32)
        self.acc_data = np.stack(self.acc_data).astype(np.float32)
        self.temp_data = np.stack(self.temp_data).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids, dtype=np.int64)
        
        # Normalize
        if normalize and len(self.ppg_data) > 0:
            self._normalize()
    
    def _pad_or_truncate(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or truncate array to target length."""
        if len(arr) > target_len:
            return arr[:target_len]
        elif len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)), mode="edge")
        return arr
    
    def _normalize(self):
        """Normalize each modality."""
        # Handle NaN
        self.ppg_data = np.nan_to_num(self.ppg_data, nan=0.0)
        self.acc_data = np.nan_to_num(self.acc_data, nan=0.0)
        self.temp_data = np.nan_to_num(self.temp_data, nan=0.0)
        
        # PPG: per-sample normalization (removes baseline drift)
        ppg_mean = self.ppg_data.mean(axis=1, keepdims=True)
        ppg_std = self.ppg_data.std(axis=1, keepdims=True)
        ppg_std = np.where(ppg_std == 0, 1, ppg_std)
        self.ppg_data = (self.ppg_data - ppg_mean) / ppg_std
        
        # ACC: per-channel normalization
        for i in range(3):
            mean = self.acc_data[:, i, :].mean()
            std = self.acc_data[:, i, :].std()
            if std > 0:
                self.acc_data[:, i, :] = (self.acc_data[:, i, :] - mean) / std
        
        # Temp: global normalization
        temp_mean = self.temp_data.mean()
        temp_std = self.temp_data.std()
        if temp_std > 0:
            self.temp_data = (self.temp_data - temp_mean) / temp_std
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.
        
        Returns:
            Dictionary with ppg, acc, temp, label, subject_id tensors
        """
        return {
            "ppg": torch.from_numpy(self.ppg_data[idx]),
            "acc": torch.from_numpy(self.acc_data[idx]),
            "temp": torch.from_numpy(self.temp_data[idx]),
            "label": torch.tensor(self.labels[idx]),
            "subject_id": torch.tensor(self.subject_ids[idx])
        }
    
    @property
    def n_subjects(self) -> int:
        return len(self.subject_to_idx)


def load_windows_multirate(
    config: MultiRateConfig,
    subjects: Optional[List[Path]] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Load all windows with signals at native sampling rates.
    
    Args:
        config: Multi-rate configuration
        subjects: List of subject folders (default: all)
    
    Returns:
        Tuple of (windows, subject_to_idx)
    """
    from tqdm import tqdm
    
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
        
        # Parse stress events
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        # Create window time ranges (using 1Hz aligned for labels)
        # Create a simple 1Hz time grid for window creation
        duration = (end - start).total_seconds()
        n_samples = int(duration)
        time_index = pd.date_range(start=start, end=end, periods=n_samples)
        
        dummy_aligned = pd.DataFrame({
            "timestamp": time_index,
            "dummy": np.zeros(n_samples)
        })
        
        windows = create_labeled_windows(
            dummy_aligned,
            event_info,
            window_size_sec=config.window_size_sec,
            overlap_ratio=config.overlap_ratio,
            horizons_minutes=config.horizons_minutes,
            skip_first_minutes=config.skip_first_minutes
        )
        
        # For each window, extract native-rate signals
        for w in windows:
            window_data = w.get("window_data")
            if window_data is None or len(window_data) == 0:
                continue
            
            window_start = window_data["timestamp"].iloc[0]
            window_end = window_data["timestamp"].iloc[-1]
            
            # Load native-rate data
            multirate = load_multirate_window(signals, window_start, window_end, config)
            
            w["subject_id"] = subject_id
            w["multirate_data"] = multirate
        
        all_windows.extend(windows)
    
    return all_windows, subject_to_idx


def create_loso_dataloaders(
    windows: List[Dict],
    config: MultiRateConfig,
    subject_to_idx: Dict[str, int],
    test_subject_id: str
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/test DataLoaders for LOSO CV.
    
    Args:
        windows: All windows
        config: Multi-rate configuration
        subject_to_idx: Subject ID to index mapping
        test_subject_id: Subject to hold out
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_windows = [w for w in windows if w.get("subject_id") != test_subject_id]
    test_windows = [w for w in windows if w.get("subject_id") == test_subject_id]
    
    train_dataset = MultiRateDataset(train_windows, config, subject_to_idx)
    test_dataset = MultiRateDataset(test_windows, config, subject_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def collate_multirate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for multi-rate batches."""
    return {
        "ppg": torch.stack([b["ppg"] for b in batch]),
        "acc": torch.stack([b["acc"] for b in batch]),
        "temp": torch.stack([b["temp"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "subject_id": torch.stack([b["subject_id"] for b in batch])
    }


if __name__ == "__main__":
    # Test dataset
    print("Testing Multi-Rate Dataset...")
    
    config = DEFAULT_MULTIRATE_CONFIG
    
    # Load subset
    subjects = get_all_subjects(config.data_path)[:3]
    print(f"Testing with {len(subjects)} subjects")
    
    windows, subject_to_idx = load_windows_multirate(config, subjects)
    print(f"Loaded {len(windows)} windows")
    
    if windows:
        dataset = MultiRateDataset(windows, config, subject_to_idx)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Label distribution: {np.bincount(dataset.labels)}")
        
        # Test sample
        sample = dataset[0]
        print(f"\nSample shapes:")
        print(f"  PPG: {sample['ppg'].shape}")
        print(f"  ACC: {sample['acc'].shape}")
        print(f"  Temp: {sample['temp'].shape}")
        print(f"  Label: {sample['label']}")
        
        # Test DataLoader
        loader = DataLoader(
            dataset, batch_size=4, shuffle=True,
            collate_fn=collate_multirate
        )
        
        for batch in loader:
            print(f"\nBatch shapes:")
            print(f"  PPG: {batch['ppg'].shape}")
            print(f"  ACC: {batch['acc'].shape}")
            print(f"  Temp: {batch['temp'].shape}")
            break
    
    print("\nAll tests passed!")
