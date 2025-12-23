"""
Dataset for Multi-Rate Late Fusion.

Loads signals at their native sampling rates (Same 8 channels as MOMENT):
- ACC: 32 Hz (3 channels: acc_x, acc_y, acc_z)
- Physio: 1 Hz (5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd)

Note: PPG is disabled. Using HR and HRV (RMSSD) instead, which are more meaningful at 1Hz.

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
    t_data = (window_timestamps - target_start).dt.total_seconds().values
    
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
    Same 8 channels as MOMENT but at native rates:
    - ACC (32Hz): 3 channels
    - Physio (1Hz): 5 channels
    
    Args:
        signals: Dictionary of signal DataFrames
        window_start: Window start time
        window_end: Window end time
        config: Multi-rate configuration
    
    Returns:
        Dictionary with:
        - acc: (3, n_acc_samples) at 32Hz - acc_x, acc_y, acc_z
        - physio: (5, n_physio_samples) at 1Hz - skin_temp, heatflux, cbt, hr_bpm, rmssd
    """
    result = {}
    
    # Helper function to ensure correct length
    def ensure_length(arr, target_len):
        """Pad or truncate array to target length."""
        if len(arr) > target_len:
            return arr[:target_len]
        elif len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)), mode='edge')
        return arr
    
    # # PPG at 64Hz (DISABLED - using HR/HRV instead)
    # ppg_df = signals.get("ppg")
    # if ppg_df is not None and "value" in ppg_df.columns:
    #     result["ppg"] = extract_signal_at_native_rate(
    #         ppg_df, "value", ppg_df["timestamp"],
    #         window_start, window_end, config.ppg_sample_rate
    #     )
    # else:
    #     result["ppg"] = np.zeros(config.ppg_samples_per_window, dtype=np.float32)
    
    # ACC at 32Hz (3 channels: acc_x, acc_y, acc_z)
    acc_df = signals.get("acc")
    if acc_df is not None:
        acc_channels = []
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in acc_df.columns:
                channel = extract_signal_at_native_rate(
                    acc_df, col, acc_df["timestamp"],
                    window_start, window_end, config.acc_sample_rate
                )
                channel = ensure_length(channel, config.acc_samples_per_window)
            else:
                channel = np.zeros(config.acc_samples_per_window, dtype=np.float32)
            acc_channels.append(channel)
        result["acc"] = np.stack(acc_channels, axis=0)
    else:
        result["acc"] = np.zeros((3, config.acc_samples_per_window), dtype=np.float32)
    
    # Physiological signals at 1Hz (5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd)
    physio_channels = []
    
    # 1. skin_temp from heatflux sensor
    hf_df = signals.get("heatflux")
    if hf_df is not None and "skin_temp" in hf_df.columns:
        skin_temp = extract_signal_at_native_rate(
            hf_df, "skin_temp", hf_df["timestamp"],
            window_start, window_end, config.physio_sample_rate
        )
        skin_temp = ensure_length(skin_temp, config.physio_samples_per_window)
    else:
        skin_temp = np.zeros(config.physio_samples_per_window, dtype=np.float32)
    physio_channels.append(skin_temp)
    
    # 2. heatflux from heatflux sensor
    if hf_df is not None and "heatflux" in hf_df.columns:
        heatflux = extract_signal_at_native_rate(
            hf_df, "heatflux", hf_df["timestamp"],
            window_start, window_end, config.physio_sample_rate
        )
        heatflux = ensure_length(heatflux, config.physio_samples_per_window)
    else:
        heatflux = np.zeros(config.physio_samples_per_window, dtype=np.float32)
    physio_channels.append(heatflux)
    
    # 3. cbt (core body temperature) from heatflux sensor
    if hf_df is not None and "cbt" in hf_df.columns:
        cbt = extract_signal_at_native_rate(
            hf_df, "cbt", hf_df["timestamp"],
            window_start, window_end, config.physio_sample_rate
        )
        cbt = ensure_length(cbt, config.physio_samples_per_window)
    else:
        cbt = np.zeros(config.physio_samples_per_window, dtype=np.float32)
    physio_channels.append(cbt)
    
    # 4. hr_bpm from aligned data (already at 1Hz)
    aligned_df = signals.get("aligned")
    if aligned_df is not None and "hr_bpm" in aligned_df.columns:
        hr_bpm = extract_signal_at_native_rate(
            aligned_df, "hr_bpm", aligned_df["timestamp"],
            window_start, window_end, config.physio_sample_rate
        )
        hr_bpm = ensure_length(hr_bpm, config.physio_samples_per_window)
    else:
        hr_bpm = np.zeros(config.physio_samples_per_window, dtype=np.float32)
    physio_channels.append(hr_bpm)
    
    # 5. rmssd (HRV) from aligned data (already at 1Hz)
    if aligned_df is not None and "rmssd" in aligned_df.columns:
        rmssd = extract_signal_at_native_rate(
            aligned_df, "rmssd", aligned_df["timestamp"],
            window_start, window_end, config.physio_sample_rate
        )
        rmssd = ensure_length(rmssd, config.physio_samples_per_window)
    else:
        rmssd = np.zeros(config.physio_samples_per_window, dtype=np.float32)
    physio_channels.append(rmssd)
    
    result["physio"] = np.stack(physio_channels, axis=0)  # (5, n_physio_samples)
    
    return result


class MultiRateDataset(Dataset):
    """
    PyTorch Dataset for Multi-Rate Late Fusion.
    
    Returns samples as dictionary of modality tensors (Same 8 channels as MOMENT):
    - acc: (3, acc_samples) at 32Hz - acc_x, acc_y, acc_z
    - physio: (5, physio_samples) at 1Hz - skin_temp, heatflux, cbt, hr_bpm, rmssd
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
        self.acc_data = []
        self.physio_data = []
        self.labels = []
        self.subject_ids = []
        
        for window in windows:
            multirate = window.get("multirate_data")
            if multirate is None:
                continue
            
            acc = multirate.get("acc", np.zeros((3, config.acc_samples_per_window)))
            physio = multirate.get("physio", np.zeros((5, config.physio_samples_per_window)))
            
            # Ensure correct shapes
            if acc.shape[1] != config.acc_samples_per_window:
                acc = np.stack([
                    self._pad_or_truncate(acc[i], config.acc_samples_per_window)
                    for i in range(3)
                ])
            if physio.shape[1] != config.physio_samples_per_window:
                physio = np.stack([
                    self._pad_or_truncate(physio[i], config.physio_samples_per_window)
                    for i in range(5)
                ])
            
            self.acc_data.append(acc)
            self.physio_data.append(physio)
            self.labels.append(window.get(config.target_label, 0))
            
            subject_str = window.get("subject_id", "unknown")
            self.subject_ids.append(subject_to_idx.get(subject_str, 0))
        
        # Convert to arrays
        self.acc_data = np.stack(self.acc_data).astype(np.float32)
        self.physio_data = np.stack(self.physio_data).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids, dtype=np.int64)
        
        # Normalize
        if normalize and len(self.acc_data) > 0:
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
        self.acc_data = np.nan_to_num(self.acc_data, nan=0.0).astype(np.float32)
        self.physio_data = np.nan_to_num(self.physio_data, nan=0.0).astype(np.float32)
        
        # ACC: per-channel normalization (3 channels: acc_x, acc_y, acc_z)
        for i in range(3):
            mean = self.acc_data[:, i, :].mean()
            std = self.acc_data[:, i, :].std()
            if std > 0:
                self.acc_data[:, i, :] = ((self.acc_data[:, i, :] - mean) / std).astype(np.float32)
        
        # Physio: per-channel normalization (5 channels: skin_temp, heatflux, cbt, hr_bpm, rmssd)
        for i in range(5):
            mean = self.physio_data[:, i, :].mean()
            std = self.physio_data[:, i, :].std()
            if std > 0:
                self.physio_data[:, i, :] = ((self.physio_data[:, i, :] - mean) / std).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.
        
        Returns:
            Dictionary with acc, physio, label, subject_id tensors
        """
        return {
            "acc": torch.from_numpy(self.acc_data[idx]),
            "physio": torch.from_numpy(self.physio_data[idx]),
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
        "acc": torch.stack([b["acc"] for b in batch]),
        "physio": torch.stack([b["physio"] for b in batch]),
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
        print(f"  ACC: {sample['acc'].shape}")
        print(f"  Physio: {sample['physio'].shape}")
        print(f"  Label: {sample['label']}")
        
        # Test DataLoader
        loader = DataLoader(
            dataset, batch_size=4, shuffle=True,
            collate_fn=collate_multirate
        )
        
        for batch in loader:
            print(f"\nBatch shapes:")
            print(f"  ACC: {batch['acc'].shape}")
            print(f"  Physio: {batch['physio'].shape}")
            break
    
    print("\nAll tests passed!")

