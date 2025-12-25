"""
PyTorch Dataset for TCN model with on-the-fly feature extraction.

Unlike MOMENT which uses raw time series, TCN uses extracted features
similar to classical ML approach (excluding HR/HRV which are pre-computed).

Features extracted per window (~65 total, excluding HR/HRV):
- Accelerometer (41): stress indicators + activity classification
- Temperature (7): skin_temp stats, slope, change
- Heat Flux (9): heatflux + CBT enhanced features

Note: HR/HRV features are excluded to match MOMENT's feature set.
These are pre-computed from 64Hz PPG before windowing.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.config import Config

# Import feature extractor from classical ML experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "01_classical_ml"))
from feature_extraction import BasicFeatureExtractor

logger = logging.getLogger(__name__)


class VitaStressTCNDataset(Dataset):
    """
    Dataset for TCN model with on-the-fly feature extraction.
    
    Converts aligned 1Hz windows to feature vectors using the same
    feature extraction pipeline as classical ML (excluding HR/HRV).
    
    Features (~65 total, excluding HR/HRV):
    - Accelerometer (41): stress + activity features
    - Temperature (7): skin_temp statistics
    - Heat Flux (9): heatflux + CBT features
    
    Output shape per window: [n_features, seq_len=1]
    TCN expects input: [batch, n_features, seq_len]
    """
    
    def __init__(self, 
                 windows: List[Dict],
                 label_col: str = "label_5min",
                 normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            windows: List of window dictionaries from create_labeled_windows()
            label_col: Label column to use
            normalize: Whether to normalize features (z-score)
        """
        self.windows = windows
        self.label_col = label_col
        self.normalize = normalize
        
        # Initialize feature extractor
        self.feature_extractor = BasicFeatureExtractor(default_sampling_rate=1.0)
        
        # Pre-compute all features for faster training
        self.samples = []
        self.labels = []
        self.subject_ids = []
        
        logger.info(f"Extracting features from {len(windows)} windows...")
        
        for i, window in enumerate(windows):
            features = self._extract_window_features(window)
            if features is not None:
                self.samples.append(features)
                self.labels.append(window.get(label_col, 0))
                self.subject_ids.append(window.get("subject_id", "unknown"))
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i+1}/{len(windows)} windows")
        
        logger.info(f"Dataset created: {len(self.samples)} samples with {len(features)} features each")
        
        # Compute normalization statistics if needed
        if self.normalize and len(self.samples) > 0:
            self._compute_normalization_stats()
    
    def _extract_window_features(self, window: Dict) -> Optional[np.ndarray]:
        """
        Extract features from a single window.
        
        Excludes HR/HRV features to match MOMENT's feature set.
        
        Args:
            window: Window dictionary with 'window_data' DataFrame
        
        Returns:
            Array of shape (n_features,) or None if invalid
        """
        df = window.get("window_data")
        if df is None or len(df) == 0:
            return None
        
        # Extract features (excluding HR/HRV by not passing hr_hrv_features)
        features_dict = self.feature_extractor.extract_from_window(df, hr_hrv_features=None)
        
        # Filter out HR/HRV features if they ended up in the dict
        features_dict = {
            k: v for k, v in features_dict.items() 
            if not k.startswith("hr_") and not k.startswith("hrv_") and k != "breathing_rate"
        }
        
        # Convert to array
        feature_names = sorted(features_dict.keys())  # Sort for consistency
        features = np.array([features_dict[name] for name in feature_names], dtype=np.float32)
        
        # Handle NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization across all samples."""
        # Stack all samples
        all_features = np.stack(self.samples, axis=0)  # Shape: (n_samples, n_features)
        
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)
        
        # Avoid division by zero
        self.feature_std[self.feature_std == 0] = 1.0
        
        logger.info(f"Normalization stats computed: mean={self.feature_mean.mean():.3f}, std={self.feature_std.mean():.3f}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Tuple of (x, y) where:
            - x: Tensor of shape (n_features, seq_len=1) for TCN
            - y: Tensor of shape () (scalar label)
        """
        features = self.samples[idx].copy()
        
        # Normalize if requested
        if self.normalize:
            features = (features - self.feature_mean) / self.feature_std
        
        # Reshape for TCN: (n_features,) -> (n_features, 1)
        # TCN expects (n_channels, seq_len) where each feature is a "channel"
        x = torch.tensor(features[:, np.newaxis], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y
    
    def get_subject_ids(self) -> np.ndarray:
        """Get array of subject IDs for each sample."""
        return np.array(self.subject_ids)
    
    def get_n_features(self) -> int:
        """Get number of features."""
        if len(self.samples) > 0:
            return self.samples[0].shape[0]
        return 0


def create_tcn_datasets(windows_by_subject: Dict[str, List[Dict]],
                        test_subject: str,
                        label_col: str = "label_5min") -> Tuple[VitaStressTCNDataset, VitaStressTCNDataset]:
    """
    Create train and test datasets for LOSO cross-validation.
    
    Args:
        windows_by_subject: Dictionary mapping subject_id to list of windows
        test_subject: Subject ID to hold out for testing
        label_col: Label column to use
    
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
    
    train_dataset = VitaStressTCNDataset(train_windows, label_col, normalize=True)
    test_dataset = VitaStressTCNDataset(test_windows, label_col, normalize=True)
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test dataset creation
    print("Testing TCN dataset with on-the-fly feature extraction...")
    
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
        })
        
        dummy_windows.append({
            "window_data": df,
            "label_5min": np.random.randint(0, 2),
            "subject_id": f"subject_{i % 3}"
        })
    
    dataset = VitaStressTCNDataset(dummy_windows)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of features: {dataset.get_n_features()}")
    
    x, y = dataset[0]
    print(f"Sample shape: {x.shape}")  # Should be (n_features, 1)
    print(f"Label shape: {y.shape}")
    
    print("\n✅ TCN dataset working!")

