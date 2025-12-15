"""
Dataset utilities for TS2Vec.

TS2Vec expects input shape: [n_samples, seq_len, n_features]
(different from MOMENT which uses channels-first)

References:
- https://github.com/zhihanyue/ts2vec
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.signal import resample

# Add shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config


# Feature columns to use (same as MOMENT but for consistency)
FEATURE_COLUMNS = ["acc_magnitude", "skin_temp", "eda_stress_skin", "ppg_mean"]


def prepare_ts2vec_data(windows: List[Dict],
                        label_col: str = "label_5min",
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for TS2Vec.
    
    TS2Vec expects: [n_samples, seq_len, n_features]
    
    Args:
        windows: List of window dictionaries
        label_col: Label column to use
        normalize: Whether to normalize each feature
    
    Returns:
        Tuple of (X, y, subject_ids) where:
        - X: Array of shape (n_samples, seq_len, n_features)
        - y: Array of labels
        - subject_ids: Array of subject IDs
    """
    samples = []
    labels = []
    subject_ids = []
    
    for window in windows:
        df = window.get("window_data")
        if df is None or len(df) == 0:
            continue
        
        features = []
        for col_name in FEATURE_COLUMNS:
            if col_name in df.columns:
                values = df[col_name].values
            else:
                # Try to find similar column
                found = False
                for col in df.columns:
                    if col_name.lower() in col.lower():
                        values = df[col].values
                        found = True
                        break
                if not found:
                    values = np.zeros(len(df))
            
            # Handle NaN
            values = np.nan_to_num(values, nan=0.0)
            
            # Normalize
            if normalize:
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    values = (values - mean) / std
                else:
                    values = values - mean
            
            features.append(values)
        
        # Stack features: shape (seq_len, n_features)
        x = np.stack(features, axis=-1)
        
        samples.append(x)
        labels.append(window.get(label_col, 0))
        subject_ids.append(window.get("subject_id", "unknown"))
    
    if not samples:
        return np.array([]), np.array([]), np.array([])
    
    # Stack all samples: shape (n_samples, seq_len, n_features)
    X = np.stack(samples, axis=0).astype(np.float32)
    y = np.array(labels)
    subject_ids = np.array(subject_ids)
    
    return X, y, subject_ids


def prepare_ts2vec_splits(windows_by_subject: Dict[str, List[Dict]],
                          test_subject: str,
                          label_col: str = "label_5min") -> Tuple:
    """
    Prepare train/test splits for LOSO.
    
    Args:
        windows_by_subject: Windows organized by subject
        test_subject: Subject to hold out
        label_col: Label column to use
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, test_subjects)
    """
    train_windows = []
    test_windows = []
    
    for subject_id, windows in windows_by_subject.items():
        for w in windows:
            w["subject_id"] = subject_id
        
        if subject_id == test_subject:
            test_windows.extend(windows)
        else:
            train_windows.extend(windows)
    
    X_train, y_train, _ = prepare_ts2vec_data(train_windows, label_col)
    X_test, y_test, test_subjects = prepare_ts2vec_data(test_windows, label_col)
    
    return X_train, y_train, X_test, y_test, test_subjects


if __name__ == "__main__":
    # Test dataset preparation
    import pandas as pd
    
    print("Testing TS2Vec dataset preparation...")
    
    np.random.seed(42)
    
    # Create dummy windows
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
    
    X, y, subjects = prepare_ts2vec_data(dummy_windows)
    
    print(f"X shape: {X.shape}")  # Expected: (10, 120, 4)
    print(f"y shape: {y.shape}")
    print(f"Features: {FEATURE_COLUMNS}")

