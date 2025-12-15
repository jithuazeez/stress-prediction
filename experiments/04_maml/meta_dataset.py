"""
Meta-learning dataset for MAML stress prediction.

Each subject is treated as a separate "task" for meta-learning.
Implements BALANCED episode sampling for meta-training.

Key features:
- Balanced k-shot sampling (critical for imbalanced stress data!)
- Uses extracted statistical features (not raw flattened signals)
- Compatible with learn2learn task distribution

References:
- https://arxiv.org/pdf/1703.03400 (MAML paper)
- https://github.com/learnables/learn2learn
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "01_classical_ml"))

from shared.config import Config

# Try to import feature extraction
try:
    from feature_extraction import BasicFeatureExtractor, FEATURE_NAMES
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False
    # Fallback feature names (basic stats)
    FEATURE_NAMES = [
        "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_min", "acc_magnitude_max",
        "acc_magnitude_range", "acc_magnitude_median", "acc_magnitude_iqr",
        "acc_x_std", "acc_y_std", "acc_z_std", "acc_sma", "acc_ima", "acc_energy", "acc_zcr",
        "acc_magnitude_skewness", "acc_magnitude_kurtosis",
        "acc_jerk_mean", "acc_jerk_std", "acc_jerk_max", "acc_jerk_energy",
        "motion_flag", "is_stationary", "is_high_activity",
        "temp_mean", "temp_std", "temp_min", "temp_max", "temp_range", "temp_slope", "temp_change",
        "heatflux_mean", "heatflux_std", "cbt_mean", "cbt_change",
        "eda_mean", "eda_median", "eda_std", "eda_min", "eda_max",
        "eda_range", "eda_num_peaks", "eda_trend", "eda_change",
    ]


class BalancedTaskSampler:
    """
    Balanced k-shot sampler for meta-learning.
    
    CRITICAL for stress detection:
    - Without balanced sampling, support sets can be all-negative
    - This causes adaptation to fail (no gradient for positive class)
    
    Strategy:
    - Sample k/2 from positive class, k/2 from negative class
    - If a class has fewer samples, sample all available + rest from other class
    """
    
    def __init__(self, k_shot: int = 5):
        """
        Args:
            k_shot: Total number of support samples per task
        """
        self.k_shot = k_shot
    
    def sample_balanced(self, 
                        y: np.ndarray, 
                        n_support: int = None,
                        n_query: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample balanced support and query indices.
        
        Args:
            y: Label array
            n_support: Number of support samples (default: self.k_shot)
            n_query: Number of query samples (default: remaining samples)
        
        Returns:
            Tuple of (support_indices, query_indices)
        """
        if n_support is None:
            n_support = self.k_shot
        
        n_samples = len(y)
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        # Target: balanced support set
        n_support_per_class = n_support // 2
        
        support_indices = []
        
        # Sample from positive class
        if n_pos >= n_support_per_class:
            sampled_pos = np.random.choice(pos_indices, n_support_per_class, replace=False)
        elif n_pos > 0:
            sampled_pos = pos_indices  # Take all available
        else:
            sampled_pos = np.array([], dtype=int)
        support_indices.extend(sampled_pos)
        
        # Sample from negative class (fill remaining)
        remaining = n_support - len(support_indices)
        if n_neg >= remaining:
            sampled_neg = np.random.choice(neg_indices, remaining, replace=False)
        elif n_neg > 0:
            sampled_neg = neg_indices  # Take all available
        else:
            sampled_neg = np.array([], dtype=int)
        support_indices.extend(sampled_neg)
        
        support_indices = np.array(support_indices, dtype=int)
        
        # Query set: everything not in support
        all_indices = np.arange(n_samples)
        query_indices = np.setdiff1d(all_indices, support_indices)
        
        # Limit query size if specified
        if n_query is not None and len(query_indices) > n_query:
            query_indices = np.random.choice(query_indices, n_query, replace=False)
        
        return support_indices, query_indices


class StressMetaDataset:
    """
    Meta-learning dataset where each subject is a task.
    
    For MAML, we need to sample:
    - Support set (k-shot): Small BALANCED set for adaptation
    - Query set: Remaining samples for evaluation
    
    This enables learning to quickly adapt to new subjects.
    
    CRITICAL: Uses BALANCED sampling to ensure positive samples in support set!
    """
    
    def __init__(self,
                 features_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 k_support: int = 10,
                 k_query: int = 20):
        """
        Initialize meta-dataset.
        
        Args:
            features_by_subject: Dict mapping subject_id to (X, y) tuple
                X: Feature array of shape (n_samples, n_features)
                y: Label array of shape (n_samples,)
            k_support: Number of support samples (will be balanced across classes)
            k_query: Number of query samples (or remaining if fewer available)
        """
        self.features_by_subject = features_by_subject
        self.k_support = k_support
        self.k_query = k_query
        
        self.subjects = list(features_by_subject.keys())
        self.n_subjects = len(self.subjects)
        
        # Create balanced sampler
        self.sampler = BalancedTaskSampler(k_shot=k_support)
        
        # Compute class distribution per subject
        self._compute_class_stats()
    
    def _compute_class_stats(self):
        """Compute class statistics for each subject."""
        self.class_stats = {}
        for subject_id, (X, y) in self.features_by_subject.items():
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            self.class_stats[subject_id] = {
                "n_samples": len(y),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "pos_ratio": n_pos / len(y) if len(y) > 0 else 0
            }
    
    def sample_task(self, subject_id: str) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """
        Sample a task (support and query sets) for a specific subject.
        
        Uses BALANCED sampling to ensure positive samples in support set!
        
        Args:
            subject_id: Subject ID to sample from
        
        Returns:
            Tuple of (support_x, support_y, query_x, query_y) as tensors
        """
        X, y = self.features_by_subject[subject_id]
        
        # Use balanced sampler
        support_indices, query_indices = self.sampler.sample_balanced(
            y, 
            n_support=self.k_support,
            n_query=self.k_query
        )
        
        # Convert to tensors
        support_x = torch.tensor(X[support_indices], dtype=torch.float32)
        support_y = torch.tensor(y[support_indices], dtype=torch.long)
        query_x = torch.tensor(X[query_indices], dtype=torch.float32)
        query_y = torch.tensor(y[query_indices], dtype=torch.long)
        
        return support_x, support_y, query_x, query_y
    
    def sample_tasks(self, n_tasks: int) -> List[Tuple[torch.Tensor, torch.Tensor,
                                                        torch.Tensor, torch.Tensor]]:
        """
        Sample multiple tasks for meta-training.
        
        Args:
            n_tasks: Number of tasks to sample
        
        Returns:
            List of (support_x, support_y, query_x, query_y) tuples
        """
        tasks = []
        
        # Sample subjects (with replacement if n_tasks > n_subjects)
        if n_tasks <= self.n_subjects:
            selected_subjects = np.random.choice(self.subjects, n_tasks, replace=False)
        else:
            selected_subjects = np.random.choice(self.subjects, n_tasks, replace=True)
        
        for subject_id in selected_subjects:
            task = self.sample_task(subject_id)
            tasks.append(task)
        
        return tasks
    
    def get_subject_data(self, subject_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all data for a subject (for evaluation).
        
        Args:
            subject_id: Subject ID
        
        Returns:
            Tuple of (X, y) tensors
        """
        X, y = self.features_by_subject[subject_id]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )


def extract_features_for_window(window_df, extractor=None) -> np.ndarray:
    """
    Extract features from a single window DataFrame.
    
    Args:
        window_df: DataFrame with aligned signal data
        extractor: BasicFeatureExtractor instance (optional)
    
    Returns:
        Feature array of shape (n_features,)
    """
    if extractor is None:
        if FEATURE_EXTRACTOR_AVAILABLE:
            extractor = BasicFeatureExtractor()
        else:
            raise ImportError("Feature extractor not available")
    
    features_dict = extractor.extract_from_window(window_df)
    
    # Convert to array in consistent order
    feature_values = [features_dict.get(name, np.nan) for name in FEATURE_NAMES]
    return np.array(feature_values, dtype=np.float32)


def prepare_features_by_subject(windows_by_subject: Dict[str, List[Dict]],
                                label_col: str = "label_5min",
                                normalize: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract features for each subject.
    
    Args:
        windows_by_subject: Dict mapping subject_id to list of window dicts
        label_col: Label column to use
        normalize: Whether to apply subject-wise normalization
    
    Returns:
        Dict mapping subject_id to (X, y) tuple where:
            X: Feature array of shape (n_windows, n_features)
            y: Label array of shape (n_windows,)
    """
    if FEATURE_EXTRACTOR_AVAILABLE:
        extractor = BasicFeatureExtractor()
    else:
        extractor = None
        print("Warning: Using fallback feature extraction")
    
    features_by_subject = {}
    
    for subject_id, windows in windows_by_subject.items():
        X_list = []
        y_list = []
        
        for window in windows:
            window_df = window.get("window_data")
            if window_df is None:
                continue
            
            # Extract features
            if extractor is not None:
                features_dict = extractor.extract_from_window(window_df)
                feature_values = [features_dict.get(name, np.nan) for name in FEATURE_NAMES]
            else:
                # Fallback: basic statistics from raw data
                feature_values = _extract_basic_features(window_df)
            
            X_list.append(feature_values)
            y_list.append(window.get(label_col, 0))
        
        if X_list:
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int64)
            
            # Handle NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Subject-wise normalization (z-score)
            if normalize:
                mean = np.mean(X, axis=0, keepdims=True)
                std = np.std(X, axis=0, keepdims=True) + 1e-8
                X = (X - mean) / std
            
            features_by_subject[subject_id] = (X, y)
    
    return features_by_subject


def _extract_basic_features(window_df) -> List[float]:
    """
    Fallback basic feature extraction when main extractor unavailable.
    
    Args:
        window_df: Window DataFrame
    
    Returns:
        List of feature values
    """
    features = []
    
    # Try to get accelerometer magnitude
    if "acc_magnitude" in window_df.columns:
        acc = window_df["acc_magnitude"].dropna().values
        if len(acc) > 0:
            features.extend([
                np.mean(acc), np.std(acc), np.min(acc), np.max(acc),
                np.ptp(acc), np.median(acc), np.percentile(acc, 75) - np.percentile(acc, 25)
            ])
        else:
            features.extend([0.0] * 7)
    else:
        features.extend([0.0] * 7)
    
    # Pad to expected length
    while len(features) < len(FEATURE_NAMES):
        features.append(0.0)
    
    return features[:len(FEATURE_NAMES)]


def get_feature_dim() -> int:
    """Get the number of features."""
    return len(FEATURE_NAMES)


if __name__ == "__main__":
    print("Testing BALANCED Meta-Dataset for Stress Prediction")
    print("=" * 60)
    
    # Create dummy data with imbalanced classes (typical for stress)
    np.random.seed(42)
    
    features_by_subject = {}
    for i in range(5):
        subject_id = f"subject_{i}"
        n_samples = np.random.randint(30, 60)
        n_features = len(FEATURE_NAMES)
        
        # Create imbalanced data: ~15% positive (stress)
        n_positive = max(3, int(n_samples * 0.15))
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.zeros(n_samples, dtype=np.int64)
        y[:n_positive] = 1
        np.random.shuffle(y)  # Shuffle labels
        
        features_by_subject[subject_id] = (X, y)
        print(f"  {subject_id}: {n_samples} samples, {n_positive} positive ({100*n_positive/n_samples:.1f}%)")
    
    # Create meta-dataset
    print("\nCreating StressMetaDataset with k_support=10, k_query=20...")
    dataset = StressMetaDataset(features_by_subject, k_support=10, k_query=20)
    
    print(f"  Number of subjects/tasks: {dataset.n_subjects}")
    print(f"  Feature dimension: {len(FEATURE_NAMES)}")
    
    # Test balanced sampling
    print("\nTesting BALANCED task sampling:")
    for subject_id in list(features_by_subject.keys())[:3]:
        support_x, support_y, query_x, query_y = dataset.sample_task(subject_id)
        
        support_pos = (support_y == 1).sum().item()
        support_neg = (support_y == 0).sum().item()
        query_pos = (query_y == 1).sum().item()
        query_neg = (query_y == 0).sum().item()
        
        print(f"\n  {subject_id}:")
        print(f"    Support: {len(support_y)} samples ({support_pos} pos, {support_neg} neg)")
        print(f"    Query:   {len(query_y)} samples ({query_pos} pos, {query_neg} neg)")
        print(f"    Shapes: support_x={support_x.shape}, query_x={query_x.shape}")
    
    # Verify balanced sampling works
    print("\n" + "=" * 60)
    print("Verifying balanced sampling over 100 trials...")
    
    all_support_ratios = []
    for _ in range(100):
        for subject_id in features_by_subject.keys():
            _, support_y, _, _ = dataset.sample_task(subject_id)
            ratio = (support_y == 1).sum().item() / len(support_y)
            all_support_ratios.append(ratio)
    
    mean_ratio = np.mean(all_support_ratios)
    std_ratio = np.std(all_support_ratios)
    print(f"  Mean positive ratio in support: {mean_ratio:.3f} ± {std_ratio:.3f}")
    print(f"  (Target: ~0.5 for balanced, original data: ~0.15)")
    
    if mean_ratio > 0.3:
        print("\n✅ Balanced sampling is working! Support sets are class-balanced.")
    else:
        print("\n❌ Warning: Support sets are still imbalanced!")
