"""
Meta-learning dataset for MAML.

Each subject is treated as a separate "task" for meta-learning.
Implements episode sampling for meta-training.

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
    FEATURE_NAMES = []


class StressMetaDataset:
    """
    Meta-learning dataset where each subject is a task.
    
    For MAML, we need to sample:
    - Support set (k-shot): Small set for adaptation
    - Query set: Remaining samples for evaluation
    
    This enables learning to quickly adapt to new subjects.
    """
    
    def __init__(self,
                 features_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 k_support: int = 5,
                 k_query: int = 15):
        """
        Initialize meta-dataset.
        
        Args:
            features_by_subject: Dict mapping subject_id to (X, y) tuple
            k_support: Number of support samples per class
            k_query: Number of query samples (or remaining samples if less)
        """
        self.features_by_subject = features_by_subject
        self.k_support = k_support
        self.k_query = k_query
        
        self.subjects = list(features_by_subject.keys())
        self.n_subjects = len(self.subjects)
    
    def sample_task(self, subject_id: str) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """
        Sample a task (support and query sets) for a specific subject.
        
        Args:
            subject_id: Subject ID to sample from
        
        Returns:
            Tuple of (support_x, support_y, query_x, query_y) as tensors
        """
        X, y = self.features_by_subject[subject_id]
        
        n_samples = len(X)
        
        # Get indices for each class
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        # Sample support set (balanced if possible)
        support_indices = []
        
        # Sample k_support from each class if available
        n_support_per_class = self.k_support // 2
        
        if len(pos_indices) >= n_support_per_class:
            support_pos = np.random.choice(pos_indices, n_support_per_class, replace=False)
            support_indices.extend(support_pos)
        elif len(pos_indices) > 0:
            support_indices.extend(pos_indices)
        
        if len(neg_indices) >= n_support_per_class:
            support_neg = np.random.choice(neg_indices, n_support_per_class, replace=False)
            support_indices.extend(support_neg)
        elif len(neg_indices) > 0:
            remaining = self.k_support - len(support_indices)
            n_to_sample = min(remaining, len(neg_indices))
            support_neg = np.random.choice(neg_indices, n_to_sample, replace=False)
            support_indices.extend(support_neg)
        
        support_indices = np.array(support_indices)
        
        # Query set is everything else
        all_indices = np.arange(n_samples)
        query_indices = np.setdiff1d(all_indices, support_indices)
        
        # Limit query size
        if len(query_indices) > self.k_query:
            query_indices = np.random.choice(query_indices, self.k_query, replace=False)
        
        # Extract data
        support_x = torch.tensor(X[support_indices], dtype=torch.float32)
        support_y = torch.tensor(y[support_indices], dtype=torch.long)
        query_x = torch.tensor(X[query_indices], dtype=torch.float32)
        query_y = torch.tensor(y[query_indices], dtype=torch.long)
        
        return support_x, support_y, query_x, query_y
    
    def sample_tasks(self, n_tasks: int) -> List[Tuple]:
        """
        Sample multiple tasks for meta-training.
        
        Args:
            n_tasks: Number of tasks to sample
        
        Returns:
            List of (support_x, support_y, query_x, query_y) tuples
        """
        tasks = []
        
        # Sample subjects (with replacement if needed)
        if n_tasks <= self.n_subjects:
            selected_subjects = np.random.choice(self.subjects, n_tasks, replace=False)
        else:
            selected_subjects = np.random.choice(self.subjects, n_tasks, replace=True)
        
        for subject_id in selected_subjects:
            task = self.sample_task(subject_id)
            tasks.append(task)
        
        return tasks


def prepare_features_by_subject(windows_by_subject: Dict[str, List[Dict]],
                                label_col: str = "label_5min") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract features for each subject.
    
    Args:
        windows_by_subject: Windows organized by subject
        label_col: Label column to use
    
    Returns:
        Dict mapping subject_id to (X, y) tuple
    """
    if not FEATURE_EXTRACTOR_AVAILABLE:
        raise ImportError("Feature extractor not available")
    
    extractor = BasicFeatureExtractor()
    features_by_subject = {}
    
    for subject_id, windows in windows_by_subject.items():
        X_list = []
        y_list = []
        
        for window in windows:
            features = extractor.extract_from_window(window["window_data"])
            feature_values = [features.get(name, np.nan) for name in FEATURE_NAMES]
            
            X_list.append(feature_values)
            y_list.append(window.get(label_col, 0))
        
        if X_list:
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int64)
            
            # Handle NaN
            X = np.nan_to_num(X, nan=0.0)
            
            features_by_subject[subject_id] = (X, y)
    
    return features_by_subject


if __name__ == "__main__":
    print("Testing meta-dataset...")
    
    # Create dummy data
    np.random.seed(42)
    
    features_by_subject = {}
    for i in range(5):
        subject_id = f"subject_{i}"
        n_samples = np.random.randint(20, 50)
        X = np.random.randn(n_samples, 38).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.int64)
        features_by_subject[subject_id] = (X, y)
    
    dataset = StressMetaDataset(features_by_subject, k_support=5, k_query=15)
    
    print(f"Number of subjects/tasks: {dataset.n_subjects}")
    
    # Sample a task
    support_x, support_y, query_x, query_y = dataset.sample_task("subject_0")
    print(f"\nSampled task:")
    print(f"  Support: X={support_x.shape}, y={support_y.shape}")
    print(f"  Query: X={query_x.shape}, y={query_y.shape}")
    print(f"  Support labels: {support_y.tolist()}")

