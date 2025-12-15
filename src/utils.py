"""
Utility functions for VitaStress stress prediction project.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_directory(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_subject_ids(data_path: str) -> List[str]:
    """
    Get list of subject IDs from VitaStress dataset.
    
    Args:
        data_path: Path to VitaStress data directory
        
    Returns:
        Sorted list of subject IDs
    """
    data_dir = Path(data_path)
    subjects = [d.name for d in data_dir.iterdir() 
                if d.is_dir() and d.name.startswith('id_')]
    return sorted(subjects)


def time_to_seconds(time_str: str) -> float:
    """
    Convert time string to seconds.
    
    Args:
        time_str: Time string (e.g., "00:05:30")
        
    Returns:
        Time in seconds
    """
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    else:
        return float(parts[0])


def get_logger(name: str, log_file: Optional[str] = None):
    """
    Create logger for consistent logging.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    import logging
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger








