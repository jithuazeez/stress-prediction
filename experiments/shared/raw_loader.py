"""
Raw signal loader for VitaStress dataset.

Loads the 4 raw signal modalities + annotations:
- acc.csv: 3-axis accelerometer (~32 Hz)
- emography.csv: EDA/skin conductance (~0.017 Hz)
- heat_flux_sensor_temperature.csv: temperature, heatflux, etc. (1 Hz)
- ppg2_green_6.csv: raw PPG signal (~64 Hz)
- annotation.csv: stress event timestamps

Note: Signal files use 'date' column, annotation uses 'timestamp' column.
      All are normalized to 'timestamp' for consistency.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore")

# Module logger
logger = logging.getLogger(__name__)


def get_all_subjects(data_path: Path) -> List[Path]:
    """
    Get list of all subject folders in the data directory.
    
    Args:
        data_path: Path to VitaStress data directory
    
    Returns:
        List of subject folder paths sorted by name
    """
    subject_folders = sorted([
        f for f in data_path.iterdir() 
        if f.is_dir() and f.name.startswith("id_")
    ])
    return subject_folders


def load_raw_signals(subject_folder: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load all 4 raw signal files + annotations for a subject.
    
    Args:
        subject_folder: Path to subject folder (e.g., id_0a73ef1b-...)
    
    Returns:
        Dictionary with keys: 'acc', 'emography', 'heatflux', 'ppg', 'annotation'
        Values are DataFrames or None if file doesn't exist.
        All DataFrames have 'timestamp' column (normalized from 'date' where applicable).
    """
    # Extract subject ID from folder name
    subject_id = subject_folder.name.replace("id_", "")
    
    signals = {
        "acc": None,
        "emography": None,
        "heatflux": None,
        "ppg": None,
        "annotation": None,
        "subject_id": subject_id
    }
    
    # Load accelerometer (~32 Hz)
    # CSV has columns: date, metric_id, chunk_index, quality, body_pose, accX, accY, accZ
    acc_file = subject_folder / f"{subject_id}_acc.csv"
    if acc_file.exists():
        try:
            df = pd.read_csv(acc_file)
            # Rename 'date' to 'timestamp' for consistency
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            # Rename accelerometer columns for consistency
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == "accx" or "raw_x" in col_lower or col_lower == "x":
                    rename_map[col] = "acc_x"
                elif col_lower == "accy" or "raw_y" in col_lower or col_lower == "y":
                    rename_map[col] = "acc_y"
                elif col_lower == "accz" or "raw_z" in col_lower or col_lower == "z":
                    rename_map[col] = "acc_z"
            df = df.rename(columns=rename_map)
            signals["acc"] = df
        except Exception as e:
            print(f"Warning: Failed to load acc for {subject_id}: {e}")
    
    # Load emography/EDA (~0.017 Hz)
    # CSV has columns: date, cz, pcz, pczt, czh, cc, quality, stress_skin, stress_skin_quality
    eda_file = subject_folder / f"{subject_id}_emography.csv"
    if eda_file.exists():
        try:
            df = pd.read_csv(eda_file)
            # Rename 'date' to 'timestamp' for consistency
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            signals["emography"] = df
        except Exception as e:
            print(f"Warning: Failed to load emography for {subject_id}: {e}")
    
    # Load heat flux + temperature (1 Hz)
    # CSV has columns: date, skin_temp, heatflux, acc_x, acc_y, acc_z, pulse_rate, cbt
    hf_file = subject_folder / f"{subject_id}_heat_flux_sensor_temperature.csv"
    if hf_file.exists():
        try:
            df = pd.read_csv(hf_file)
            # Rename 'date' to 'timestamp' for consistency
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            signals["heatflux"] = df
        except Exception as e:
            print(f"Warning: Failed to load heatflux for {subject_id}: {e}")
    
    # Load PPG (~64 Hz)
    # CSV has columns: date, metric_id, chunk_index, quality, body_pose, led_pd_pos, offset, exp, led, gain, value
    ppg_file = subject_folder / f"{subject_id}_ppg2_green_6.csv"
    if ppg_file.exists():
        try:
            df = pd.read_csv(ppg_file)
            # Rename 'date' to 'timestamp' for consistency
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            # Handle zeros as missing
            if "value" in df.columns:
                df.loc[df["value"] == 0, "value"] = np.nan
            signals["ppg"] = df
        except Exception as e:
            print(f"Warning: Failed to load ppg for {subject_id}: {e}")
    
    # Load annotations
    # CSV has columns: timestamp, Button Name
    ann_file = subject_folder / f"{subject_id}_annotation.csv"
    if ann_file.exists():
        try:
            df = pd.read_csv(ann_file)
            # Annotation file already uses 'timestamp'
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            signals["annotation"] = df
        except Exception as e:
            print(f"Warning: Failed to load annotation for {subject_id}: {e}")
    
    return signals


def get_experiment_time_range(signals: Dict[str, Optional[pd.DataFrame]]) -> tuple:
    """
    Determine the experiment time range from loaded signals.
    
    Uses heatflux (1Hz reference) if available, otherwise finds overlap.
    
    Args:
        signals: Dictionary of loaded signals
    
    Returns:
        Tuple of (start_time, end_time) as pd.Timestamp
    """
    start_times = []
    end_times = []
    
    for key in ["heatflux", "acc", "ppg", "emography"]:
        df = signals.get(key)
        if df is not None and len(df) > 0 and "timestamp" in df.columns:
            start_times.append(df["timestamp"].min())
            end_times.append(df["timestamp"].max())
    
    if not start_times or not end_times:
        raise ValueError("No valid signals found to determine time range")
    
    # Use the latest start and earliest end (overlap region)
    start_time = max(start_times)
    end_time = min(end_times)
    
    return start_time, end_time


if __name__ == "__main__":
    # Test loading
    from config import DEFAULT_CONFIG
    
    print("Testing raw signal loader...")
    subjects = get_all_subjects(DEFAULT_CONFIG.data_path)
    print(f"Found {len(subjects)} subjects")
    
    if subjects:
        # Load first subject
        signals = load_raw_signals(subjects[0])
        print(f"\nSubject: {signals['subject_id']}")
        
        for key, df in signals.items():
            if key == "subject_id":
                continue
            if df is not None:
                print(f"  {key}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
            else:
                print(f"  {key}: Not found")
        
        # Get time range
        start, end = get_experiment_time_range(signals)
        duration = (end - start).total_seconds() / 60
        print(f"\nExperiment duration: {duration:.1f} minutes")
