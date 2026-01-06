"""
Time alignment utilities for multimodal signal data.

Aligns all modalities to a common 1Hz time grid using scipy for resampling.

UPDATED: Now uses HR derived from PPG (via HeartPy) instead of raw PPG.
EDA is commented out due to very low sampling rate (~0.017 Hz).
"""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")

# Path to pre-extracted HR data (from HeartPy)
# Try multiple possible locations
HR_DATA_PATHS = [
    # Local development
    Path("/Users/jithuazeez/Documents/Msc/Dissertation/reports/hr_1hz_cleaned.csv"),
    Path("/Users/jithuazeez/Documents/Msc/Dissertation/reports/hr_1hz_from_ppg.csv"),
   
    # Relative paths
    Path("reports/hr_1hz_cleaned.csv"),
    Path("../reports/hr_1hz_cleaned.csv"),
     # Kaggle paths (if you upload the HR data)
    # Path("/kaggle/input/hr-data-ppg/hr_1hz_from_ppg.csv"),
    Path("/kaggle/input/hr-1hz-ppg/hr_1hz_from_ppg.csv"),
    Path("/kaggle/input/hr-data-ppg/hr_1hz_from_ppg.csv")
    
]

# Cache for HR data to avoid reloading
_HR_DATA_CACHE = None
_HR_DATA_PATH_FOUND = None


def find_hr_data_path() -> Optional[Path]:
    """Find the HR data file from possible locations."""
    global _HR_DATA_PATH_FOUND
    
    if _HR_DATA_PATH_FOUND is not None:
        return _HR_DATA_PATH_FOUND
    
    for path in HR_DATA_PATHS:
        if path.exists():
            _HR_DATA_PATH_FOUND = path
            print(f"Found HR data at: {path}")
            return path
    
    return None


def load_hr_data() -> pd.DataFrame:
    """
    Load pre-extracted HR data from HeartPy processing.
    
    This HR was extracted from raw PPG using HeartPy and interpolated to 1Hz.
    Much more meaningful than downsampling raw PPG from 64Hz to 1Hz.
    
    Returns:
        DataFrame with columns: timestamp, subject_id, hr_bpm, rmssd, hr_jump, artifact_flag
    """
    global _HR_DATA_CACHE
    
    if _HR_DATA_CACHE is not None:
        return _HR_DATA_CACHE
    
    hr_path = find_hr_data_path()
    
    if hr_path is None:
        print("="*60)
        print("WARNING: HR data file not found!")
        print("Searched paths:")
        for p in HR_DATA_PATHS:
            print(f"  - {p}")
        print("\nTo fix this:")
        print("1. Run notebooks/data_quality_analysis.ipynb to generate hr_1hz_cleaned.csv")
        print("2. For Kaggle: Upload hr_1hz_cleaned.csv as a dataset")
        print("="*60)
        return pd.DataFrame()
    
    hr_df = pd.read_csv(hr_path)
    hr_df["timestamp"] = pd.to_datetime(hr_df["timestamp"])
    
    _HR_DATA_CACHE = hr_df
    return hr_df


def get_hr_for_subject(subject_id: str) -> Optional[pd.DataFrame]:
    """
    Get HR data for a specific subject.
    
    Args:
        subject_id: Subject ID (without 'id_' prefix)
    
    Returns:
        DataFrame with timestamp and hr_bpm columns, or None if not found
    """
    hr_df = load_hr_data()
    
    if hr_df is None or len(hr_df) == 0:
        return None
    
    # Filter for this subject
    subject_hr = hr_df[hr_df["subject_id"] == subject_id].copy()
    
    if len(subject_hr) == 0:
        return None
    
    return subject_hr


def create_time_grid(start_time: pd.Timestamp, 
                     end_time: pd.Timestamp, 
                     freq: str = "1S") -> pd.DatetimeIndex:
    """
    Create a regular time grid.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        freq: Frequency string (default "1S" for 1 Hz)
    
    Returns:
        DatetimeIndex representing the time grid (preserves timezone)
    """
    # pd.date_range preserves timezone from start_time
    time_grid = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Ensure timezone consistency - if start_time has tz, the grid should too
    # If grid is naive but start_time has tz, localize it
    if start_time.tz is not None and time_grid.tz is None:
        time_grid = time_grid.tz_localize(start_time.tz)
    
    return time_grid


def resample_signal_scipy(timestamps: np.ndarray, 
                          values: np.ndarray,
                          target_timestamps: np.ndarray,
                          method: str = "linear") -> np.ndarray:
    """
    Resample a signal to target timestamps using scipy interpolation.
    
    Args:
        timestamps: Original timestamps as numpy array (numeric, e.g., epoch seconds)
        values: Original signal values
        target_timestamps: Target timestamps to interpolate to
        method: Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        Resampled signal values at target timestamps
    """
    if len(timestamps) < 2:
        return np.full(len(target_timestamps), np.nan)
    
    # Remove NaN values for interpolation
    valid_mask = ~np.isnan(values)
    if np.sum(valid_mask) < 2:
        return np.full(len(target_timestamps), np.nan)
    
    valid_ts = timestamps[valid_mask]
    valid_vals = values[valid_mask]
    
    # Create interpolation function
    try:
        f = interp1d(
            valid_ts, 
            valid_vals, 
            kind="linear", 
            bounds_error=False, 
            fill_value=np.nan
        )
        resampled = f(target_timestamps)
    except Exception:
        resampled = np.full(len(target_timestamps), np.nan)
    
    return resampled


def downsample_mean(df: pd.DataFrame, 
                    value_col: str,
                    time_grid: pd.DatetimeIndex,
                    resample_period: str = "1S",
                    tolerance_sec: float = 1.0) -> np.ndarray:
    """
    Downsample high-frequency signal using mean aggregation.
    
    Takes the mean of all samples that fall within each time period.
    
    Args:
        df: DataFrame with 'timestamp' and value column
        value_col: Name of the value column
        time_grid: Target time grid
        resample_period: Pandas frequency string (e.g., "1S" for 1Hz, "250ms" for 4Hz)
        tolerance_sec: Tolerance in seconds for timestamp matching
    
    Returns:
        Downsampled values aligned to time grid
    """
    # if df is None or len(df) == 0:
    #     return np.full(len(time_grid), np.nan)
    
    # # Set timestamp as index
    # df = df.copy()
    # df = df.set_index("timestamp")
    
    # # Resample using specified period, taking mean
    # try:
    #     resampled = df[value_col].resample(resample_period).mean()
        
    #     # Align to our time grid
    #     result = np.full(len(time_grid), np.nan)
    #     for i, t in enumerate(time_grid):
    #         if t in resampled.index:
    #             result[i] = resampled.loc[t]
    #         else:
    #             # Find nearest timestamp within tolerance
    #             time_diff = np.abs((resampled.index - t).total_seconds())
    #             if len(time_diff) > 0 and time_diff.min() < tolerance_sec:
    #                 nearest_idx = time_diff.argmin()
    #                 result[i] = resampled.iloc[nearest_idx]
        
    #     return result
    # except Exception as e:
    #     print(f"Warning: Downsample failed for {value_col}: {e}")
    #     return np.full(len(time_grid), np.nan)
    if df is None or len(df) == 0:
        return np.full(len(time_grid), np.nan)
    
    if value_col not in df.columns:
        return np.full(len(time_grid), np.nan)

    df = df.copy()
    df = df.set_index("timestamp")

    resampled = df[value_col].resample(resample_period).mean()

    # Reindex to the desired grid with nearest neighbour + tolerance
    aligned = resampled.reindex(
        time_grid, 
        method="nearest", 
        tolerance=pd.Timedelta(seconds=tolerance_sec)
    )

    return aligned.to_numpy()


def forward_fill_signal(df: pd.DataFrame,
                        value_col: str,
                        time_grid: pd.DatetimeIndex) -> np.ndarray:
    """
    Forward-fill a low-frequency signal to 1Hz.
    
    For signals with very low sampling rate (like EDA at ~1/min),
    use the last known value until a new sample arrives.
    
    Args:
        df: DataFrame with 'timestamp' and value column
        value_col: Name of the value column
        time_grid: Target 1Hz time grid
    
    Returns:
        Forward-filled values aligned to time grid
    """
    if df is None or len(df) == 0:
        return np.full(len(time_grid), np.nan)
    
    result = np.full(len(time_grid), np.nan)
    
    # Sort by timestamp
    df = df.sort_values("timestamp").copy()
    
    # Convert timestamps to pandas Series for proper comparison
    df_timestamps = pd.Series(df["timestamp"].values)
    
    # Ensure timezone consistency
    if hasattr(df_timestamps.iloc[0], 'tz') or (hasattr(df_timestamps.iloc[0], 'tzinfo') and df_timestamps.iloc[0].tzinfo is not None):
        # df timestamps are tz-aware, ensure grid is too
        if time_grid.tz is None:
            # Get timezone from data
            sample_ts = pd.Timestamp(df_timestamps.iloc[0])
            if sample_ts.tz is not None:
                time_grid = time_grid.tz_localize(sample_ts.tz)
    
    values = df[value_col].values
    
    # Convert to numpy datetime64 for comparison (removes tz issues)
    df_ts_np = pd.to_datetime(df_timestamps).values.astype('datetime64[ns]')
    grid_ts_np = pd.to_datetime(time_grid).values.astype('datetime64[ns]')
    
    # For each grid point, find the most recent sample
    for i, t in enumerate(grid_ts_np):
        # Find samples at or before this time
        mask = df_ts_np <= t
        if np.any(mask):
            # Use the most recent value
            last_idx = np.where(mask)[0][-1]
            result[i] = values[last_idx]
    
    return result


def align_signals(signals: Dict[str, Optional[pd.DataFrame]],
                  start_time: pd.Timestamp,
                  end_time: pd.Timestamp,
                  target_hz: float = 4.0) -> pd.DataFrame:
    """
    Align all modalities to a common time grid at specified sampling rate.
    
    Strategy:
    - heatflux (1Hz): upsample or match to target rate using interpolation
    - acc (~32Hz): downsample to target rate using mean aggregation
    - hr: interpolate from pre-extracted 1Hz HR to target rate
    
    Args:
        signals: Dictionary of loaded signal DataFrames
        start_time: Start of alignment window
        end_time: End of alignment window
        target_hz: Target sampling rate in Hz (default: 4.0 Hz)
    
    Returns:
        DataFrame with columns: timestamp, acc_x, acc_y, acc_z, acc_magnitude,
                               skin_temp, heatflux, cbt, pulse_rate, hr_bpm, rmssd
    """
    # Calculate frequency string for pandas (e.g., "250ms" for 4 Hz)
    period_ms = int(1000 / target_hz)
    freq_str = f"{period_ms}ms"
    
    # Create time grid at target frequency
    time_grid = create_time_grid(start_time, end_time, freq=freq_str)
    
    aligned = pd.DataFrame({"timestamp": time_grid})
    
    # Tolerance for timestamp matching (half the sampling period)
    tolerance_sec = 1.0 / (2.0 * target_hz)
    
    # --- Align heatflux data (1Hz -> target_hz) ---
    hf_df = signals.get("heatflux")
    if hf_df is not None and len(hf_df) > 0:
        if target_hz <= 1.0:
            # Downsample or match
            for col in ["skin_temp", "heatflux", "cbt", "pulse_rate"]:
                if col in hf_df.columns:
                    aligned[col] = downsample_mean(hf_df, col, time_grid, 
                                                   resample_period=freq_str,
                                                   tolerance_sec=tolerance_sec)
        else:
            # Upsample using interpolation for higher frequencies
            for col in ["skin_temp", "heatflux", "cbt", "pulse_rate"]:
                if col in hf_df.columns:
                    hf_ts = hf_df["timestamp"].values.astype('datetime64[ns]').astype(float)
                    grid_ts = time_grid.values.astype('datetime64[ns]').astype(float)
                    aligned[col] = resample_signal_scipy(hf_ts, hf_df[col].values, 
                                                        grid_ts, method="linear")
    
    # --- Align accelerometer (~32Hz -> target_hz) ---
    acc_df = signals.get("acc")
    if acc_df is not None and len(acc_df) > 0:
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in acc_df.columns:
                aligned[col] = downsample_mean(acc_df, col, time_grid,
                                              resample_period=freq_str,
                                              tolerance_sec=tolerance_sec)
        
        # Calculate magnitude
        if all(c in aligned.columns for c in ["acc_x", "acc_y", "acc_z"]):
            aligned["acc_magnitude"] = np.sqrt(
                aligned["acc_x"]**2 + 
                aligned["acc_y"]**2 + 
                aligned["acc_z"]**2
            )
    
    # --- Align HR from HeartPy (1Hz -> target_hz) ---
    subject_id = signals.get("subject_id")
    if subject_id:
        hr_df = get_hr_for_subject(subject_id)
        if hr_df is not None and len(hr_df) > 0:
            # Interpolate HR data to target frequency
            hr_ts = hr_df["timestamp"].values.astype('datetime64[ns]').astype(float)
            grid_ts = time_grid.values.astype('datetime64[ns]').astype(float)
            
            aligned["hr_bpm"] = resample_signal_scipy(hr_ts, hr_df["hr_bpm"].values,
                                                     grid_ts, method="linear")
            
            if "rmssd" in hr_df.columns:
                aligned["rmssd"] = resample_signal_scipy(hr_ts, hr_df["rmssd"].values,
                                                        grid_ts, method="linear")
    
    return aligned


# COMMENTED OUT: Original 1Hz alignment function (keep for backward compatibility)
# def align_to_1hz(signals: Dict[str, Optional[pd.DataFrame]],
#                  start_time: pd.Timestamp,
#                  end_time: pd.Timestamp) -> pd.DataFrame:
#     """
#     Align all modalities to a 1Hz common time grid.
#     
#     Strategy:
#     - heatflux: already 1Hz, use directly
#     - acc (~32Hz): downsample to 1Hz using mean per second
#     - hr: use pre-extracted 1Hz HR from HeartPy (NOT raw PPG!)
#     - emography: DISABLED (too low sampling rate ~0.017Hz)
#     
#     Args:
#         signals: Dictionary of loaded signal DataFrames
#         start_time: Start of alignment window
#         end_time: End of alignment window
#     
#     Returns:
#         DataFrame with columns: timestamp, acc_x, acc_y, acc_z, acc_magnitude,
#                                skin_temp, heatflux, cbt, pulse_rate, hr_bpm
#     """
#     # Create 1Hz time grid
#     time_grid = create_time_grid(start_time, end_time, freq="1S")
#     
#     aligned = pd.DataFrame({"timestamp": time_grid})
#     
#     # --- Align heatflux data (already 1Hz) ---
#     hf_df = signals.get("heatflux")
#     if hf_df is not None and len(hf_df) > 0:
#         # These columns are at 1Hz, just need to match timestamps
#         for col in ["skin_temp", "heatflux", "cbt", "pulse_rate"]:
#             if col in hf_df.columns:
#                 aligned[col] = downsample_mean(hf_df, col, time_grid)
#     
#     # --- Align accelerometer (~32Hz -> 1Hz) ---
#     acc_df = signals.get("acc")
#     if acc_df is not None and len(acc_df) > 0:
#         for col in ["acc_x", "acc_y", "acc_z"]:
#             if col in acc_df.columns:
#                 aligned[col] = downsample_mean(acc_df, col, time_grid)
#         
#         # Calculate magnitude
#         if all(c in aligned.columns for c in ["acc_x", "acc_y", "acc_z"]):
#             aligned["acc_magnitude"] = np.sqrt(
#                 aligned["acc_x"]**2 + 
#                 aligned["acc_y"]**2 + 
#                 aligned["acc_z"]**2
#             )
#     
#     # --- Align HR from HeartPy (already at 1Hz) ---
#     # This is much better than downsampling raw PPG from 64Hz to 1Hz!
#     subject_id = signals.get("subject_id")
#     if subject_id:
#         hr_df = get_hr_for_subject(subject_id)
#         if hr_df is not None and len(hr_df) > 0:
#             # Merge HR data with time grid
#             hr_values = np.full(len(time_grid), np.nan)
#             rmssd_values = np.full(len(time_grid), np.nan)
#             
#             # Convert timestamps for matching
#             hr_df_ts = hr_df["timestamp"].values.astype('datetime64[ns]')
#             grid_ts = time_grid.values.astype('datetime64[ns]')
#             
#             for i, t in enumerate(grid_ts):
#                 # Find matching timestamp (within 1 second)
#                 time_diff = np.abs((hr_df_ts - t).astype('timedelta64[s]').astype(float))
#                 if len(time_diff) > 0:
#                     min_diff_idx = np.argmin(time_diff)
#                     if time_diff[min_diff_idx] <= 1.0:
#                         hr_values[i] = hr_df["hr_bpm"].iloc[min_diff_idx]
#                         if "rmssd" in hr_df.columns:
#                             rmssd_values[i] = hr_df["rmssd"].iloc[min_diff_idx]
#             
#             aligned["hr_bpm"] = hr_values
#             aligned["rmssd"] = rmssd_values
#     
#     return aligned


if __name__ == "__main__":
    # Test alignment
    from raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
    from config import DEFAULT_CONFIG
    
    print("Testing signal alignment...")
    
    # Check if HR data is available
    hr_path = find_hr_data_path()
    print(f"HR data path found: {hr_path}")
    
    hr_df = load_hr_data()
    if len(hr_df) > 0:
        print(f"HR data loaded: {len(hr_df)} samples, {hr_df['subject_id'].nunique()} subjects")
    else:
        print("Warning: No HR data loaded. Run HeartPy extraction first!")
    
    subjects = get_all_subjects(DEFAULT_CONFIG.data_path)
    
    if subjects:
        signals = load_raw_signals(subjects[0])
        start, end = get_experiment_time_range(signals)
        
        # Test 4 Hz alignment
        print(f"\nAligning signals from {start} to {end} at 4 Hz")
        aligned = align_signals(signals, start, end, target_hz=4.0)
        
        print(f"\nAligned DataFrame shape: {aligned.shape}")
        print(f"Columns: {list(aligned.columns)}")
        print(f"Expected samples for 120s window: ~480 samples")
        print(f"\nSample of aligned data:")
        print(aligned.head(10))
        
        # Check missing values
        print(f"\nMissing value counts:")
        for col in aligned.columns:
            if col != "timestamp":
                missing = aligned[col].isna().sum()
                pct = 100 * missing / len(aligned)
                print(f"  {col}: {missing} ({pct:.1f}%)")
        
        # Check HR specifically
        if "hr_bpm" in aligned.columns:
            hr_valid = (~aligned["hr_bpm"].isna()).sum()
            print(f"\nHR data: {hr_valid}/{len(aligned)} valid samples ({100*hr_valid/len(aligned):.1f}%)")
        else:
            print("\nWarning: hr_bpm column not in aligned data!")

