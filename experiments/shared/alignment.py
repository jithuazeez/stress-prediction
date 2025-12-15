"""
Time alignment utilities for multimodal signal data.

Aligns all modalities to a common 1Hz time grid using scipy for resampling.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")


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
            kind=method, 
            bounds_error=False, 
            fill_value=np.nan
        )
        resampled = f(target_timestamps)
    except Exception:
        resampled = np.full(len(target_timestamps), np.nan)
    
    return resampled


def downsample_mean(df: pd.DataFrame, 
                    value_col: str,
                    time_grid: pd.DatetimeIndex) -> np.ndarray:
    """
    Downsample high-frequency signal to 1Hz using mean aggregation.
    
    For each second in the time grid, takes the mean of all samples
    that fall within that second.
    
    Args:
        df: DataFrame with 'timestamp' and value column
        value_col: Name of the value column
        time_grid: Target 1Hz time grid
    
    Returns:
        Downsampled values aligned to time grid
    """
    if df is None or len(df) == 0:
        return np.full(len(time_grid), np.nan)
    
    # Set timestamp as index
    df = df.copy()
    df = df.set_index("timestamp")
    
    # Resample to 1 second, taking mean
    try:
        resampled = df[value_col].resample("1S").mean()
        
        # Align to our time grid
        result = np.full(len(time_grid), np.nan)
        for i, t in enumerate(time_grid):
            if t in resampled.index:
                result[i] = resampled.loc[t]
            else:
                # Find nearest timestamp within 1 second
                time_diff = np.abs((resampled.index - t).total_seconds())
                if len(time_diff) > 0 and time_diff.min() < 1.0:
                    nearest_idx = time_diff.argmin()
                    result[i] = resampled.iloc[nearest_idx]
        
        return result
    except Exception as e:
        print(f"Warning: Downsample failed for {value_col}: {e}")
        return np.full(len(time_grid), np.nan)


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


def align_to_1hz(signals: Dict[str, Optional[pd.DataFrame]],
                 start_time: pd.Timestamp,
                 end_time: pd.Timestamp) -> pd.DataFrame:
    """
    Align all modalities to a 1Hz common time grid.
    
    Strategy:
    - heatflux: already 1Hz, use directly
    - acc (~32Hz): downsample to 1Hz using mean per second
    - ppg (~64Hz): downsample to 1Hz using mean per second
    - emography (~0.017Hz): forward-fill to 1Hz
    
    Args:
        signals: Dictionary of loaded signal DataFrames
        start_time: Start of alignment window
        end_time: End of alignment window
    
    Returns:
        DataFrame with columns: timestamp, acc_x, acc_y, acc_z, acc_magnitude,
                               skin_temp, heatflux, cbt, pulse_rate,
                               ppg_mean, ppg_std, eda_stress_skin
    """
    # Create 1Hz time grid
    time_grid = create_time_grid(start_time, end_time, freq="1S")
    
    aligned = pd.DataFrame({"timestamp": time_grid})
    
    # --- Align heatflux data (already 1Hz) ---
    hf_df = signals.get("heatflux")
    if hf_df is not None and len(hf_df) > 0:
        # These columns are at 1Hz, just need to match timestamps
        for col in ["skin_temp", "heatflux", "cbt", "pulse_rate"]:
            if col in hf_df.columns:
                aligned[col] = downsample_mean(hf_df, col, time_grid)
        
        # # Also get accelerometer from heatflux if available
        # for col in ["acc_x", "acc_y", "acc_z"]:
        #     if col in hf_df.columns:
        #         aligned[f"hf_{col}"] = downsample_mean(hf_df, col, time_grid)
    
    # --- Align accelerometer (~32Hz -> 1Hz) ---
    acc_df = signals.get("acc")
    if acc_df is not None and len(acc_df) > 0:
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in acc_df.columns:
                aligned[col] = downsample_mean(acc_df, col, time_grid)
        
        # Calculate magnitude
        if all(c in aligned.columns for c in ["acc_x", "acc_y", "acc_z"]):
            aligned["acc_magnitude"] = np.sqrt(
                aligned["acc_x"]**2 + 
                aligned["acc_y"]**2 + 
                aligned["acc_z"]**2
            )
    
    # --- Align PPG (~64Hz -> 1Hz) ---
    # NOTE: This downsampling destroys the cardiac waveform, making PPG unsuitable
    # for deep learning models. We compute it here for completeness, but it should
    # NOT be used for MOMENT or other time-series models.
    ppg_df = signals.get("ppg")
    if ppg_df is not None and len(ppg_df) > 0 and "value" in ppg_df.columns:
        # Get mean PPG per second (loses cardiac pulse information)
        aligned["ppg_mean"] = downsample_mean(ppg_df, "value", time_grid)
        
        # Also calculate std per second for variability
        ppg_df_copy = ppg_df.copy().set_index("timestamp")
        try:
            ppg_std = ppg_df_copy["value"].resample("1S").std()
            std_values = np.full(len(time_grid), np.nan)
            for i, t in enumerate(time_grid):
                if t in ppg_std.index:
                    std_values[i] = ppg_std.loc[t]
            aligned["ppg_std"] = std_values
        except Exception:
            aligned["ppg_std"] = np.nan
    
    # --- Align emography (~0.017Hz -> 1Hz via forward fill) ---
    eda_df = signals.get("emography")
    if eda_df is not None and len(eda_df) > 0:
        # Look for stress_skin column
        eda_col = None
        for col in eda_df.columns:
            if "stress" in col.lower() or "skin" in col.lower():
                eda_col = col
                break
        
        if eda_col:
            aligned["eda_stress_skin"] = forward_fill_signal(eda_df, eda_col, time_grid)
    
    return aligned


if __name__ == "__main__":
    # Test alignment
    from raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
    from config import DEFAULT_CONFIG
    
    print("Testing signal alignment...")
    subjects = get_all_subjects(DEFAULT_CONFIG.data_path)
    
    if subjects:
        signals = load_raw_signals(subjects[0])
        start, end = get_experiment_time_range(signals)
        
        print(f"Aligning signals from {start} to {end}")
        aligned = align_to_1hz(signals, start, end)
        
        print(f"\nAligned DataFrame shape: {aligned.shape}")
        print(f"Columns: {list(aligned.columns)}")
        print(f"\nSample of aligned data:")
        print(aligned.head(10))
        
        # Check missing values
        print(f"\nMissing value counts:")
        for col in aligned.columns:
            if col != "timestamp":
                missing = aligned[col].isna().sum()
                pct = 100 * missing / len(aligned)
                print(f"  {col}: {missing} ({pct:.1f}%)")

