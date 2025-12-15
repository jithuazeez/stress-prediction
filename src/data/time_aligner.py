"""
Time alignment utilities for VitaStress multimodal data.

Aligns different modalities (PPG, accelerometer, temperature) to a common 1 Hz time grid.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class AlignedData:
    """Container for time-aligned multimodal data."""
    time_grid: pd.DatetimeIndex
    heatflux_df: Optional[pd.DataFrame]  # 1 Hz aligned temperature/heatflux data
    acc_df: Optional[pd.DataFrame]  # 1 Hz aligned accelerometer data
    ppg_df: Optional[pd.DataFrame]  # Raw PPG data (not resampled to 1Hz)
    sample_rates: Dict[str, float]  # Original sample rates


def create_time_grid(
    start_time: datetime,
    end_time: datetime,
    freq: str = "1S"
) -> pd.DatetimeIndex:
    """
    Create a regular time grid at specified frequency.
    
    Args:
        start_time: Start of the time grid
        end_time: End of the time grid
        freq: Frequency string (default "1S" for 1 Hz)
    
    Returns:
        DatetimeIndex with regular time points
    """
    return pd.date_range(start=start_time, end=end_time, freq=freq)


def calculate_sample_rate(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> float:
    """
    Calculate the sampling rate of a time series.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        Sample rate in Hz
    """
    if len(df) < 2:
        return 0.0
    
    elapsed = (df[timestamp_col].iloc[-1] - df[timestamp_col].iloc[0]).total_seconds()
    if elapsed <= 0:
        return 0.0
    
    return len(df) / elapsed


def align_heatflux_data(
    df: pd.DataFrame,
    time_grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Align heat flux sensor data to the common time grid.
    
    Heat flux data is already at 1 Hz, so this just aligns timestamps.
    
    Args:
        df: Heat flux DataFrame with columns:
            - timestamp (or date)
            - skin_temp, heatflux, acc_x, acc_y, acc_z, pulse_rate, cbt
        time_grid: Target 1 Hz time grid
    
    Returns:
        DataFrame aligned to time_grid with forward-fill for small gaps
    """
    if df is None or len(df) == 0:
        return None
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
        else:
            raise ValueError("No timestamp column found in heatflux data")
    
    # Select relevant columns
    cols_to_keep = ['timestamp', 'skin_temp', 'heatflux', 'acc_x', 'acc_y', 'acc_z', 
                    'pulse_rate', 'cbt']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Reindex to time grid with nearest neighbor interpolation
    aligned = df.reindex(time_grid, method='nearest', tolerance=pd.Timedelta(seconds=2))
    
    # Fill small gaps with forward fill (max 5 seconds)
    aligned = aligned.ffill(limit=5)
    
    # Reset index
    aligned = aligned.reset_index()
    aligned = aligned.rename(columns={'index': 'timestamp'})
    
    return aligned


def align_accelerometer_data(
    df: pd.DataFrame,
    time_grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Resample accelerometer data from ~32 Hz to 1 Hz.
    
    Uses mean aggregation within each 1-second bin.
    
    Args:
        df: Accelerometer DataFrame with columns:
            - timestamp (or date)
            - accX, accY, accZ (or acc_x, acc_y, acc_z)
        time_grid: Target 1 Hz time grid
    
    Returns:
        DataFrame resampled to 1 Hz time_grid
    """
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
        else:
            raise ValueError("No timestamp column found in accelerometer data")
    
    # Normalize column names
    col_mapping = {
        'accX': 'acc_x',
        'accY': 'acc_y', 
        'accZ': 'acc_z'
    }
    df = df.rename(columns=col_mapping)
    
    # Select relevant columns
    cols_to_keep = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Resample to 1 second using mean aggregation
    resampled = df.resample('1S').mean()
    
    # Align to time grid
    aligned = resampled.reindex(time_grid, method='nearest', tolerance=pd.Timedelta(seconds=2))
    
    # Fill small gaps
    aligned = aligned.ffill(limit=5)
    
    # Reset index
    aligned = aligned.reset_index()
    aligned = aligned.rename(columns={'index': 'timestamp'})
    
    return aligned


def load_and_prepare_ppg(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """
    Prepare PPG data for processing.
    
    PPG data is NOT resampled to 1 Hz - it needs to stay at native rate
    for HeartPy processing. This function just prepares it.
    
    Args:
        df: PPG DataFrame with columns:
            - timestamp (or date)
            - value (PPG signal)
            - quality
    
    Returns:
        Tuple of (prepared DataFrame, sample rate in Hz)
    """
    if df is None or len(df) == 0:
        return None, 0.0
    
    df = df.copy()
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
        else:
            raise ValueError("No timestamp column found in PPG data")
    
    # Calculate sample rate
    sample_rate = calculate_sample_rate(df, 'timestamp')
    
    # Select relevant columns
    cols_to_keep = ['timestamp', 'value', 'quality']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df, sample_rate


def get_common_time_range(
    dataframes: Dict[str, pd.DataFrame]
) -> Tuple[datetime, datetime]:
    """
    Find the overlapping time range across multiple dataframes.
    
    Args:
        dataframes: Dictionary of DataFrames with 'timestamp' column
    
    Returns:
        Tuple of (start_time, end_time) representing common range
    """
    starts = []
    ends = []
    
    for name, df in dataframes.items():
        if df is not None and len(df) > 0 and 'timestamp' in df.columns:
            starts.append(df['timestamp'].min())
            ends.append(df['timestamp'].max())
    
    if not starts or not ends:
        raise ValueError("No valid dataframes with timestamps found")
    
    common_start = max(starts)
    common_end = min(ends)
    
    return common_start, common_end


class TimeAligner:
    """
    Aligns multimodal VitaStress data to a common time grid.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize time aligner.
        
        Args:
            data_path: Path to VitaStress data directory
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
    
    def load_subject_data(self, subject_folder: str) -> Dict[str, pd.DataFrame]:
        """
        Load all relevant data files for a subject.
        
        Args:
            subject_folder: Subject folder name (e.g., 'id_0a73ef1b-...')
        
        Returns:
            Dictionary mapping signal type to DataFrame
        """
        subject_path = self.data_path / subject_folder
        subject_id = subject_folder.replace('id_', '')
        
        data = {}
        
        # Load heat flux sensor temperature data
        heatflux_files = list(subject_path.glob('*heat_flux_sensor_temperature*.csv'))
        if heatflux_files:
            df = pd.read_csv(heatflux_files[0])
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
            data['heatflux'] = df
        
        # Load accelerometer data
        acc_files = list(subject_path.glob('*_acc.csv'))
        if acc_files:
            df = pd.read_csv(acc_files[0])
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
            data['acc'] = df
        
        # Load PPG data (ppg2_green_6)
        ppg_files = list(subject_path.glob('*ppg2_green_6*.csv'))
        if ppg_files:
            df = pd.read_csv(ppg_files[0])
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
            data['ppg'] = df
        
        return data
    
    def align_subject_data(
        self,
        subject_folder: str,
        skip_first_minutes: int = 5
    ) -> Optional[AlignedData]:
        """
        Load and align all modalities for a subject.
        
        Args:
            subject_folder: Subject folder name
            skip_first_minutes: Minutes to skip at start (sensor settling)
        
        Returns:
            AlignedData object or None if data loading fails
        """
        # Load raw data
        raw_data = self.load_subject_data(subject_folder)
        
        if not raw_data:
            print(f"No data found for {subject_folder[:20]}...")
            return None
        
        # Calculate sample rates
        sample_rates = {}
        for name, df in raw_data.items():
            if df is not None and len(df) > 0:
                sample_rates[name] = calculate_sample_rate(df, 'timestamp')
        
        # Find common time range
        try:
            common_start, common_end = get_common_time_range(raw_data)
        except ValueError as e:
            print(f"Error finding common time range for {subject_folder[:20]}: {e}")
            return None
        
        # Skip first N minutes (sensor settling)
        common_start = common_start + timedelta(minutes=skip_first_minutes)
        
        if common_start >= common_end:
            print(f"No valid time range after skipping first {skip_first_minutes} minutes")
            return None
        
        # Create 1 Hz time grid
        time_grid = create_time_grid(common_start, common_end)
        
        # Align each modality
        heatflux_aligned = None
        if 'heatflux' in raw_data:
            heatflux_aligned = align_heatflux_data(raw_data['heatflux'], time_grid)
        
        acc_aligned = None
        if 'acc' in raw_data:
            acc_aligned = align_accelerometer_data(raw_data['acc'], time_grid)
        
        # Prepare PPG (not resampled to 1Hz)
        ppg_prepared = None
        if 'ppg' in raw_data:
            ppg_prepared, ppg_rate = load_and_prepare_ppg(raw_data['ppg'])
            if ppg_rate > 0:
                sample_rates['ppg'] = ppg_rate
            
            # Filter PPG to common time range
            if ppg_prepared is not None:
                ppg_prepared = ppg_prepared[
                    (ppg_prepared['timestamp'] >= common_start) & 
                    (ppg_prepared['timestamp'] <= common_end)
                ].reset_index(drop=True)
        
        return AlignedData(
            time_grid=time_grid,
            heatflux_df=heatflux_aligned,
            acc_df=acc_aligned,
            ppg_df=ppg_prepared,
            sample_rates=sample_rates
        )
    
    def get_aligned_dataframe(
        self,
        aligned_data: AlignedData
    ) -> pd.DataFrame:
        """
        Combine aligned data into a single DataFrame.
        
        Note: PPG is not included here as it's at different sample rate.
        
        Args:
            aligned_data: AlignedData object
        
        Returns:
            Combined DataFrame at 1 Hz
        """
        dfs_to_merge = []
        
        # Start with time grid
        base_df = pd.DataFrame({'timestamp': aligned_data.time_grid})
        
        # Merge heatflux data
        if aligned_data.heatflux_df is not None:
            hf_cols = [c for c in aligned_data.heatflux_df.columns if c != 'timestamp']
            # Prefix columns to avoid conflicts
            hf_df = aligned_data.heatflux_df.copy()
            hf_df = hf_df.rename(columns={c: f'hf_{c}' if c not in ['timestamp', 'skin_temp', 'heatflux', 'pulse_rate', 'cbt'] else c 
                                          for c in hf_cols})
            base_df = pd.merge_asof(
                base_df.sort_values('timestamp'),
                hf_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(seconds=2)
            )
        
        # Merge accelerometer data
        if aligned_data.acc_df is not None:
            # Rename to avoid conflicts with heatflux acc columns
            acc_df = aligned_data.acc_df.copy()
            acc_df = acc_df.rename(columns={
                'acc_x': 'acc_raw_x',
                'acc_y': 'acc_raw_y',
                'acc_z': 'acc_raw_z'
            })
            base_df = pd.merge_asof(
                base_df.sort_values('timestamp'),
                acc_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(seconds=2)
            )
        
        return base_df


def extract_ppg_window(
    ppg_df: pd.DataFrame,
    window_start: datetime,
    window_end: datetime
) -> np.ndarray:
    """
    Extract PPG values for a specific time window.
    
    Args:
        ppg_df: PPG DataFrame with 'timestamp' and 'value' columns
        window_start: Window start time
        window_end: Window end time
    
    Returns:
        NumPy array of PPG values within the window
    """
    if ppg_df is None or len(ppg_df) == 0:
        return np.array([])
    
    mask = (ppg_df['timestamp'] >= window_start) & (ppg_df['timestamp'] < window_end)
    window_data = ppg_df.loc[mask, 'value'].values.astype(float)
    
    return window_data


if __name__ == '__main__':
    # Test the time aligner
    data_path = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    
    aligner = TimeAligner(data_path)
    
    # Get first subject
    subjects = [d.name for d in Path(data_path).iterdir() 
                if d.is_dir() and d.name.startswith('id_')]
    subjects = sorted(subjects)
    
    if subjects:
        print(f"Testing with subject: {subjects[0][:20]}...")
        
        aligned = aligner.align_subject_data(subjects[0])
        
        if aligned:
            print(f"\nTime grid: {len(aligned.time_grid)} samples at 1 Hz")
            print(f"Duration: {(aligned.time_grid[-1] - aligned.time_grid[0]).total_seconds() / 60:.1f} minutes")
            print(f"\nSample rates:")
            for name, rate in aligned.sample_rates.items():
                print(f"  {name}: {rate:.2f} Hz")
            
            if aligned.heatflux_df is not None:
                print(f"\nHeatflux data: {len(aligned.heatflux_df)} rows")
                print(f"  Columns: {list(aligned.heatflux_df.columns)}")
            
            if aligned.acc_df is not None:
                print(f"\nAccelerometer data: {len(aligned.acc_df)} rows")
                print(f"  Columns: {list(aligned.acc_df.columns)}")
            
            if aligned.ppg_df is not None:
                print(f"\nPPG data: {len(aligned.ppg_df)} rows (raw, not resampled)")
            
            # Get combined dataframe
            combined = aligner.get_aligned_dataframe(aligned)
            print(f"\nCombined DataFrame: {len(combined)} rows x {len(combined.columns)} columns")
            print(f"  Columns: {list(combined.columns)}")

