"""
HRV feature extractor using HeartPy.

Extracts heart rate and heart rate variability features from PPG signals.
Based on validated preprocessing pipeline from experimental testing.
"""

import numpy as np
import pandas as pd
import heartpy as hp
from scipy import signal as scipy_signal
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# Feature names returned by this extractor
# Time-domain features only (frequency-domain excluded due to short 120s windows)
HRV_FEATURE_NAMES = [
    'hr_bpm',           # Mean heart rate
    'hr_std',           # Heart rate standard deviation
    'hrv_mean_rr',      # Mean RR interval
    'hrv_sdnn',         # Standard deviation of RR intervals
    'hrv_rmssd',        # Root mean square of successive differences
    'hrv_pnn50',        # Percentage of successive RR differences > 50ms
    'hrv_pnn20',        # Percentage of successive RR differences > 20ms
    'hrv_sdsd',         # Standard deviation of successive differences
]
# Excluded features:
# - hrv_lf, hrv_hf, hrv_lf_hf_ratio: Frequency-domain (unreliable for 120s windows, need 5+ min)
# - breathing_rate: Spectral analysis (unreliable for short windows)
# - hr_peak_rejection_rate: Quality metric, not a physiological feature


def preprocess_ppg_segment(
    ppg_values: np.ndarray,
    sample_rate: float,
    quality_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, bool]:
    """
    Preprocess a PPG segment for HeartPy processing.
    
    Steps:
    1. Handle quality flags (set low quality to NaN)
    2. Remove zeros (device artifact)
    3. Remove outliers using percentiles
    4. Interpolate small gaps
    5. Detrend and remove DC offset
    6. Bandpass filter
    7. Scale
    
    Args:
        ppg_values: Raw PPG values
        sample_rate: Sampling rate in Hz
        quality_mask: Optional quality flags (True = good, False = bad)
    
    Returns:
        Tuple of (preprocessed array, success flag)
    """
    if len(ppg_values) < int(10 * sample_rate):  # Need at least 10 seconds
        return np.array([]), False
    
    # Work on a copy
    values = ppg_values.astype(float).copy()
    
    # Apply quality mask if provided
    if quality_mask is not None:
        values[~quality_mask] = np.nan
    
    # Replace zeros with NaN (device artifact)
    values[values == 0] = np.nan
    
    # Check for too many NaNs
    nan_ratio = np.isnan(values).sum() / len(values)
    if nan_ratio > 0.3:  # >30% missing
        return np.array([]), False
    
    # Remove outliers using percentiles
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < int(10 * sample_rate):
        return np.array([]), False
    
    q1 = np.percentile(valid_values, 1)
    q99 = np.percentile(valid_values, 99)
    values[(values < q1) | (values > q99)] = np.nan
    
    # Interpolate small gaps (max 10 samples)
    values = pd.Series(values).interpolate(method='linear', limit=10).values
    
    # Check for remaining NaNs
    if np.isnan(values).any():
        # Find longest continuous segment
        nan_mask = np.isnan(values)
        segments = []
        start = None
        for i, is_nan in enumerate(nan_mask):
            if not is_nan and start is None:
                start = i
            elif is_nan and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(values)))
        
        if not segments:
            return np.array([]), False
        
        # Use longest segment
        longest = max(segments, key=lambda x: x[1] - x[0])
        values = values[longest[0]:longest[1]]
        
        if len(values) < int(10 * sample_rate):
            return np.array([]), False
    
    # Detrend: remove DC offset and linear trend
    values = values - np.mean(values)
    values = scipy_signal.detrend(values)
    
    # Bandpass filter (0.7-3.5 Hz covers 42-210 BPM)
    try:
        filtered = hp.filter_signal(
            values,
            cutoff=[0.7, 3.5],
            sample_rate=sample_rate,
            filtertype='bandpass'
        )
    except Exception:
        return np.array([]), False
    
    # Scale for HeartPy
    scaled = hp.scale_data(filtered)
    
    return scaled, True


def extract_hrv_features(
    ppg_values: np.ndarray,
    sample_rate: float,
    quality_mask: Optional[np.ndarray] = None,
    return_working_data: bool = False
) -> Dict[str, float]:
    """
    Extract HR and HRV features from a PPG segment using HeartPy.
    
    Args:
        ppg_values: Raw PPG values for the window
        sample_rate: Sampling rate in Hz
        quality_mask: Optional boolean mask (True = good quality)
        return_working_data: If True, include working_data in return dict
    
    Returns:
        Dictionary of HRV features. Returns NaN for all features if extraction fails.
    """
    # Initialize with NaN values
    features = {name: np.nan for name in HRV_FEATURE_NAMES}
    
    # Preprocess
    preprocessed, success = preprocess_ppg_segment(ppg_values, sample_rate, quality_mask)
    
    if not success or len(preprocessed) == 0:
        if return_working_data:
            features['working_data'] = None
        return features
    
    # Process with HeartPy
    try:
        working_data, measures = hp.process(
            preprocessed,
            sample_rate=sample_rate,
            high_precision=True,
            clean_rr=True,
            clean_rr_method='quotient-filter',
            bpmmin=40,
            bpmmax=180
        )
        
        # Calculate rejection rate
        n_peaks = len(working_data.get('peaklist', []))
        n_rejected = len(working_data.get('removed_beats', []))
        total_peaks = n_peaks + n_rejected
        rejection_rate = n_rejected / total_peaks if total_peaks > 0 else 1.0
        
        # Skip if too many peaks rejected
        if rejection_rate > 0.5:  # >50% rejection
            if return_working_data:
                features['working_data'] = working_data
            return features
        
        # Extract features (convert to float explicitly)
        features['hr_bpm'] = float(measures.get('bpm', np.nan))
        features['hrv_sdnn'] = float(measures.get('sdnn', np.nan))
        features['hrv_rmssd'] = float(measures.get('rmssd', np.nan))
        features['hrv_pnn50'] = float(measures.get('pnn50', np.nan))
        features['hrv_pnn20'] = float(measures.get('pnn20', np.nan))
        features['hrv_sdsd'] = float(measures.get('sdsd', np.nan))
        # features['hrv_lf'] = float(measures.get('lf', np.nan))
        # features['hrv_hf'] = float(measures.get('hf', np.nan))
        # features['hrv_lf_hf_ratio'] = float(measures.get('lf/hf', np.nan))
        # features['breathing_rate'] = float(measures.get('breathingrate', np.nan))
        # features['hr_peak_rejection_rate'] = float(rejection_rate)
        
        # Calculate additional features from RR intervals
        rr_list = working_data.get('RR_list_cor', working_data.get('RR_list', []))
        if len(rr_list) > 1:
            features['hrv_mean_rr'] = np.mean(rr_list)
            
            # HR std from RR intervals (convert RR to HR)
            hr_values = 60000 / np.array(rr_list)  # RR in ms to BPM
            features['hr_std'] = np.std(hr_values)
        
        if return_working_data:
            features['working_data'] = working_data
        
    except Exception as e:
        # Return NaN features on failure
        if return_working_data:
            features['working_data'] = None
    
    return features


def extract_hrv_from_window(
    ppg_df: pd.DataFrame,
    window_start,
    window_end,
    sample_rate: float
) -> Dict[str, float]:
    """
    Extract HRV features for a specific time window.
    
    Args:
        ppg_df: DataFrame with 'timestamp', 'value', and optional 'quality' columns
        window_start: Window start time
        window_end: Window end time
        sample_rate: PPG sampling rate in Hz
    
    Returns:
        Dictionary of HRV features
    """
    if ppg_df is None or len(ppg_df) == 0:
        return {name: np.nan for name in HRV_FEATURE_NAMES}
    
    # Extract window data
    mask = (ppg_df['timestamp'] >= window_start) & (ppg_df['timestamp'] < window_end)
    window_df = ppg_df.loc[mask]
    
    if len(window_df) < int(10 * sample_rate):  # Need at least 10 seconds
        return {name: np.nan for name in HRV_FEATURE_NAMES}
    
    # Get values
    ppg_values = window_df['value'].values.astype(float)
    
    # Get quality mask if available
    quality_mask = None
    if 'quality' in window_df.columns:
        quality = window_df['quality'].values
        quality_mask = quality >= 3  # Quality 3+ is acceptable
    
    return extract_hrv_features(ppg_values, sample_rate, quality_mask)


class HRVExtractor:
    """
    HRV feature extractor for VitaStress PPG data.
    """
    
    def __init__(self, sample_rate: float = 64.0):
        """
        Initialize HRV extractor.
        
        Args:
            sample_rate: Expected PPG sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.feature_names = HRV_FEATURE_NAMES.copy()
    
    def extract(
        self,
        ppg_values: np.ndarray,
        quality_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract HRV features from PPG values.
        
        Args:
            ppg_values: Raw PPG array
            quality_mask: Optional quality mask
        
        Returns:
            Dictionary of HRV features
        """
        return extract_hrv_features(ppg_values, self.sample_rate, quality_mask)
    
    def extract_from_dataframe(
        self,
        ppg_df: pd.DataFrame,
        window_start,
        window_end
    ) -> Dict[str, float]:
        """
        Extract HRV features from PPG DataFrame for a window.
        
        Args:
            ppg_df: DataFrame with timestamp, value, quality columns
            window_start: Window start time
            window_end: Window end time
        
        Returns:
            Dictionary of HRV features
        """
        return extract_hrv_from_window(ppg_df, window_start, window_end, self.sample_rate)
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.feature_names.copy()


if __name__ == '__main__':
    # Test the HRV extractor
    import os
    from pathlib import Path
    
    data_path = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    
    # Get first subject
    subjects = [d for d in os.listdir(data_path) if d.startswith('id_')]
    subjects = sorted(subjects)
    
    if subjects:
        subject = subjects[0]
        subject_id = subject.replace('id_', '')
        ppg_file = os.path.join(data_path, subject, f"{subject_id}_ppg2_green_6.csv")
        
        if os.path.exists(ppg_file):
            print(f"Testing with: {subject[:20]}...")
            
            # Load PPG
            df = pd.read_csv(ppg_file)
            df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601')
            
            # Calculate sample rate
            elapsed = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
            sample_rate = len(df) / elapsed
            print(f"Sample rate: {sample_rate:.2f} Hz")
            
            # Extract 120 second window (skip first 5 minutes)
            from datetime import timedelta
            start_time = df['timestamp'].min() + timedelta(minutes=5)
            end_time = start_time + timedelta(seconds=120)
            
            print(f"\nExtracting HRV from window:")
            print(f"  Start: {start_time}")
            print(f"  End: {end_time}")
            
            extractor = HRVExtractor(sample_rate=sample_rate)
            features = extractor.extract_from_dataframe(df, start_time, end_time)
            
            print("\nExtracted features:")
            for name, value in features.items():
                if not np.isnan(value) if isinstance(value, float) else True:
                    print(f"  {name}: {value:.2f}" if isinstance(value, float) else f"  {name}: {value}")
        else:
            print(f"PPG file not found: {ppg_file}")

