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
            bpmmax=220
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


def extract_hr_timeseries_from_ppg(ppg_df: pd.DataFrame, 
                                    start_time: pd.Timestamp,
                                    end_time: pd.Timestamp,
                                    sample_rate: float = 64.0) -> Dict[str, np.ndarray]:
    """
    Extract instantaneous HR time series from PPG using HeartPy (reuses existing logic).
    
    Returns HR values at detected peak locations (not aligned to grid yet).
    Alignment to target frequency happens in align_signals().
    
    Args:
        ppg_df: DataFrame with 'timestamp', 'value', and optional 'quality' columns
        start_time: Start of time range
        end_time: End of time range
        sample_rate: PPG sampling rate in Hz (default 64Hz)
    
    Returns:
        Dictionary with 'timestamps', 'hr_bpm', 'rmssd' arrays at irregular intervals
    """
    if ppg_df is None or len(ppg_df) == 0:
        return {'timestamps': np.array([]), 'hr_bpm': np.array([]), 'rmssd': np.array([])}
    
    # Filter to time range
    mask = (ppg_df['timestamp'] >= start_time) & (ppg_df['timestamp'] <= end_time)
    ppg_segment = ppg_df.loc[mask].copy()
    
    if len(ppg_segment) < int(10 * sample_rate):
        return {'timestamps': np.array([]), 'hr_bpm': np.array([]), 'rmssd': np.array([])}
    
    # Get PPG values and timestamps
    ppg_values = ppg_segment['value'].values.astype(float)
    ppg_timestamps = ppg_segment['timestamp'].values
    
    # Get quality mask if available
    quality_mask = None
    if 'quality' in ppg_segment.columns:
        quality_mask = ppg_segment['quality'].values >= 3
    
    # Preprocess (reuse existing function)
    # preprocessed, success = preprocess_ppg_segment(ppg_values, sample_rate, quality_mask)
    preprocessed =  hp.filter_signal(ppg_values, 
                                     cutoff=[0.5, 4.0], 
                                     sample_rate=sample_rate, 
                                     order=3, 
                                     filtertype="bandpass")
        
    # if not success or len(preprocessed) == 0:
        # return {'timestamps': np.array([]), 'hr_bpm': np.array([]), 'rmssd': np.array([])}
    
    # Process with HeartPy - reuse existing logic!
    try:
        working_data, measures = hp.process(
            preprocessed,
            sample_rate=sample_rate,
            high_precision=True,
            clean_rr=True,
            clean_rr_method='quotient-filter',
            bpmmin=40,
            bpmmax=220
        )
       

        
        
        # Extract what HeartPy already computed
        peak_indices = np.array(working_data.get('peaklist', []))
        rr_intervals = np.array(working_data.get('RR_list_cor', working_data.get('RR_list', [])))
        
        if len(peak_indices) < 2 or len(rr_intervals) < 1:
            return {'timestamps': np.array([]), 'hr_bpm': np.array([]), 'rmssd': np.array([])}
        
        # Convert peak indices to timestamps (indices are sample numbers)
        peak_times = ppg_timestamps[0] + pd.to_timedelta(peak_indices / sample_rate, unit='s')
        
        # Convert RR intervals to instantaneous HR (HeartPy already cleaned them!)
        hr_at_peaks = 60000.0 / rr_intervals  # RR in ms → BPM (rr_intervals already numpy array)
        
        # Calculate rolling RMSSD (5-beat window)
        rmssd_at_peaks = []
        for i in range(len(rr_intervals)):
            if i >= 1:
                window_size = min(5, i + 1)
                start_idx = max(0, i - window_size + 1)
                rr_window = rr_intervals[start_idx:i+1]
                
                if len(rr_window) >= 2:
                    diff_rr = np.diff(rr_window)
                    rmssd = np.sqrt(np.mean(diff_rr ** 2))
                    rmssd_at_peaks.append(rmssd)
                else:
                    rmssd_at_peaks.append(np.nan)
            else:
                rmssd_at_peaks.append(np.nan)
        
        # Align lengths (peaklist has one more element than RR_list)
        min_len = min(len(peak_times), len(hr_at_peaks), len(rmssd_at_peaks))
  
        return {
            'timestamps': peak_times[:min_len],
            'hr_bpm': hr_at_peaks[:min_len],
            'rmssd': np.array(rmssd_at_peaks[:min_len])
        }
        
    except Exception as e:
        # Return empty arrays on failure (window will be rejected)
        print(f"Error extracting HR time series from PPG: {e}")
        raise e
        # return {'timestamps': np.array([]), 'hr_bpm': np.array([]), 'rmssd': np.array([])}


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






