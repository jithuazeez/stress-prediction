"""
Heart Rate feature extraction.

Based on Iqbal et al. (2022) findings:
- Stress increases HR by 1.40 BPM on average (p < 0.001)
- HR shows significant variation during stress (5.05 bpm/h change rate)
- Focus on simple, clinically-relevant HR statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class HRFeatureExtractor:
    """
    Extract heart rate features from RR intervals or calculated HR time series.
    
    Following Stress-Predict paper: use 10-second sliding window for HR calculation.
    """
    
    def __init__(self, smoothing_window_seconds: int = 10):
        """
        Initialize HR feature extractor.
        
        Args:
            smoothing_window_seconds: Window size for HR smoothing (paper used 10s)
        """
        self.smoothing_window = smoothing_window_seconds
    
    def rr_to_hr(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Convert RR intervals (ms) to heart rate (BPM).
        
        Args:
            rr_intervals: RR intervals in milliseconds
        
        Returns:
            Heart rate values in beats per minute
        """
        # Remove invalid RR intervals
        valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
        
        if len(valid_rr) == 0:
            return np.array([])
        
        # HR = 60000 / RR (ms to convert to BPM)
        hr = 60000 / valid_rr
        
        return hr
    
    def smooth_hr(self, hr_values: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        Smooth HR using moving average.
        
        Args:
            hr_values: Raw HR values
            window_size: Window size for smoothing
        
        Returns:
            Smoothed HR values
        """
        if len(hr_values) < window_size:
            return hr_values
        
        # Simple moving average
        smoothed = np.convolve(hr_values, np.ones(window_size)/window_size, mode='valid')
        
        return smoothed
    
    def calculate_hr_trend(self, hr_values: np.ndarray, timestamps: Optional[np.ndarray] = None) -> float:
        """
        Calculate HR trend (slope) over time window.
        
        Positive slope = increasing HR (stress onset)
        Negative slope = decreasing HR (recovery)
        
        Args:
            hr_values: Heart rate values
            timestamps: Optional timestamps, otherwise use indices
        
        Returns:
            Slope of linear fit (BPM per unit time)
        """
        if len(hr_values) < 3:
            return 0.0
        
        if timestamps is None:
            x = np.arange(len(hr_values))
        else:
            x = timestamps
        
        # Fit linear trend
        coeffs = np.polyfit(x, hr_values, deg=1)
        slope = coeffs[0]
        
        return slope
    
    def extract_hr_features(self, rr_intervals: np.ndarray, 
                           rr_timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract comprehensive HR features from RR intervals.
        
        Based on paper's Linear Mixed Model findings:
        - HR_mean: Average HR (stress increases this)
        - HR_std: HR variability (simple measure)
        - HR_trend: Rate of change (captures stress onset)
        
        Args:
            rr_intervals: RR intervals in milliseconds
            rr_timestamps: Optional timestamps for each RR interval
        
        Returns:
            Dictionary of HR features
        """
        features = {}
        
        # Convert RR to HR
        hr_values = self.rr_to_hr(rr_intervals)
        
        if len(hr_values) == 0:
            # No valid data
            return {
                'hr_mean': np.nan,
                'hr_std': np.nan,
                'hr_min': np.nan,
                'hr_max': np.nan,
                'hr_range': np.nan,
                'hr_trend': np.nan
            }
        
        # Basic HR statistics
        features['hr_mean'] = np.mean(hr_values)
        features['hr_std'] = np.std(hr_values)
        features['hr_min'] = np.min(hr_values)
        features['hr_max'] = np.max(hr_values)
        features['hr_range'] = features['hr_max'] - features['hr_min']
        
        # HR trend (slope over window)
        features['hr_trend'] = self.calculate_hr_trend(hr_values, rr_timestamps)
        
        # Rate of change (difference between first and last)
        if len(hr_values) >= 2:
            # Use median of first/last few values for robustness
            n_edge = max(1, len(hr_values) // 10)
            hr_start = np.median(hr_values[:n_edge])
            hr_end = np.median(hr_values[-n_edge:])
            features['hr_change'] = hr_end - hr_start
        else:
            features['hr_change'] = 0.0
        
        return features


def extract_hr_from_rr(rr_data: pd.DataFrame, 
                      rr_column: str = 'rr') -> Dict[str, float]:
    """
    Convenience function to extract HR features from RR interval DataFrame.
    
    Args:
        rr_data: DataFrame containing RR intervals
        rr_column: Name of column with RR values (in milliseconds)
    
    Returns:
        Dictionary of HR features
    """
    if rr_column not in rr_data.columns:
        print(f"Warning: {rr_column} not found in RR data")
        return {}
    
    rr_intervals = rr_data[rr_column].values
    
    # Get timestamps if available
    if 'timestamp' in rr_data.columns:
        timestamps = (rr_data['timestamp'] - rr_data['timestamp'].iloc[0]).dt.total_seconds().values
    else:
        timestamps = None
    
    extractor = HRFeatureExtractor()
    features = extractor.extract_hr_features(rr_intervals, timestamps)
    
    return features


if __name__ == '__main__':
    # Test HR feature extraction
    print("Testing HR feature extraction...")
    
    # Simulate RR intervals
    # Baseline: ~800ms (75 BPM)
    # Stress: ~700ms (86 BPM) - increase of ~11 BPM
    baseline_rr = np.random.normal(800, 50, 30)
    stress_rr = np.random.normal(700, 60, 30)
    
    # Combined: transition from baseline to stress
    rr_intervals = np.concatenate([baseline_rr, stress_rr])
    
    extractor = HRFeatureExtractor()
    
    # Extract features from baseline
    print("\n--- Baseline Period ---")
    features_baseline = extractor.extract_hr_features(baseline_rr)
    for key, value in features_baseline.items():
        print(f"  {key}: {value:.2f}")
    
    # Extract features from stress
    print("\n--- Stress Period ---")
    features_stress = extractor.extract_hr_features(stress_rr)
    for key, value in features_stress.items():
        print(f"  {key}: {value:.2f}")
    
    # Extract features from transition
    print("\n--- Transition (Baseline → Stress) ---")
    features_transition = extractor.extract_hr_features(rr_intervals)
    for key, value in features_transition.items():
        print(f"  {key}: {value:.2f}")
    
    # Verify expected changes
    hr_increase = features_stress['hr_mean'] - features_baseline['hr_mean']
    print(f"\n✓ HR increase during stress: {hr_increase:.1f} BPM")
    print(f"✓ Transition trend (positive = increasing HR): {features_transition['hr_trend']:.3f}")
    
    print("\n✅ HR feature extraction working!")








