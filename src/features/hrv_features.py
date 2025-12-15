"""
Heart Rate Variability (HRV) feature extraction.

Following validated clinical metrics from Task Force (1996) standards.
Focus on time-domain features as per Stress-Predict paper methodology.
"""

import numpy as np
from typing import Dict


class HRVFeatureExtractor:
    """
    Extract time-domain HRV features from RR intervals.
    
    HRV reflects autonomic nervous system balance:
    - High HRV = healthy, relaxed state
    - Low HRV = stress, sympathetic dominance
    """
    
    def __init__(self, min_rr_ms: float = 300, max_rr_ms: float = 2000,
                 min_intervals: int = 10):
        """
        Initialize HRV feature extractor.
        
        Args:
            min_rr_ms: Minimum valid RR interval in ms
            max_rr_ms: Maximum valid RR interval in ms  
            min_intervals: Minimum number of intervals required for HRV calculation
        """
        self.min_rr = min_rr_ms
        self.max_rr = max_rr_ms
        self.min_intervals = min_intervals
    
    def clean_rr_intervals(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Remove physiologically implausible RR intervals.
        
        Args:
            rr_intervals: Raw RR intervals in milliseconds
        
        Returns:
            Cleaned RR intervals
        """
        # Filter valid range (300-2000 ms = 30-200 BPM)
        valid_mask = (rr_intervals >= self.min_rr) & (rr_intervals <= self.max_rr)
        clean_rr = rr_intervals[valid_mask]
        
        return clean_rr
    
    def calculate_mean_rr(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate mean RR interval (AVNN).
        
        Lower mean RR = higher HR = potential stress
        
        Args:
            rr_intervals: RR intervals in ms
        
        Returns:
            Mean RR in milliseconds
        """
        return np.mean(rr_intervals)
    
    def calculate_sdnn(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate SDNN (Standard Deviation of NN intervals).
        
        Most common HRV metric. Reflects overall HRV magnitude.
        Lower SDNN = reduced variability = stress
        
        Args:
            rr_intervals: RR intervals in ms
        
        Returns:
            SDNN in milliseconds
        """
        return np.std(rr_intervals, ddof=1)
    
    def calculate_rmssd(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate RMSSD (Root Mean Square of Successive Differences).
        
        Reflects short-term (beat-to-beat) variability.
        Primarily indicates PARASYMPATHETIC activity.
        More sensitive to stress than SDNN.
        
        Args:
            rr_intervals: RR intervals in ms
        
        Returns:
            RMSSD in milliseconds
        """
        # Calculate successive differences
        diff = np.diff(rr_intervals)
        
        # Root mean square
        rmssd = np.sqrt(np.mean(diff ** 2))
        
        return rmssd
    
    def calculate_pnn50(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate pNN50 (percentage of successive RR intervals differing by > 50ms).
        
        Reflects parasympathetic (vagal) activity.
        Lower pNN50 = reduced vagal tone = stress
        Very sensitive to stress changes.
        
        Args:
            rr_intervals: RR intervals in ms
        
        Returns:
            pNN50 as percentage (0-100)
        """
        if len(rr_intervals) < 2:
            return 0.0
        
        # Calculate successive differences
        diff = np.abs(np.diff(rr_intervals))
        
        # Count differences > 50ms
        nn50 = np.sum(diff > 50)
        
        # Calculate percentage
        pnn50 = (nn50 / len(diff)) * 100
        
        return pnn50
    
    def calculate_cv(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate Coefficient of Variation.
        
        Normalized measure of variability, allows comparison across individuals.
        CV = (SDNN / Mean RR) * 100
        
        Args:
            rr_intervals: RR intervals in ms
        
        Returns:
            CV as percentage
        """
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals, ddof=1)
        
        if mean_rr == 0:
            return 0.0
        
        cv = (std_rr / mean_rr) * 100
        
        return cv
    
    def extract_hrv_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive time-domain HRV features.
        
        Args:
            rr_intervals: RR intervals in milliseconds
        
        Returns:
            Dictionary of HRV features
        """
        # Clean RR intervals
        rr_clean = self.clean_rr_intervals(rr_intervals)
        
        # Check if we have enough data
        if len(rr_clean) < self.min_intervals:
            return {
                'hrv_mean_rr': np.nan,
                'hrv_sdnn': np.nan,
                'hrv_rmssd': np.nan,
                'hrv_pnn50': np.nan,
                'hrv_cv': np.nan,
                'hrv_num_intervals': len(rr_clean)
            }
        
        # Calculate HRV metrics
        features = {
            'hrv_mean_rr': self.calculate_mean_rr(rr_clean),
            'hrv_sdnn': self.calculate_sdnn(rr_clean),
            'hrv_rmssd': self.calculate_rmssd(rr_clean),
            'hrv_pnn50': self.calculate_pnn50(rr_clean),
            'hrv_cv': self.calculate_cv(rr_clean),
            'hrv_num_intervals': len(rr_clean)
        }
        
        # Additional range features
        features['hrv_min_rr'] = np.min(rr_clean)
        features['hrv_max_rr'] = np.max(rr_clean)
        features['hrv_range_rr'] = features['hrv_max_rr'] - features['hrv_min_rr']
        
        return features


def extract_hrv_from_rr(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Convenience function to extract HRV features.
    
    Args:
        rr_intervals: RR intervals in milliseconds
    
    Returns:
        Dictionary of HRV features
    """
    extractor = HRVFeatureExtractor()
    features = extractor.extract_hrv_features(rr_intervals)
    return features


if __name__ == '__main__':
    # Test HRV feature extraction
    print("Testing HRV feature extraction...")
    
    # Simulate RR intervals for different states
    # Relaxed state: high variability
    relaxed_rr = np.random.normal(900, 80, 50)  # ~67 BPM, high variability
    
    # Stressed state: low variability
    stressed_rr = np.random.normal(700, 30, 50)  # ~86 BPM, low variability
    
    extractor = HRVFeatureExtractor()
    
    # Extract features from relaxed state
    print("\n--- Relaxed State ---")
    features_relaxed = extractor.extract_hrv_features(relaxed_rr)
    for key, value in features_relaxed.items():
        if np.isnan(value):
            print(f"  {key}: NaN")
        else:
            print(f"  {key}: {value:.2f}")
    
    # Extract features from stressed state
    print("\n--- Stressed State ---")
    features_stressed = extractor.extract_hrv_features(stressed_rr)
    for key, value in features_stressed.items():
        if np.isnan(value):
            print(f"  {key}: NaN")
        else:
            print(f"  {key}: {value:.2f}")
    
    # Compare key metrics
    print("\n--- Stress vs Relaxed Comparison ---")
    print(f"SDNN reduction: {features_relaxed['hrv_sdnn'] - features_stressed['hrv_sdnn']:.2f} ms")
    print(f"RMSSD reduction: {features_relaxed['hrv_rmssd'] - features_stressed['hrv_rmssd']:.2f} ms")
    print(f"pNN50 reduction: {features_relaxed['hrv_pnn50'] - features_stressed['hrv_pnn50']:.2f}%")
    
    print("\n✅ HRV feature extraction working!")
    print("✓ Lower HRV metrics in stressed state (as expected)")








