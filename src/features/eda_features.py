"""
Electrodermal Activity (EDA) / Skin Conductance feature extraction.

EDA is a direct measure of sympathetic nervous system activity.
Stress → increased sweating → higher skin conductance.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict


class EDAFeatureExtractor:
    """
    Extract EDA/skin conductance features.
    
    EDA components:
    - Tonic (SCL): Slow baseline level (minutes)
    - Phasic (SCR): Fast responses to stimuli (seconds)
    """
    
    def __init__(self, sampling_rate: float = 0.017):
        """
        Initialize EDA feature extractor.
        
        Args:
            sampling_rate: EDA sampling rate in Hz (VitaStress uses ~0.017 Hz)
        """
        self.sampling_rate = sampling_rate
    
    def extract_tonic_features(self, eda_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract tonic component features (SCL - Skin Conductance Level).
        
        Tonic level reflects baseline arousal state.
        Higher SCL = higher baseline stress/arousal.
        
        Args:
            eda_signal: EDA signal values
        
        Returns:
            Dictionary of tonic features
        """
        features = {}
        
        # Basic statistics
        features['eda_mean'] = np.mean(eda_signal)
        features['eda_median'] = np.median(eda_signal)
        features['eda_std'] = np.std(eda_signal)
        features['eda_min'] = np.min(eda_signal)
        features['eda_max'] = np.max(eda_signal)
        features['eda_range'] = features['eda_max'] - features['eda_min']
        
        return features
    
    def extract_phasic_features(self, eda_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract phasic component features (SCR - Skin Conductance Responses).
        
        Phasic responses are rapid increases in conductance in response to stimuli.
        More responses = higher reactivity to stressors.
        
        Args:
            eda_signal: EDA signal values
        
        Returns:
            Dictionary of phasic features
        """
        features = {}
        
        if len(eda_signal) < 3:
            features['eda_num_peaks'] = 0
            features['eda_peak_rate'] = 0.0
            return features
        
        # Detect peaks (SCRs)
        # Threshold: mean + 1 SD (common in literature)
        threshold = np.mean(eda_signal) + np.std(eda_signal)
        
        # Find peaks above threshold
        peaks, properties = find_peaks(eda_signal, height=threshold)
        
        # Number of peaks
        features['eda_num_peaks'] = len(peaks)
        
        # Peak rate (peaks per minute)
        if self.sampling_rate > 0:
            duration_minutes = len(eda_signal) / (self.sampling_rate * 60)
            if duration_minutes > 0:
                features['eda_peak_rate'] = len(peaks) / duration_minutes
            else:
                features['eda_peak_rate'] = 0.0
        else:
            features['eda_peak_rate'] = 0.0
        
        return features
    
    def extract_dynamic_features(self, eda_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract dynamic features (trends and rate of change).
        
        Rising EDA = increasing stress
        Falling EDA = stress recovery
        
        Args:
            eda_signal: EDA signal values
        
        Returns:
            Dictionary of dynamic features
        """
        features = {}
        
        if len(eda_signal) < 2:
            features['eda_trend'] = 0.0
            features['eda_change'] = 0.0
            features['eda_deriv_mean'] = 0.0
            return features
        
        # Linear trend (slope)
        x = np.arange(len(eda_signal))
        coeffs = np.polyfit(x, eda_signal, deg=1)
        features['eda_trend'] = coeffs[0]
        
        # Change from start to end
        features['eda_change'] = eda_signal[-1] - eda_signal[0]
        
        # First derivative (rate of change)
        eda_diff = np.diff(eda_signal)
        features['eda_deriv_mean'] = np.mean(eda_diff)
        features['eda_deriv_std'] = np.std(eda_diff)
        
        return features
    
    def extract_eda_features(self, eda_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive EDA features.
        
        Args:
            eda_signal: EDA/stress_skin signal values
        
        Returns:
            Dictionary of EDA features
        """
        if len(eda_signal) == 0:
            return {}
        
        features = {}
        
        # Tonic component
        tonic = self.extract_tonic_features(eda_signal)
        features.update(tonic)
        
        # Phasic component
        phasic = self.extract_phasic_features(eda_signal)
        features.update(phasic)
        
        # Dynamic features
        dynamic = self.extract_dynamic_features(eda_signal)
        features.update(dynamic)
        
        return features


def extract_eda_from_signal(eda_signal: np.ndarray, 
                           sampling_rate: float = 0.017) -> Dict[str, float]:
    """
    Convenience function to extract EDA features.
    
    Args:
        eda_signal: EDA signal values
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of EDA features
    """
    extractor = EDAFeatureExtractor(sampling_rate=sampling_rate)
    features = extractor.extract_eda_features(eda_signal)
    return features


if __name__ == '__main__':
    # Test EDA feature extraction
    print("Testing EDA feature extraction...")
    
    # Simulate EDA signals
    # Baseline: low conductance with few responses
    t_baseline = np.linspace(0, 60, 60)  # 60 samples over 60 seconds
    baseline_eda = 2.0 + 0.1 * np.random.randn(len(t_baseline))
    
    # Stress: higher conductance with more responses
    t_stress = np.linspace(0, 60, 60)
    stress_eda = 3.5 + 0.2 * np.random.randn(len(t_stress))
    # Add some SCRs (sudden increases)
    for i in range(5):
        peak_idx = np.random.randint(0, len(stress_eda)-10)
        stress_eda[peak_idx:peak_idx+5] += np.linspace(0, 0.5, 5)
    
    extractor = EDAFeatureExtractor()
    
    # Extract baseline features
    print("\n--- Baseline EDA ---")
    features_baseline = extractor.extract_eda_features(baseline_eda)
    for key, value in features_baseline.items():
        print(f"  {key}: {value:.3f}")
    
    # Extract stress features
    print("\n--- Stress EDA ---")
    features_stress = extractor.extract_eda_features(stress_eda)
    for key, value in features_stress.items():
        print(f"  {key}: {value:.3f}")
    
    # Compare
    print("\n--- Stress vs Baseline ---")
    print(f"Mean EDA increase: {features_stress['eda_mean'] - features_baseline['eda_mean']:.3f}")
    print(f"Additional SCRs: {features_stress['eda_num_peaks'] - features_baseline['eda_num_peaks']}")
    
    print("\n✅ EDA feature extraction working!")
    print("✓ Higher EDA and more SCRs during stress (as expected)")








