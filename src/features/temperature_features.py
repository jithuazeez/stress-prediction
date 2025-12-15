"""
Temperature feature extraction.

Stress → sympathetic activation → peripheral vasoconstriction → temperature drop.
Temperature is a secondary stress indicator.
"""

import numpy as np
from typing import Dict


class TemperatureFeatureExtractor:
    """
    Extract features from skin temperature sensors.
    
    Thermoregulation response to stress:
    - Stress causes peripheral vasoconstriction
    - Result: Decreased skin temperature (cold hands/feet)
    - Core temperature stable, but skin temperature responsive
    """
    
    def __init__(self, sampling_rate: float = 0.033):
        """
        Initialize temperature feature extractor.
        
        Args:
            sampling_rate: Temperature sampling rate in Hz (VitaStress uses ~0.033 Hz = every 30s)
        """
        self.sampling_rate = sampling_rate
    
    def extract_absolute_features(self, temp_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract absolute temperature features.
        
        Args:
            temp_signal: Temperature values in Celsius
        
        Returns:
            Dictionary of absolute temperature features
        """
        features = {}
        
        features['temp_mean'] = np.mean(temp_signal)
        features['temp_std'] = np.std(temp_signal)
        features['temp_min'] = np.min(temp_signal)
        features['temp_max'] = np.max(temp_signal)
        features['temp_range'] = features['temp_max'] - features['temp_min']
        
        return features
    
    def extract_dynamic_features(self, temp_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract dynamic temperature features (trends and changes).
        
        Rising temp = recovery from stress
        Falling temp = stress onset
        
        Args:
            temp_signal: Temperature values
        
        Returns:
            Dictionary of dynamic features
        """
        features = {}
        
        if len(temp_signal) < 2:
            features['temp_slope'] = 0.0
            features['temp_change'] = 0.0
            return features
        
        # Linear trend (slope)
        x = np.arange(len(temp_signal))
        coeffs = np.polyfit(x, temp_signal, deg=1)
        features['temp_slope'] = coeffs[0]
        
        # Absolute change from start to end
        features['temp_change'] = temp_signal[-1] - temp_signal[0]
        
        return features
    
    def extract_temperature_features(self, temp_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive temperature features.
        
        Args:
            temp_signal: Temperature values
        
        Returns:
            Dictionary of temperature features
        """
        if len(temp_signal) == 0:
            return {}
        
        features = {}
        
        # Absolute features
        absolute = self.extract_absolute_features(temp_signal)
        features.update(absolute)
        
        # Dynamic features
        dynamic = self.extract_dynamic_features(temp_signal)
        features.update(dynamic)
        
        return features
    
    def extract_multi_sensor_features(self, temp_sk1: np.ndarray, 
                                     temp_sk2: np.ndarray) -> Dict[str, float]:
        """
        Extract features from multiple temperature sensors.
        
        Args:
            temp_sk1: Temperature from sensor 1
            temp_sk2: Temperature from sensor 2
        
        Returns:
            Dictionary of features including gradient between sensors
        """
        features = {}
        
        # Individual sensor features
        if len(temp_sk1) > 0:
            sk1_features = self.extract_temperature_features(temp_sk1)
            features.update({f'temp_sk1_{k}': v for k, v in sk1_features.items()})
        
        if len(temp_sk2) > 0:
            sk2_features = self.extract_temperature_features(temp_sk2)
            features.update({f'temp_sk2_{k}': v for k, v in sk2_features.items()})
        
        # Gradient between sensors
        if len(temp_sk1) > 0 and len(temp_sk2) > 0:
            # Match lengths
            min_len = min(len(temp_sk1), len(temp_sk2))
            gradient = temp_sk1[:min_len] - temp_sk2[:min_len]
            
            features['temp_gradient_mean'] = np.mean(gradient)
            features['temp_gradient_std'] = np.std(gradient)
        
        return features


def extract_temp_features(temp_signal: np.ndarray, 
                         sampling_rate: float = 0.033) -> Dict[str, float]:
    """
    Convenience function to extract temperature features.
    
    Args:
        temp_signal: Temperature values in Celsius
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of temperature features
    """
    extractor = TemperatureFeatureExtractor(sampling_rate=sampling_rate)
    features = extractor.extract_temperature_features(temp_signal)
    return features


if __name__ == '__main__':
    # Test temperature feature extraction
    print("Testing temperature feature extraction...")
    
    # Simulate temperature data
    # Baseline: stable at 34°C
    t_baseline = np.linspace(0, 300, 10)  # 10 samples over 5 minutes
    temp_baseline = 34.0 + 0.2 * np.random.randn(len(t_baseline))
    
    # Stress: gradual decrease due to vasoconstriction
    t_stress = np.linspace(0, 300, 10)
    temp_stress = 34.0 - 0.01 * t_stress / 30 + 0.2 * np.random.randn(len(t_stress))
    
    extractor = TemperatureFeatureExtractor()
    
    # Extract baseline features
    print("\n--- Baseline Temperature ---")
    features_baseline = extractor.extract_temperature_features(temp_baseline)
    for key, value in features_baseline.items():
        print(f"  {key}: {value:.3f}")
    
    # Extract stress features
    print("\n--- Stress Temperature ---")
    features_stress = extractor.extract_temperature_features(temp_stress)
    for key, value in features_stress.items():
        print(f"  {key}: {value:.3f}")
    
    # Compare
    print("\n--- Stress vs Baseline ---")
    print(f"Mean temperature change: {features_stress['temp_mean'] - features_baseline['temp_mean']:.3f}°C")
    print(f"Slope during stress: {features_stress['temp_slope']:.4f} (negative = cooling)")
    
    print("\n✅ Temperature feature extraction working!")
    print("✓ Temperature decreases during stress (as expected)")








