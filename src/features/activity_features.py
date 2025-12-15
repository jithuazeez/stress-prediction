"""
Accelerometer feature extraction for EMOTIONAL STRESS PREDICTION.

Two feature groups:
1. STRESS INDICATORS: fidgeting, tremors, restlessness
2. ACTIVITY CLASSIFICATION: distinguish sitting vs. running/exercise

The goal is to detect emotional stress ONLY, not physical exertion.
Activity features help the model learn:
- "High HR + sitting still" → Likely emotional stress
- "High HR + running/walking" → Just exercise, not emotional stress

Based on: "Feature Engineering for Human Activity Recognition" (Atalaa et al.)
"""

import numpy as np
from typing import Dict, Optional
from scipy import stats
from scipy.signal import welch
from scipy.fft import fft, fftfreq


class ActivityFeatureExtractor:
    """
    Extract features from 3-axis accelerometer data.
    
    Two purposes:
    1. Detect stress-related movement (fidgeting, tremors)
    2. Classify activity level (sitting vs walking vs running)
    
    This helps distinguish:
    - Emotional stress (sitting, high HR, fidgeting)
    - Physical activity (moving, high HR, rhythmic motion)
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        """
        Initialize activity feature extractor.
        
        Args:
            sampling_rate: Will be calculated dynamically from data
        """
        self.sampling_rate = sampling_rate
    
    # =========================================================================
    # BASIC CALCULATIONS
    # =========================================================================
    
    def calculate_magnitude(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                           acc_z: np.ndarray) -> np.ndarray:
        """Calculate acceleration magnitude from 3-axis data."""
        return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    def calculate_jerk(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate jerk (rate of change of acceleration).
        
        Jerk captures sudden movements - key stress indicator:
        - Tremors, startle responses, fidgeting
        """
        if len(signal) < 2:
            return np.array([0.0])
        return np.diff(signal) * self.sampling_rate
    
    # =========================================================================
    # STRESS INDICATOR FEATURES
    # =========================================================================
    
    def calculate_sma(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                     acc_z: np.ndarray) -> float:
        """Signal Magnitude Area - overall movement intensity."""
        n = len(acc_x)
        if n == 0:
            return 0.0
        return (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) / n
    
    def calculate_ima(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                     acc_z: np.ndarray) -> float:
        """Integral of Modulus of Accelerations."""
        magnitude = self.calculate_magnitude(acc_x, acc_y, acc_z)
        return np.mean(np.abs(magnitude))
    
    def calculate_energy(self, signal: np.ndarray) -> float:
        """Signal energy = Σ(signal²) / n"""
        if len(signal) == 0:
            return 0.0
        return np.sum(signal**2) / len(signal)
    
    def calculate_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Zero-crossing rate - oscillatory motion indicator."""
        if len(signal) < 2:
            return 0.0
        mean_centered = signal - np.mean(signal)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(mean_centered))) > 0)
        return zero_crossings / len(signal)
    
    def calculate_skewness(self, signal: np.ndarray) -> float:
        """Skewness - asymmetry of movement distribution."""
        if len(signal) < 3:
            return 0.0
        return float(stats.skew(signal))
    
    def calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Kurtosis - sharp peaks indicate tremors."""
        if len(signal) < 4:
            return 0.0
        return float(stats.kurtosis(signal))
    
    def extract_jerk_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Jerk features - sudden movements, tremors, fidgeting."""
        jerk = self.calculate_jerk(signal)
        
        if len(jerk) == 0:
            return {
                "acc_jerk_mean": np.nan,
                "acc_jerk_std": np.nan,
                "acc_jerk_max": np.nan,
                "acc_jerk_energy": np.nan
            }
        
        return {
            "acc_jerk_mean": np.mean(np.abs(jerk)),
            "acc_jerk_std": np.std(jerk),
            "acc_jerk_max": np.max(np.abs(jerk)),
            "acc_jerk_energy": self.calculate_energy(jerk)
        }
    
    # =========================================================================
    # ACTIVITY CLASSIFICATION FEATURES
    # (Help distinguish sitting vs walking vs running)
    # =========================================================================
    
    def calculate_tilt_angles(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                              acc_z: np.ndarray) -> Dict[str, float]:
        """
        Calculate tilt angles - POSTURE/ACTIVITY indicator.
        
        Useful for:
        - Detecting if user is sitting (stable tilt) vs moving
        - Different activities have characteristic tilt patterns
        """
        features = {}
        
        mean_x = np.mean(acc_x)
        mean_y = np.mean(acc_y)
        mean_z = np.mean(acc_z)
        
        magnitude = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
        
        if magnitude > 0:
            features["tilt_x"] = np.degrees(np.arcsin(np.clip(mean_x / magnitude, -1, 1)))
            features["tilt_y"] = np.degrees(np.arcsin(np.clip(mean_y / magnitude, -1, 1)))
            features["tilt_z"] = np.degrees(np.arcsin(np.clip(mean_z / magnitude, -1, 1)))
        else:
            features["tilt_x"] = 0.0
            features["tilt_y"] = 0.0
            features["tilt_z"] = 0.0
        
        # Roll and pitch
        features["roll_angle"] = np.degrees(np.arctan2(mean_y, mean_z))
        features["pitch_angle"] = np.degrees(np.arctan2(-mean_x, np.sqrt(mean_y**2 + mean_z**2)))
        
        return features
    
    def calculate_dominant_frequency(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Dominant frequency - KEY for activity classification.
        
        - Walking: ~1.5-2.5 Hz
        - Running: ~2.5-4 Hz
        - Sitting: ~0 Hz (no periodic motion)
        - Fidgeting: irregular, no clear dominant frequency
        
        This helps distinguish:
        - Exercise (clear periodic motion) vs emotional stress (irregular)
        """
        features = {}
        
        if len(signal) < 8:
            return {
                "acc_dominant_freq": 0.0,
                "acc_dominant_freq_power": 0.0,
                "acc_freq_ratio_low": 0.0,
                "acc_freq_ratio_activity": 0.0
            }
        
        # Compute FFT
        n = len(signal)
        fft_vals = np.abs(fft(signal))[:n//2]
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        if len(fft_vals) == 0 or np.max(fft_vals) == 0:
            return {
                "acc_dominant_freq": 0.0,
                "acc_dominant_freq_power": 0.0,
                "acc_freq_ratio_low": 0.0,
                "acc_freq_ratio_activity": 0.0
            }
        
        psd = fft_vals ** 2
        total_power = np.sum(psd)
        
        # Dominant frequency
        features["acc_dominant_freq"] = freqs[np.argmax(psd)]
        features["acc_dominant_freq_power"] = np.max(psd) / total_power if total_power > 0 else 0.0
        
        # Frequency band ratios for activity classification
        # Low freq (0-0.5 Hz): postural sway, stillness
        # Activity freq (1-4 Hz): walking, running
        
        low_mask = (freqs >= 0) & (freqs < 0.5)
        activity_mask = (freqs >= 1) & (freqs < 4)
        
        low_power = np.sum(psd[low_mask]) if np.any(low_mask) else 0
        activity_power = np.sum(psd[activity_mask]) if np.any(activity_mask) else 0
        
        features["acc_freq_ratio_low"] = low_power / total_power if total_power > 0 else 0
        features["acc_freq_ratio_activity"] = activity_power / total_power if total_power > 0 else 0
        
        return features
    
    def calculate_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Spectral energy and entropy."""
        features = {}
        
        if len(signal) < 4:
            return {
                "acc_spectral_energy": np.nan,
                "acc_spectral_entropy": np.nan
            }
        
        n = len(signal)
        fft_vals = np.abs(fft(signal))[:n//2]
        psd = fft_vals ** 2
        
        if np.sum(psd) == 0:
            return {
                "acc_spectral_energy": 0.0,
                "acc_spectral_entropy": 0.0
            }
        
        features["acc_spectral_energy"] = np.sum(psd) / len(psd)
        
        psd_norm = psd / np.sum(psd)
        psd_norm_safe = psd_norm[psd_norm > 0]
        if len(psd_norm_safe) > 0:
            features["acc_spectral_entropy"] = -np.sum(psd_norm_safe * np.log2(psd_norm_safe))
        else:
            features["acc_spectral_entropy"] = 0.0
        
        return features
    
    def calculate_psd_bands(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Power in frequency bands.
        
        Bands chosen for activity classification:
        - VLF (0-0.5 Hz): stillness, breathing
        - LF (0.5-1.5 Hz): slow movements, fidgeting
        - Walking (1.5-2.5 Hz): walking cadence
        - Running (2.5-4 Hz): running cadence
        """
        features = {}
        
        bands = {
            "stillness": (0, 0.5),
            "slow_move": (0.5, 1.5),
            "walking": (1.5, 2.5),
            "running": (2.5, 4.0),
        }
        
        if len(signal) < self.sampling_rate * 2:
            for band_name in bands:
                features[f"acc_psd_{band_name}"] = np.nan
            features["acc_psd_total"] = np.nan
            return features
        
        try:
            nperseg = min(len(signal), int(self.sampling_rate * 2))
            freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=max(nperseg, 4))
            
            total_power = np.trapezoid(psd, freqs) if len(freqs) > 1 else 0
            features["acc_psd_total"] = total_power
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs < high)
                band_power = np.trapezoid(psd[band_mask], freqs[band_mask]) if np.any(band_mask) and len(freqs[band_mask]) > 1 else 0.0
                features[f"acc_psd_{band_name}"] = band_power
            
        except Exception:
            for band_name in bands:
                features[f"acc_psd_{band_name}"] = np.nan
            features["acc_psd_total"] = np.nan
        
        return features
    
    # =========================================================================
    # ACTIVITY LEVEL CLASSIFICATION
    # =========================================================================
    
    def classify_activity_level(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Add activity level indicators based on extracted features.
        
        Helps model learn:
        - is_stationary: likely sitting/standing still
        - is_walking: walking-like motion detected
        - is_high_activity: running or vigorous exercise
        """
        result = {}
        
        # Get relevant features
        magnitude_std = features.get("acc_magnitude_std", 0)
        dominant_freq = features.get("acc_dominant_freq", 0)
        activity_freq_ratio = features.get("acc_freq_ratio_activity", 0)
        energy = features.get("acc_energy", 0)
        
        # Stationary: low movement variability
        result["is_stationary"] = 1 if magnitude_std < 0.3 else 0
        
        # Walking: dominant freq in walking range (1.5-2.5 Hz) + moderate activity
        is_walking = (1.0 <= dominant_freq <= 3.0) and (0.3 <= magnitude_std <= 1.5)
        result["is_walking"] = 1 if is_walking else 0
        
        # High activity: high energy + high variability
        result["is_high_activity"] = 1 if (magnitude_std > 1.0 or energy > 2.0) else 0
        
        # Activity score (0-1): higher = more physical activity
        activity_score = min(1.0, (magnitude_std / 2.0 + activity_freq_ratio) / 2)
        result["activity_score"] = activity_score
        
        return result
    
    # =========================================================================
    # MAIN EXTRACTION METHOD
    # =========================================================================
    
    def extract_activity_features(self, acc_x: np.ndarray, acc_y: np.ndarray,
                                  acc_z: np.ndarray,
                                  include_frequency: bool = True) -> Dict[str, float]:
        """
        Extract all features for emotional stress detection.
        
        Features include:
        1. Stress indicators (fidgeting, tremors, variability)
        2. Activity classification (sitting vs walking vs running)
        
        The combination helps distinguish:
        - Emotional stress (sitting + high HR + fidgeting)
        - Physical exercise (running + high HR + rhythmic motion)
        
        Args:
            acc_x, acc_y, acc_z: Acceleration values for each axis
            include_frequency: Include frequency domain features
        
        Returns:
            Dictionary of ~35 features
        """
        if len(acc_x) == 0:
            return {}
        
        features = {}
        
        # Calculate magnitude
        magnitude = self.calculate_magnitude(acc_x, acc_y, acc_z)
        
        # -----------------------------------------------------------------
        # MOVEMENT VARIABILITY (stress indicators)
        # -----------------------------------------------------------------
        
        features["acc_magnitude_mean"] = np.mean(magnitude)
        features["acc_magnitude_std"] = np.std(magnitude)
        features["acc_magnitude_min"] = np.min(magnitude)
        features["acc_magnitude_max"] = np.max(magnitude)
        features["acc_magnitude_range"] = np.ptp(magnitude)
        features["acc_magnitude_median"] = np.median(magnitude)
        
        q75, q25 = np.percentile(magnitude, [75, 25])
        features["acc_magnitude_iqr"] = q75 - q25
        
        features["acc_x_std"] = np.std(acc_x)
        features["acc_y_std"] = np.std(acc_y)
        features["acc_z_std"] = np.std(acc_z)
        
        # -----------------------------------------------------------------
        # ACTIVITY LEVEL
        # -----------------------------------------------------------------
        
        features["acc_sma"] = self.calculate_sma(acc_x, acc_y, acc_z)
        features["acc_ima"] = self.calculate_ima(acc_x, acc_y, acc_z)
        features["acc_energy"] = self.calculate_energy(magnitude)
        
        # -----------------------------------------------------------------
        # MOVEMENT PATTERNS (stress indicators)
        # -----------------------------------------------------------------
        
        features["acc_zcr"] = self.calculate_zero_crossing_rate(magnitude)
        features["acc_magnitude_skewness"] = self.calculate_skewness(magnitude)
        features["acc_magnitude_kurtosis"] = self.calculate_kurtosis(magnitude)
        
        # -----------------------------------------------------------------
        # JERK FEATURES (sudden movements - KEY for stress)
        # -----------------------------------------------------------------
        
        jerk_features = self.extract_jerk_features(magnitude)
        features.update(jerk_features)
        
        # -----------------------------------------------------------------
        # ACTIVITY CLASSIFICATION FEATURES
        # -----------------------------------------------------------------
        
        # Tilt/posture
        tilt_features = self.calculate_tilt_angles(acc_x, acc_y, acc_z)
        features.update(tilt_features)
        
        # -----------------------------------------------------------------
        # FREQUENCY DOMAIN
        # -----------------------------------------------------------------
        
        if include_frequency:
            # Dominant frequency (key for activity classification)
            freq_features = self.calculate_dominant_frequency(magnitude)
            features.update(freq_features)
            
            # Spectral features
            spectral_features = self.calculate_spectral_features(magnitude)
            features.update(spectral_features)
            
            # PSD bands (activity-specific)
            psd_features = self.calculate_psd_bands(magnitude)
            features.update(psd_features)
        
        # -----------------------------------------------------------------
        # ACTIVITY LEVEL CLASSIFICATION
        # -----------------------------------------------------------------
        
        activity_class = self.classify_activity_level(features)
        features.update(activity_class)
        
        # Motion flag
        features["motion_flag"] = 1 if features["acc_magnitude_std"] > 0.5 else 0
        
        return features


def extract_activity_from_acc(acc_x: np.ndarray, acc_y: np.ndarray,
                              acc_z: np.ndarray,
                              sampling_rate: float = 1.0,
                              include_frequency: bool = True) -> Dict[str, float]:
    """Convenience function to extract activity features."""
    extractor = ActivityFeatureExtractor(sampling_rate=sampling_rate)
    return extractor.extract_activity_features(acc_x, acc_y, acc_z, include_frequency)


# List of features
STRESS_ACC_FEATURES = [
    # Movement variability
    "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_min",
    "acc_magnitude_max", "acc_magnitude_range", "acc_magnitude_median",
    "acc_magnitude_iqr", "acc_x_std", "acc_y_std", "acc_z_std",
    
    # Activity level
    "acc_sma", "acc_ima", "acc_energy",
    
    # Movement patterns
    "acc_zcr", "acc_magnitude_skewness", "acc_magnitude_kurtosis",
    
    # Jerk (stress indicators)
    "acc_jerk_mean", "acc_jerk_std", "acc_jerk_max", "acc_jerk_energy",
    
    # Posture/tilt (activity classification)
    "tilt_x", "tilt_y", "tilt_z", "roll_angle", "pitch_angle",
    
    # Frequency domain (activity classification)
    "acc_dominant_freq", "acc_dominant_freq_power",
    "acc_freq_ratio_low", "acc_freq_ratio_activity",
    "acc_spectral_energy", "acc_spectral_entropy",
    
    # PSD bands (activity classification)
    "acc_psd_stillness", "acc_psd_slow_move", "acc_psd_walking",
    "acc_psd_running", "acc_psd_total",
    
    # Activity classification
    "is_stationary", "is_walking", "is_high_activity", "activity_score",
    
    # Motion flag
    "motion_flag",
]


if __name__ == "__main__":
    print("Testing EMOTIONAL STRESS feature extraction")
    print("(with activity classification to distinguish from exercise)")
    print("="*60)
    
    np.random.seed(42)
    duration = 10
    fs = 32
    t = np.linspace(0, duration, duration * fs)
    
    # SITTING + CALM (baseline)
    sitting_calm_x = 0 + 0.02 * np.random.randn(len(t))
    sitting_calm_y = 0 + 0.02 * np.random.randn(len(t))
    sitting_calm_z = 1.0 + 0.02 * np.random.randn(len(t))
    
    # SITTING + STRESSED (fidgeting, tremors)
    sitting_stress_x = 0 + 0.12 * np.random.randn(len(t)) + 0.03 * np.sin(2*np.pi*5*t)
    sitting_stress_y = 0 + 0.10 * np.random.randn(len(t))
    sitting_stress_z = 1.0 + 0.08 * np.random.randn(len(t))
    
    # WALKING (rhythmic motion at ~2 Hz) - NOT emotional stress
    walk_x = 0.25 * np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.randn(len(t))
    walk_y = 0.15 * np.cos(2 * np.pi * 2 * t) + 0.05 * np.random.randn(len(t))
    walk_z = 1.0 + 0.10 * np.sin(2 * np.pi * 4 * t) + 0.05 * np.random.randn(len(t))
    
    # RUNNING (rhythmic motion at ~3 Hz) - NOT emotional stress
    run_x = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(len(t))
    run_y = 0.3 * np.cos(2 * np.pi * 3 * t) + 0.1 * np.random.randn(len(t))
    run_z = 1.0 + 0.2 * np.sin(2 * np.pi * 6 * t) + 0.1 * np.random.randn(len(t))
    
    extractor = ActivityFeatureExtractor(sampling_rate=fs)
    
    scenarios = [
        ("SITTING + CALM (no stress)", sitting_calm_x, sitting_calm_y, sitting_calm_z),
        ("SITTING + STRESSED (emotional)", sitting_stress_x, sitting_stress_y, sitting_stress_z),
        ("WALKING (exercise, not stress)", walk_x, walk_y, walk_z),
        ("RUNNING (exercise, not stress)", run_x, run_y, run_z),
    ]
    
    print(f"\n{'Scenario':<35} {'Std':>8} {'Jerk':>8} {'DomFreq':>8} {'Stationary':>10} {'Walking':>8} {'HighAct':>8}")
    print("-" * 95)
    
    for name, x, y, z in scenarios:
        f = extractor.extract_activity_features(x, y, z)
        print(f"{name:<35} {f['acc_magnitude_std']:>8.3f} {f['acc_jerk_mean']:>8.3f} "
              f"{f['acc_dominant_freq']:>8.2f} {f['is_stationary']:>10} "
              f"{f['is_walking']:>8} {f['is_high_activity']:>8}")
    
    print("\n" + "="*60)
    print(f"Total features: {len(STRESS_ACC_FEATURES)}")
    print("\n✅ Features help distinguish:")
    print("   - Emotional stress (sitting + fidgeting)")
    print("   - Physical activity (walking/running + rhythmic motion)")
