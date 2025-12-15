"""
Feature extraction for classical ML models - EMOTIONAL STRESS PREDICTION.

Goal: Detect emotional/mental stress, NOT physical exertion.

Features include:
1. STRESS INDICATORS: fidgeting, tremors, restlessness
2. ACTIVITY CLASSIFICATION: distinguish sitting vs exercise

This helps the model learn:
- Emotional stress (sitting + fidgeting + elevated arousal)
- Exercise (running + rhythmic motion) → NOT emotional stress

Key features:
- Dynamic sampling rate (never hardcoded)
- Missing data analysis per modality
- Class ratio reporting

Features (~61 total, NO PPG):
- Accelerometer (41): stress indicators + activity classification
- Temperature (7): skin_temp stats, slope, change
- Heat Flux (4): heatflux mean, std, cbt mean, change
- EDA (9): tonic stats, peaks, trend, change

NOTE: PPG features REMOVED because resampling from 64Hz to 1Hz destroys
cardiac waveform structure. For meaningful HR/HRV extraction, need native
sampling rate and HeartPy or similar toolkit.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

# Add src to path for importing existing feature extractors
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from features.activity_features import ActivityFeatureExtractor, STRESS_ACC_FEATURES
    EXTRACTORS_AVAILABLE = True
except ImportError:
    EXTRACTORS_AVAILABLE = False
    STRESS_ACC_FEATURES = []
    print("Warning: Could not import feature extractors from src/features/")

# Module logger
logger = logging.getLogger(__name__)


def calculate_sampling_rate(timestamps: pd.Series) -> float:
    """
    Calculate sampling rate from timestamps.
    
    NEVER hardcode sampling rate - always calculate from data.
    
    Args:
        timestamps: Series of timestamps
    
    Returns:
        Sampling rate in Hz
    """
    if len(timestamps) < 2:
        return 1.0  # Default fallback
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)
    
    # Calculate time differences
    time_diffs = timestamps.diff().dropna()
    
    # Convert to seconds
    time_diffs_sec = time_diffs.dt.total_seconds()
    
    # Filter out outliers (>10 seconds gaps are likely data gaps, not sampling)
    valid_diffs = time_diffs_sec[(time_diffs_sec > 0) & (time_diffs_sec < 10)]
    
    if len(valid_diffs) == 0:
        return 1.0
    
    # Median interval (more robust than mean)
    median_interval = valid_diffs.median()
    
    if median_interval > 0:
        return 1.0 / median_interval
    else:
        return 1.0


def analyze_missing_data(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict]:
    """
    Analyze missing data for each modality/feature group.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
    
    Returns:
        Dictionary with missing analysis per modality
    """
    analysis = {}
    
    # Group features by modality
    modalities = {
        "accelerometer": [c for c in feature_cols if c.startswith("acc_") or 
                         c in ["motion_flag", "is_stationary", "is_walking", 
                               "is_high_activity", "activity_score",
                               "tilt_x", "tilt_y", "tilt_z", 
                               "roll_angle", "pitch_angle"]],
        "temperature": [c for c in feature_cols if c.startswith("temp_")],
        "heatflux": [c for c in feature_cols if c.startswith("heatflux") or c.startswith("cbt")],
        "eda": [c for c in feature_cols if c.startswith("eda_")],
        # PPG removed - 1Hz resampling destroys cardiac waveform structure
    }
    
    for modality, cols in modalities.items():
        cols_in_df = [c for c in cols if c in df.columns]
        
        if not cols_in_df:
            analysis[modality] = {
                "n_features": 0,
                "missing_rate": 1.0,
                "features_missing": {},
                "status": "NOT_AVAILABLE"
            }
            continue
        
        # Count missing per feature
        feature_missing = {}
        for col in cols_in_df:
            n_missing = df[col].isna().sum()
            pct_missing = n_missing / len(df) * 100
            feature_missing[col] = {
                "n_missing": n_missing,
                "pct_missing": pct_missing
            }
        
        # Overall modality stats
        total_missing = sum(df[c].isna().sum() for c in cols_in_df)
        total_cells = len(df) * len(cols_in_df)
        overall_missing_rate = total_missing / total_cells if total_cells > 0 else 0
        
        # Rows with any missing in this modality
        rows_with_any_missing = df[cols_in_df].isna().any(axis=1).sum()
        
        analysis[modality] = {
            "n_features": len(cols_in_df),
            "missing_rate": overall_missing_rate,
            "rows_with_missing": rows_with_any_missing,
            "pct_rows_missing": rows_with_any_missing / len(df) * 100 if len(df) > 0 else 0,
            "features_missing": feature_missing,
            "status": "OK" if overall_missing_rate < 0.5 else "HIGH_MISSING"
        }
    
    return analysis


def calculate_class_ratio(labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate class distribution and ratio.
    
    For EMOTIONAL stress prediction:
    - Stress (1): Emotional/mental stress (cognitive, social)
    - No Stress (0): Baseline OR physical exercise
    
    Args:
        labels: Array of binary labels (0/1)
    
    Returns:
        Dictionary with class statistics
    """
    n_total = len(labels)
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)
    
    ratio = n_negative / n_positive if n_positive > 0 else float('inf')
    
    return {
        "n_total": n_total,
        "n_emotional_stress": n_positive,
        "n_no_stress_or_physical": n_negative,
        "pct_emotional_stress": n_positive / n_total * 100 if n_total > 0 else 0,
        "pct_no_stress": n_negative / n_total * 100 if n_total > 0 else 0,
        "class_ratio": f"1:{ratio:.1f}",
        "imbalance_ratio": ratio
    }


class BasicFeatureExtractor:
    """
    Extract features for EMOTIONAL stress detection.
    
    Two feature groups:
    1. STRESS INDICATORS: fidgeting, tremors, variability
    2. ACTIVITY CLASSIFICATION: sitting vs walking vs running
    
    This helps distinguish:
    - Emotional stress (sitting + high HR + fidgeting) → STRESS
    - Physical activity (running + high HR + rhythmic) → NOT STRESS
    
    NO HRV extraction - just simple stats from each modality.
    """
    
    def __init__(self, default_sampling_rate: float = 1.0):
        """
        Initialize feature extractors.
        
        Args:
            default_sampling_rate: Default sampling rate if can't be calculated
        """
        self.default_sampling_rate = default_sampling_rate
        self.acc_extractor = None
        
        if EXTRACTORS_AVAILABLE:
            # Will set actual sampling rate when extracting
            self.acc_extractor = ActivityFeatureExtractor(sampling_rate=default_sampling_rate)
    
    def _get_window_sampling_rate(self, window_df: pd.DataFrame) -> float:
        """Calculate sampling rate from window data."""
        if "timestamp" in window_df.columns:
            return calculate_sampling_rate(window_df["timestamp"])
        return self.default_sampling_rate
    
    def extract_accelerometer_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract accelerometer features for emotional stress detection.
        
        Two types of features:
        1. STRESS INDICATORS: fidgeting, tremors, variability
        2. ACTIVITY CLASSIFICATION: sitting vs exercise
        
        Dynamically calculates sampling rate from the data.
        """
        features = {}
        
        # Find accelerometer columns
        x_col = y_col = z_col = None
        for col in window_df.columns:
            col_lower = col.lower()
            if "acc" in col_lower:
                if "_x" in col_lower or col_lower.endswith("x"):
                    x_col = col
                elif "_y" in col_lower or col_lower.endswith("y"):
                    y_col = col
                elif "_z" in col_lower or col_lower.endswith("z"):
                    z_col = col
        
        if x_col is None or y_col is None or z_col is None:
            return {name: np.nan for name in STRESS_ACC_FEATURES}
        
        # Get values
        acc_x = window_df[x_col].dropna().values
        acc_y = window_df[y_col].dropna().values
        acc_z = window_df[z_col].dropna().values
        
        # Ensure same length
        min_len = min(len(acc_x), len(acc_y), len(acc_z))
        if min_len < 2:
            return {name: np.nan for name in STRESS_ACC_FEATURES}
        
        acc_x = acc_x[:min_len]
        acc_y = acc_y[:min_len]
        acc_z = acc_z[:min_len]
        
        # Get sampling rate from data (NEVER hardcoded)
        sampling_rate = self._get_window_sampling_rate(window_df)
        
        # Use activity extractor with stress + activity features
        if self.acc_extractor is not None:
            try:
                # Update sampling rate dynamically
                self.acc_extractor.sampling_rate = sampling_rate
                
                include_freq = min_len >= 10
                features = self.acc_extractor.extract_activity_features(
                    acc_x, acc_y, acc_z,
                    include_frequency=include_freq
                )
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                features = self._extract_basic_acc_features(acc_x, acc_y, acc_z, sampling_rate)
        else:
            features = self._extract_basic_acc_features(acc_x, acc_y, acc_z, sampling_rate)
        
        return features
    
    def _extract_basic_acc_features(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                                    acc_z: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Fallback basic accelerometer feature extraction."""
        features = {}
        
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Basic stats
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
        
        features["acc_sma"] = (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + 
                              np.sum(np.abs(acc_z))) / len(acc_x)
        features["acc_ima"] = np.mean(np.abs(magnitude))
        features["acc_energy"] = np.sum(magnitude**2) / len(magnitude)
        
        mean_centered = magnitude - np.mean(magnitude)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(mean_centered))) > 0)
        features["acc_zcr"] = zero_crossings / len(mean_centered) if len(mean_centered) > 1 else 0.0
        
        features["acc_magnitude_skewness"] = float(stats.skew(magnitude)) if len(magnitude) > 2 else 0.0
        features["acc_magnitude_kurtosis"] = float(stats.kurtosis(magnitude)) if len(magnitude) > 3 else 0.0
        
        # Jerk - use dynamic sampling rate
        if len(magnitude) > 1:
            jerk = np.diff(magnitude) * sampling_rate
            features["acc_jerk_mean"] = np.mean(np.abs(jerk))
            features["acc_jerk_std"] = np.std(jerk)
            features["acc_jerk_max"] = np.max(np.abs(jerk))
            features["acc_jerk_energy"] = np.sum(jerk**2) / len(jerk)
        else:
            features["acc_jerk_mean"] = np.nan
            features["acc_jerk_std"] = np.nan
            features["acc_jerk_max"] = np.nan
            features["acc_jerk_energy"] = np.nan
        
        # Activity classification (basic)
        features["motion_flag"] = 1 if features["acc_magnitude_std"] > 0.5 else 0
        features["is_stationary"] = 1 if features["acc_magnitude_std"] < 0.3 else 0
        features["is_high_activity"] = 1 if features["acc_magnitude_std"] > 1.0 else 0
        
        return features
    
    def extract_temperature_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """Extract temperature features from window data."""
        features = {}
        
        if "skin_temp" in window_df.columns:
            temp = window_df["skin_temp"].dropna().values
            if len(temp) > 0:
                features["temp_mean"] = np.mean(temp)
                features["temp_std"] = np.std(temp) if len(temp) > 1 else 0.0
                features["temp_min"] = np.min(temp)
                features["temp_max"] = np.max(temp)
                features["temp_range"] = np.ptp(temp)
                
                if len(temp) > 1:
                    x = np.arange(len(temp))
                    slope, _, _, _, _ = stats.linregress(x, temp)
                    features["temp_slope"] = slope
                else:
                    features["temp_slope"] = 0.0
                
                features["temp_change"] = temp[-1] - temp[0] if len(temp) > 1 else 0.0
            else:
                for key in ["temp_mean", "temp_std", "temp_min", "temp_max", 
                           "temp_range", "temp_slope", "temp_change"]:
                    features[key] = np.nan
        else:
            for key in ["temp_mean", "temp_std", "temp_min", "temp_max", 
                       "temp_range", "temp_slope", "temp_change"]:
                features[key] = np.nan
        
        return features
    
    def extract_heatflux_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """Extract heat flux features from window data."""
        features = {}
        
        if "heatflux" in window_df.columns:
            hf = window_df["heatflux"].dropna().values
            if len(hf) > 0:
                features["heatflux_mean"] = np.mean(hf)
                features["heatflux_std"] = np.std(hf) if len(hf) > 1 else 0.0
            else:
                features["heatflux_mean"] = np.nan
                features["heatflux_std"] = np.nan
        else:
            features["heatflux_mean"] = np.nan
            features["heatflux_std"] = np.nan
        
        if "cbt" in window_df.columns:
            cbt = window_df["cbt"].dropna().values
            if len(cbt) > 0:
                features["cbt_mean"] = np.mean(cbt)
                features["cbt_change"] = cbt[-1] - cbt[0] if len(cbt) > 1 else 0.0
            else:
                features["cbt_mean"] = np.nan
                features["cbt_change"] = np.nan
        else:
            features["cbt_mean"] = np.nan
            features["cbt_change"] = np.nan
        
        return features
    
    def extract_eda_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """Extract EDA features from window data."""
        features = {}
        
        eda_col = None
        for col in window_df.columns:
            if "eda" in col.lower() or "stress_skin" in col.lower():
                eda_col = col
                break
        
        if eda_col is not None:
            eda = window_df[eda_col].dropna().values
            if len(eda) > 0:
                features["eda_mean"] = np.mean(eda)
                features["eda_median"] = np.median(eda)
                features["eda_std"] = np.std(eda) if len(eda) > 1 else 0.0
                features["eda_min"] = np.min(eda)
                features["eda_max"] = np.max(eda)
                features["eda_range"] = np.ptp(eda)
                
                if len(eda) > 2:
                    threshold = np.mean(eda) + np.std(eda)
                    features["eda_num_peaks"] = np.sum(eda > threshold)
                else:
                    features["eda_num_peaks"] = 0
                
                if len(eda) > 1:
                    x = np.arange(len(eda))
                    slope, _, _, _, _ = stats.linregress(x, eda)
                    features["eda_trend"] = slope
                else:
                    features["eda_trend"] = 0.0
                
                features["eda_change"] = eda[-1] - eda[0] if len(eda) > 1 else 0.0
            else:
                for key in ["eda_mean", "eda_median", "eda_std", "eda_min", "eda_max",
                           "eda_range", "eda_num_peaks", "eda_trend", "eda_change"]:
                    features[key] = np.nan
        else:
            for key in ["eda_mean", "eda_median", "eda_std", "eda_min", "eda_max",
                       "eda_range", "eda_num_peaks", "eda_trend", "eda_change"]:
                features[key] = np.nan
        
        return features
    
    # NOTE: PPG features removed - resampling PPG to 1Hz destroys cardiac waveform structure
    # At 1Hz, you lose heartbeat peaks, HRV, and any meaningful PPG morphology.
    # For PPG-based features, you need native sampling rate (64Hz) and HeartPy for HR/HRV extraction.
    
    def extract_from_window(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all features for emotional stress detection.
        
        Features extracted (~61 total):
        - Accelerometer (41): stress indicators + activity classification  
        - Temperature (7): skin temp stats
        - Heat flux (4): heatflux, cbt
        - EDA (9): electrodermal activity
        
        NOTE: PPG features removed - 1Hz resampling destroys cardiac waveform.
        
        Args:
            window_df: DataFrame with aligned 1Hz data for the window
        
        Returns:
            Dictionary of ~61 features
        """
        features = {}
        
        # Accelerometer features (stress + activity classification, ~41)
        features.update(self.extract_accelerometer_features(window_df))
        
        # Temperature features (7)
        features.update(self.extract_temperature_features(window_df))
        
        # Heat flux features (4)
        features.update(self.extract_heatflux_features(window_df))
        
        # EDA features (9)
        features.update(self.extract_eda_features(window_df))
        
        # PPG features REMOVED - 1Hz resampling destroys cardiac waveform structure
        # For HR/HRV, need native 64Hz sampling rate and HeartPy extraction
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names (excluding PPG)."""
        return STRESS_ACC_FEATURES + [
            "temp_mean", "temp_std", "temp_min", "temp_max", "temp_range",
            "temp_slope", "temp_change",
            "heatflux_mean", "heatflux_std", "cbt_mean", "cbt_change",
            "eda_mean", "eda_median", "eda_std", "eda_min", "eda_max",
            "eda_range", "eda_num_peaks", "eda_trend", "eda_change",
        ]


# Feature names for reference (~61 features, NO PPG)
# PPG removed because 1Hz resampling destroys cardiac waveform structure
FEATURE_NAMES = STRESS_ACC_FEATURES + [
    "temp_mean", "temp_std", "temp_min", "temp_max", "temp_range",
    "temp_slope", "temp_change",
    "heatflux_mean", "heatflux_std", "cbt_mean", "cbt_change",
    "eda_mean", "eda_median", "eda_std", "eda_min", "eda_max",
    "eda_range", "eda_num_peaks", "eda_trend", "eda_change",
]


if __name__ == "__main__":
    print("Testing EMOTIONAL STRESS feature extraction")
    print("="*60)
    print("Includes activity classification to distinguish from exercise")
    print("="*60)
    
    # Create dummy window data (1Hz aligned)
    np.random.seed(42)
    n_samples = 120  # 120 seconds at 1Hz
    
    window_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1S"),
        "acc_x": np.random.normal(0, 0.1, n_samples),
        "acc_y": np.random.normal(0, 0.1, n_samples),
        "acc_z": np.random.normal(-1, 0.1, n_samples),
        "skin_temp": np.random.normal(32, 1, n_samples),
        "heatflux": np.random.normal(20, 5, n_samples),
        "cbt": np.random.normal(37, 0.5, n_samples),
        "eda_stress_skin": np.random.normal(2, 0.5, n_samples),
        # PPG removed - 1Hz resampling destroys cardiac waveform structure
    })
    
    # Calculate sampling rate from data
    sr = calculate_sampling_rate(window_df["timestamp"])
    print(f"\nDetected sampling rate: {sr:.2f} Hz")
    
    extractor = BasicFeatureExtractor()
    features = extractor.extract_from_window(window_df)
    
    print(f"\nTotal features: {len(features)}")
    
    # Count by type
    stress_features = [k for k in features.keys() if k.startswith("acc_") and 
                       "tilt" not in k and "is_" not in k and "activity" not in k]
    activity_features = [k for k in features.keys() if 
                        "tilt" in k or "is_" in k or "activity" in k or 
                        "dominant_freq" in k or "psd_" in k]
    
    print(f"\nFeature breakdown:")
    print(f"  Movement/Stress indicators: {len(stress_features)}")
    print(f"  Activity classification: {len(activity_features)}")
    
    # Show key activity classification features
    print(f"\nActivity classification indicators:")
    print(f"  is_stationary: {features.get('is_stationary', 'N/A')}")
    print(f"  is_walking: {features.get('is_walking', 'N/A')}")
    print(f"  is_high_activity: {features.get('is_high_activity', 'N/A')}")
    print(f"  activity_score: {features.get('activity_score', 'N/A'):.3f}")
    
    # Missing data analysis
    feature_df = pd.DataFrame([features])
    missing_analysis = analyze_missing_data(feature_df, list(features.keys()))
    
    print("\nMissing Data Analysis:")
    for modality, info in missing_analysis.items():
        print(f"  {modality}: {info['n_features']} features, "
              f"{info['missing_rate']*100:.1f}% missing [{info['status']}]")
    
    # Class ratio example (emotional stress vs no stress/physical)
    dummy_labels = np.array([0]*80 + [1]*40)  # More no-stress (includes physical activity)
    class_stats = calculate_class_ratio(dummy_labels)
    print(f"\nClass Distribution (example):")
    print(f"  Emotional Stress: {class_stats['n_emotional_stress']} ({class_stats['pct_emotional_stress']:.1f}%)")
    print(f"  No Stress + Physical: {class_stats['n_no_stress_or_physical']} ({class_stats['pct_no_stress']:.1f}%)")
    print(f"  Ratio: {class_stats['class_ratio']}")
    
    print(f"\n✅ EMOTIONAL stress feature extraction working!")
    print(f"✅ Includes activity features to distinguish from exercise")
