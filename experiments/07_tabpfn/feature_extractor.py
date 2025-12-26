"""
Master feature extraction pipeline for VitaStress stress prediction.

Extracts 39 features per window:
- 12 HR/HRV features (from PPG via HeartPy)
- 12 Temperature/HeatFlux features 
- 15 Accelerometer features

Updated to use HeartPy for HRV extraction from PPG (ppg2_green_6.csv).
EDA dropped due to insufficient resolution (~1 sample/minute).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import HRV extractor (HeartPy-based)
from hrv_extractor import HRVExtractor, HRV_FEATURE_NAMES


# All feature names for the 39 features
FEATURE_NAMES = [
    # HRV features (12) - from HeartPy
    'hr_bpm', 'hr_std', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50', 'hrv_pnn20',
    'hrv_sdsd', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio', 'hrv_mean_rr', 'breathing_rate',
    # Temperature/HeatFlux features (12)
    'temp_skin_mean', 'temp_skin_std', 'temp_skin_min', 'temp_skin_max',
    'temp_skin_range', 'temp_skin_slope', 'heatflux_mean', 'heatflux_std',
    'heatflux_range', 'cbt_mean', 'cbt_change', 'pulse_rate_hf',
    # Accelerometer features (15)
    'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_min',
    'acc_magnitude_max', 'acc_magnitude_range', 'acc_sma', 'acc_energy',
    'acc_zcr', 'acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std',
    'acc_z_mean', 'acc_z_std', 'motion_intensity'
]


def extract_temperature_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract temperature and heat flux features from aligned window data.
    
    Expected columns: skin_temp, heatflux, cbt, pulse_rate
    
    Args:
        df: DataFrame with temperature data for the window
    
    Returns:
        Dictionary of 12 temperature features
    """
    features = {}
    
    # Skin temperature features
    if 'skin_temp' in df.columns and not df['skin_temp'].isna().all():
        temp = df['skin_temp'].dropna().values
        if len(temp) > 0:
            features['temp_skin_mean'] = np.mean(temp)
            features['temp_skin_std'] = np.std(temp) if len(temp) > 1 else 0.0
            features['temp_skin_min'] = np.min(temp)
            features['temp_skin_max'] = np.max(temp)
            features['temp_skin_range'] = np.max(temp) - np.min(temp)
            
            # Linear slope (trend)
            if len(temp) > 1:
                x = np.arange(len(temp))
                slope, _, _, _, _ = stats.linregress(x, temp)
                features['temp_skin_slope'] = slope
            else:
                features['temp_skin_slope'] = 0.0
        else:
            for key in ['temp_skin_mean', 'temp_skin_std', 'temp_skin_min', 
                        'temp_skin_max', 'temp_skin_range', 'temp_skin_slope']:
                features[key] = np.nan
    else:
        for key in ['temp_skin_mean', 'temp_skin_std', 'temp_skin_min', 
                    'temp_skin_max', 'temp_skin_range', 'temp_skin_slope']:
            features[key] = np.nan
    
    # Heat flux features
    if 'heatflux' in df.columns and not df['heatflux'].isna().all():
        hf = df['heatflux'].dropna().values
        if len(hf) > 0:
            features['heatflux_mean'] = np.mean(hf)
            features['heatflux_std'] = np.std(hf) if len(hf) > 1 else 0.0
            features['heatflux_range'] = np.max(hf) - np.min(hf)
        else:
            features['heatflux_mean'] = np.nan
            features['heatflux_std'] = np.nan
            features['heatflux_range'] = np.nan
    else:
        features['heatflux_mean'] = np.nan
        features['heatflux_std'] = np.nan
        features['heatflux_range'] = np.nan
    
    # Core body temperature features
    if 'cbt' in df.columns and not df['cbt'].isna().all():
        cbt = df['cbt'].dropna().values
        if len(cbt) > 0:
            features['cbt_mean'] = np.mean(cbt)
            features['cbt_change'] = cbt[-1] - cbt[0] if len(cbt) > 1 else 0.0
        else:
            features['cbt_mean'] = np.nan
            features['cbt_change'] = np.nan
    else:
        features['cbt_mean'] = np.nan
        features['cbt_change'] = np.nan
    
    # Pulse rate from heat flux sensor (backup HR)
    if 'pulse_rate' in df.columns and not df['pulse_rate'].isna().all():
        pr = df['pulse_rate'].dropna().values
        if len(pr) > 0:
            features['pulse_rate_hf'] = np.mean(pr)
        else:
            features['pulse_rate_hf'] = np.nan
    else:
        features['pulse_rate_hf'] = np.nan
    
    return features


def extract_accelerometer_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract accelerometer features from aligned window data.
    
    Expected columns: acc_raw_x, acc_raw_y, acc_raw_z (or acc_x, acc_y, acc_z)
    
    Args:
        df: DataFrame with accelerometer data for the window
    
    Returns:
        Dictionary of 15 accelerometer features
    """
    features = {}
    
    # Find accelerometer columns
    x_col = y_col = z_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'acc' in col_lower:
            if 'x' in col_lower:
                x_col = col
            elif 'y' in col_lower:
                y_col = col
            elif 'z' in col_lower:
                z_col = col
    
    if x_col is None or y_col is None or z_col is None:
        # Return NaN for all features
        return {
            'acc_magnitude_mean': np.nan, 'acc_magnitude_std': np.nan,
            'acc_magnitude_min': np.nan, 'acc_magnitude_max': np.nan,
            'acc_magnitude_range': np.nan, 'acc_sma': np.nan, 'acc_energy': np.nan,
            'acc_zcr': np.nan, 'acc_x_mean': np.nan, 'acc_x_std': np.nan,
            'acc_y_mean': np.nan, 'acc_y_std': np.nan, 'acc_z_mean': np.nan,
            'acc_z_std': np.nan, 'motion_intensity': np.nan
        }
    
    # Get values
    acc_x = df[x_col].dropna().values
    acc_y = df[y_col].dropna().values
    acc_z = df[z_col].dropna().values
    
    # Ensure same length
    min_len = min(len(acc_x), len(acc_y), len(acc_z))
    if min_len < 2:
        return {
            'acc_magnitude_mean': np.nan, 'acc_magnitude_std': np.nan,
            'acc_magnitude_min': np.nan, 'acc_magnitude_max': np.nan,
            'acc_magnitude_range': np.nan, 'acc_sma': np.nan, 'acc_energy': np.nan,
            'acc_zcr': np.nan, 'acc_x_mean': np.nan, 'acc_x_std': np.nan,
            'acc_y_mean': np.nan, 'acc_y_std': np.nan, 'acc_z_mean': np.nan,
            'acc_z_std': np.nan, 'motion_intensity': np.nan
        }
    
    acc_x = acc_x[:min_len]
    acc_y = acc_y[:min_len]
    acc_z = acc_z[:min_len]
    
    # Calculate magnitude
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Magnitude features
    features['acc_magnitude_mean'] = np.mean(magnitude)
    features['acc_magnitude_std'] = np.std(magnitude)
    features['acc_magnitude_min'] = np.min(magnitude)
    features['acc_magnitude_max'] = np.max(magnitude)
    features['acc_magnitude_range'] = np.max(magnitude) - np.min(magnitude)
    
    # Signal Magnitude Area (SMA) - total activity
    features['acc_sma'] = (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) / len(acc_x)
    
    # Energy (sum of squared magnitudes)
    features['acc_energy'] = np.sum(magnitude**2) / len(magnitude)
    
    # Zero Crossing Rate (movement frequency indicator)
    magnitude_centered = magnitude - np.mean(magnitude)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(magnitude_centered))) > 0)
    features['acc_zcr'] = zero_crossings / len(magnitude_centered) if len(magnitude_centered) > 1 else 0.0
    
    # Per-axis features
    features['acc_x_mean'] = np.mean(acc_x)
    features['acc_x_std'] = np.std(acc_x)
    features['acc_y_mean'] = np.mean(acc_y)
    features['acc_y_std'] = np.std(acc_y)
    features['acc_z_mean'] = np.mean(acc_z)
    features['acc_z_std'] = np.std(acc_z)
    
    # Motion intensity (categorical)
    # Based on standard deviation of magnitude
    mag_std = features['acc_magnitude_std']
    if mag_std < 0.1:
        features['motion_intensity'] = 0  # Stationary
    elif mag_std < 0.5:
        features['motion_intensity'] = 1  # Low motion
    elif mag_std < 1.0:
        features['motion_intensity'] = 2  # Moderate motion
    else:
        features['motion_intensity'] = 3  # High motion
    
    return features


class MasterFeatureExtractor:
    """
    Extract 39 stress-relevant features from VitaStress multimodal data.
    
    Feature breakdown:
    - HR/HRV: 12 features (from PPG via HeartPy)
    - Temperature/HeatFlux: 12 features
    - Accelerometer: 15 features
    
    Total: 39 features per window
    """
    
    def __init__(self, ppg_sample_rate: float = 64.0):
        """
        Initialize feature extractor.
        
        Args:
            ppg_sample_rate: Expected PPG sample rate in Hz
        """
        self.hrv_extractor = HRVExtractor(sample_rate=ppg_sample_rate)
        self.ppg_sample_rate = ppg_sample_rate
        self.feature_names = FEATURE_NAMES.copy()
    
    def extract_from_window(
        self,
        ppg_values: Optional[np.ndarray],
        aligned_df: Optional[pd.DataFrame],
        ppg_quality: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract all 39 features from a window.
        
        Args:
            ppg_values: Raw PPG values for the window (at ~64 Hz)
            aligned_df: 1Hz aligned DataFrame with temperature and accelerometer data
            ppg_quality: Optional quality flags for PPG (True = good)
        
        Returns:
            Dictionary of 39 features
        """
        features = {}
        
        # Extract HRV features from PPG
        if ppg_values is not None and len(ppg_values) > 0:
            hrv_features = self.hrv_extractor.extract(ppg_values, ppg_quality)
            features.update(hrv_features)
        else:
            # Fill HRV features with NaN
            for name in HRV_FEATURE_NAMES:
                features[name] = np.nan
        
        # Extract temperature features
        if aligned_df is not None and len(aligned_df) > 0:
            temp_features = extract_temperature_features(aligned_df)
            features.update(temp_features)
        else:
            for name in ['temp_skin_mean', 'temp_skin_std', 'temp_skin_min', 
                        'temp_skin_max', 'temp_skin_range', 'temp_skin_slope',
                        'heatflux_mean', 'heatflux_std', 'heatflux_range',
                        'cbt_mean', 'cbt_change', 'pulse_rate_hf']:
                features[name] = np.nan
        
        # Extract accelerometer features
        if aligned_df is not None and len(aligned_df) > 0:
            acc_features = extract_accelerometer_features(aligned_df)
            features.update(acc_features)
        else:
            for name in ['acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_min',
                        'acc_magnitude_max', 'acc_magnitude_range', 'acc_sma', 'acc_energy',
                        'acc_zcr', 'acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std',
                        'acc_z_mean', 'acc_z_std', 'motion_intensity']:
                features[name] = np.nan
        
        return features
    
    def extract_from_window_data(
        self,
        ppg_df: Optional[pd.DataFrame],
        aligned_df: Optional[pd.DataFrame],
        window_start,
        window_end
    ) -> Dict[str, float]:
        """
        Extract features from DataFrames for a specific time window.
        
        Args:
            ppg_df: Raw PPG DataFrame with 'timestamp', 'value', 'quality' columns
            aligned_df: 1Hz aligned DataFrame with timestamp column
            window_start: Window start time
            window_end: Window end time
        
        Returns:
            Dictionary of 39 features
        """
        # Extract PPG values for window
        ppg_values = None
        ppg_quality = None
        
        if ppg_df is not None and len(ppg_df) > 0:
            mask = (ppg_df['timestamp'] >= window_start) & (ppg_df['timestamp'] < window_end)
            window_ppg = ppg_df.loc[mask]
            if len(window_ppg) > 0:
                ppg_values = window_ppg['value'].values.astype(float)
                if 'quality' in window_ppg.columns:
                    ppg_quality = window_ppg['quality'].values >= 3
        
        # Extract aligned data for window
        window_aligned = None
        if aligned_df is not None and 'timestamp' in aligned_df.columns:
            mask = (aligned_df['timestamp'] >= window_start) & (aligned_df['timestamp'] < window_end)
            window_aligned = aligned_df.loc[mask].copy()
        
        return self.extract_from_window(ppg_values, window_aligned, ppg_quality)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names (39 features)."""
        return self.feature_names.copy()
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by modality."""
        return {
            'hrv': [
                'hr_bpm', 'hr_std', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50', 'hrv_pnn20',
                'hrv_sdsd', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio', 'hrv_mean_rr', 'breathing_rate'
            ],
            'temperature': [
                'temp_skin_mean', 'temp_skin_std', 'temp_skin_min', 'temp_skin_max',
                'temp_skin_range', 'temp_skin_slope', 'heatflux_mean', 'heatflux_std',
                'heatflux_range', 'cbt_mean', 'cbt_change', 'pulse_rate_hf'
            ],
            'accelerometer': [
                'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_min',
                'acc_magnitude_max', 'acc_magnitude_range', 'acc_sma', 'acc_energy',
                'acc_zcr', 'acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std',
                'acc_z_mean', 'acc_z_std', 'motion_intensity'
            ]
        }


if __name__ == '__main__':
    # Test the feature extractor
    print("Testing master feature extractor...")
    
    # Create dummy data
    np.random.seed(42)
    
    # Dummy aligned data (1 Hz, 120 seconds)
    aligned_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=120, freq='1S'),
        'skin_temp': np.random.normal(32, 1, 120),
        'heatflux': np.random.normal(20, 5, 120),
        'cbt': np.random.normal(37, 0.5, 120),
        'pulse_rate': np.random.normal(70, 10, 120),
        'acc_raw_x': np.random.normal(0, 0.1, 120),
        'acc_raw_y': np.random.normal(0, 0.1, 120),
        'acc_raw_z': np.random.normal(-1, 0.1, 120),  # Gravity
    })
    
    # Dummy PPG data (64 Hz, 120 seconds)
    ppg_values = np.random.normal(30000, 1000, int(64 * 120))
    
    # Initialize extractor
    extractor = MasterFeatureExtractor(ppg_sample_rate=64.0)
    
    # Extract features
    features = extractor.extract_from_window(ppg_values, aligned_df)
    
    print(f"\nExtracted {len(features)} features")
    
    # Show features by group
    groups = extractor.get_feature_groups()
    for group_name, group_features in groups.items():
        print(f"\n{group_name.upper()} features ({len(group_features)}):")
        for feat in group_features[:5]:  # Show first 5 of each group
            value = features.get(feat, np.nan)
            if np.isnan(value) if isinstance(value, float) else False:
                print(f"  {feat}: NaN")
            else:
                print(f"  {feat}: {value:.3f}")
        if len(group_features) > 5:
            print(f"  ... and {len(group_features) - 5} more")
    
    # Verify feature count
    all_features = extractor.get_feature_names()
    print(f"\n✅ Total features: {len(all_features)} (expected: 39)")
