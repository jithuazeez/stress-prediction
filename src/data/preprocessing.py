"""
Data preprocessing and windowing for VitaStress stress prediction.

Implements sliding window extraction with configurable sizes and overlap.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class SignalResampler:
    """
    Resample physiological signals to unified 1Hz timeline.
    """
    
    @staticmethod
    def resample_to_1hz(data: pd.DataFrame, value_col: str, 
                       method: str = 'linear') -> pd.DataFrame:
        """
        Resample signal to 1Hz.
        
        Args:
            data: DataFrame with 'timestamp' column
            value_col: Name of value column to resample
            method: Interpolation method ('linear', 'nearest', 'ffill', 'bfill')
        
        Returns:
            Resampled DataFrame at 1Hz
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")
        
        # Set timestamp as index
        data = data.set_index('timestamp').sort_index()
        
        # Create 1Hz timeline
        start_time = data.index.min()
        end_time = data.index.max()
        timeline = pd.date_range(start=start_time, end=end_time, freq='1S')
        
        # Resample to 1Hz
        resampled = data[value_col].reindex(timeline)
        
        # Interpolate missing values
        if method == 'linear':
            resampled = resampled.interpolate(method='linear', limit=5)
        elif method == 'nearest':
            resampled = resampled.interpolate(method='nearest', limit=5)
        elif method == 'ffill':
            resampled = resampled.fillna(method='ffill', limit=5)
        elif method == 'bfill':
            resampled = resampled.fillna(method='bfill', limit=5)
        
        result = pd.DataFrame({
            'timestamp': timeline,
            value_col: resampled.values
        })
        
        return result


class WindowExtractor:
    """
    Extract sliding windows from physiological signals with labels.
    """
    
    def __init__(self, window_size_seconds: int = 60, 
                 overlap_seconds: int = 30,
                 prediction_horizon_minutes: int = 5):
        """
        Initialize window extractor.
        
        Args:
            window_size_seconds: Size of each window in seconds
            overlap_seconds: Overlap between consecutive windows
            prediction_horizon_minutes: How many minutes ahead to predict stress
        """
        self.window_size = window_size_seconds
        self.overlap = overlap_seconds
        self.stride = window_size_seconds - overlap_seconds
        self.prediction_horizon = timedelta(minutes=prediction_horizon_minutes)
        
    def create_windows(self, data: pd.DataFrame, 
                      stress_periods: List[Tuple], 
                      baseline_periods: List[Tuple]) -> List[Dict]:
        """
        Create sliding windows with labels.
        
        Args:
            data: Resampled data at 1Hz with 'timestamp' column
            stress_periods: List of (start, end) tuples for stress periods
            baseline_periods: List of (start, end) tuples for baseline periods
        
        Returns:
            List of window dictionaries with data and labels
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")
        
        windows = []
        
        # Generate window start times
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max() - timedelta(seconds=self.window_size)
        
        current_time = start_time
        while current_time <= end_time:
            window_end = current_time + timedelta(seconds=self.window_size)
            
            # Extract window data
            window_mask = (data['timestamp'] >= current_time) & \
                         (data['timestamp'] < window_end)
            window_data = data[window_mask].copy()
            
            # Skip if window doesn't have enough data
            if len(window_data) < self.window_size * 0.8:  # At least 80% coverage
                current_time += timedelta(seconds=self.stride)
                continue
            
            # Determine label based on prediction horizon
            prediction_time = window_end + self.prediction_horizon
            label = self._get_label(prediction_time, stress_periods, baseline_periods)
            
            # Only include windows with clear labels
            if label is not None:
                windows.append({
                    'start_time': current_time,
                    'end_time': window_end,
                    'prediction_time': prediction_time,
                    'data': window_data,
                    'label': label,
                    'window_size': self.window_size
                })
            
            current_time += timedelta(seconds=self.stride)
        
        return windows
    
    def _get_label(self, prediction_time: pd.Timestamp,
                  stress_periods: List[Tuple],
                  baseline_periods: List[Tuple]) -> Optional[int]:
        """
        Determine label based on prediction time.
        
        Args:
            prediction_time: Time point to predict
            stress_periods: List of stress (start, end) tuples
            baseline_periods: List of baseline (start, end) tuples
        
        Returns:
            1 for stress, 0 for no stress, None if ambiguous
        """
        # Check if prediction time falls within stress period
        for start, end in stress_periods:
            if start <= prediction_time <= end:
                return 1
        
        # Check if prediction time falls within baseline period  
        for start, end in baseline_periods:
            if start <= prediction_time <= end:
                return 0
        
        # Ambiguous - not in any labeled period
        return None
    
    def extract_window_features_matrix(self, window: Dict) -> np.ndarray:
        """
        Extract fixed-size feature matrix from window data.
        
        For RNN/Transformer: returns (window_size, num_features) matrix
        For classical ML: this will be flattened or aggregated
        
        Args:
            window: Window dictionary from create_windows()
        
        Returns:
            Feature matrix of shape (window_size, num_features)
        """
        data = window['data']
        
        # Remove timestamp column
        feature_cols = [col for col in data.columns if col != 'timestamp']
        
        # Extract feature matrix
        feature_matrix = data[feature_cols].values
        
        # Ensure fixed size by padding or truncating
        expected_size = self.window_size
        current_size = len(feature_matrix)
        
        if current_size < expected_size:
            # Pad with last value
            padding = np.repeat(feature_matrix[-1:], expected_size - current_size, axis=0)
            feature_matrix = np.vstack([feature_matrix, padding])
        elif current_size > expected_size:
            # Truncate
            feature_matrix = feature_matrix[:expected_size]
        
        return feature_matrix


class LabelCreator:
    """
    Create stress labels with prediction horizon.
    """
    
    @staticmethod
    def create_labels_with_horizon(stress_periods: List[Tuple],
                                   baseline_periods: List[Tuple],
                                   prediction_horizon_minutes: int = 5) -> pd.DataFrame:
        """
        Create labeled timeline with prediction horizon.
        
        Windows that occur N minutes before stress onset are labeled as positive.
        
        Args:
            stress_periods: List of (start, end) stress period tuples
            baseline_periods: List of (start, end) baseline period tuples  
            prediction_horizon_minutes: Minutes before stress to label as positive
        
        Returns:
            DataFrame with timestamp and label columns
        """
        horizon = timedelta(minutes=prediction_horizon_minutes)
        
        # Create labeled periods
        labeled_periods = []
        
        # Positive samples: periods before stress onset
        for start, end in stress_periods:
            # Label period before stress start as positive
            pre_stress_start = start - horizon
            labeled_periods.append({
                'start': pre_stress_start,
                'end': start,
                'label': 1,
                'period_type': 'pre_stress'
            })
            # Actual stress period
            labeled_periods.append({
                'start': start,
                'end': end,
                'label': 1,
                'period_type': 'stress'
            })
        
        # Negative samples: baseline periods
        for start, end in baseline_periods:
            labeled_periods.append({
                'start': start,
                'end': end,
                'label': 0,
                'period_type': 'baseline'
            })
        
        return pd.DataFrame(labeled_periods)


def prepare_subject_windows(signals: Dict[str, pd.DataFrame],
                            stress_periods: List[Tuple],
                            baseline_periods: List[Tuple],
                            window_size: int = 60,
                            overlap: int = 30,
                            prediction_horizon: int = 5) -> List[Dict]:
    """
    Prepare windowed data for a single subject.
    
    Args:
        signals: Dictionary of signal DataFrames
        stress_periods: List of stress (start, end) tuples
        baseline_periods: List of baseline (start, end) tuples
        window_size: Window size in seconds
        overlap: Overlap in seconds
        prediction_horizon: Prediction horizon in minutes
    
    Returns:
        List of window dictionaries
    """
    # Combine all signals into unified 1Hz timeline
    # Start with RR intervals as base if available
    if 'rr_interval' in signals and len(signals['rr_interval']) > 0:
        base_signal = signals['rr_interval'].copy()
        if 'timestamp' in base_signal.columns:
            min_time = base_signal['timestamp'].min()
            max_time = base_signal['timestamp'].max()
        else:
            print("Warning: No timestamp in RR data")
            return []
    else:
        # Use first available signal
        for signal_type, signal_data in signals.items():
            if len(signal_data) > 0 and 'timestamp' in signal_data.columns:
                base_signal = signal_data.copy()
                min_time = base_signal['timestamp'].min()
                max_time = base_signal['timestamp'].max()
                break
        else:
            print("Warning: No valid signals found")
            return []
    
    # Create 1Hz timeline
    timeline = pd.date_range(start=min_time, end=max_time, freq='1S')
    unified_data = pd.DataFrame({'timestamp': timeline})
    
    # Resample each signal to 1Hz and merge
    resampler = SignalResampler()
    
    for signal_type, signal_data in signals.items():
        if len(signal_data) == 0 or 'timestamp' not in signal_data.columns:
            continue
        
        # Determine value column(s)
        value_cols = [col for col in signal_data.columns 
                     if col not in ['timestamp', 'date', 'metric_id', 
                                   'chunk_index', 'quality', 'body_pose']]
        
        for value_col in value_cols[:3]:  # Limit to first 3 columns per signal
            try:
                resampled = resampler.resample_to_1hz(signal_data, value_col, method='linear')
                col_name = f"{signal_type}_{value_col}"
                unified_data = unified_data.merge(resampled.rename(columns={value_col: col_name}),
                                                 on='timestamp', how='left')
            except Exception as e:
                print(f"Warning: Could not resample {signal_type}.{value_col}: {e}")
                continue
    
    # Create windows
    extractor = WindowExtractor(window_size, overlap, prediction_horizon)
    windows = extractor.create_windows(unified_data, stress_periods, baseline_periods)
    
    return windows


if __name__ == '__main__':
    # Test windowing
    from vitastress_loader import VitaStressLoader
    
    data_path = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    loader = VitaStressLoader(data_path)
    
    if loader.subjects:
        subject_id = loader.subjects[0]
        print(f"Testing windowing with subject: {subject_id[:20]}...")
        
        signals = loader.load_all_signals(subject_id)
        stress_periods, baseline_periods = loader.get_stress_periods(subject_id)
        
        print(f"Stress periods: {len(stress_periods)}")
        print(f"Baseline periods: {len(baseline_periods)}")
        
        windows = prepare_subject_windows(signals, stress_periods, baseline_periods,
                                         window_size=60, overlap=30, prediction_horizon=5)
        
        print(f"\nGenerated {len(windows)} windows")
        if windows:
            print(f"First window:")
            print(f"  Start: {windows[0]['start_time']}")
            print(f"  End: {windows[0]['end_time']}")
            print(f"  Label: {windows[0]['label']}")
            print(f"  Data shape: {windows[0]['data'].shape}")








