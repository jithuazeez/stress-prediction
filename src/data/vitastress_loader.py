"""
VitaStress dataset loader.

Loads physiological signals and annotations from VitaStress dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class VitaStressLoader:
    """
    Load and parse VitaStress dataset files.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize VitaStress data loader.
        
        Args:
            data_path: Path to VitaStress data directory
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        self.subjects = self._get_subjects()
        
    def _get_subjects(self) -> List[str]:
        """Get list of subject IDs."""
        subjects = [d.name for d in self.data_path.iterdir() 
                   if d.is_dir() and d.name.startswith('id_')]
        return sorted(subjects)
    
    def load_signal(self, subject_id: str, signal_type: str) -> Optional[pd.DataFrame]:
        """
        Load a specific signal type for a subject.
        
        Args:
            subject_id: Subject ID (e.g., 'id_0a73ef1b-...')
            signal_type: Signal type ('rr_interval', 'ppg2_green_6', 'acc', 
                                     'temperature', 'emography', 'activity')
        
        Returns:
            DataFrame with signal data or None if not found
        """
        subject_path = self.data_path / subject_id
        
        # Find file matching signal type
        files = list(subject_path.glob(f'*{signal_type}*.csv'))
        
        if not files:
            print(f"⚠️  No {signal_type} file found for {subject_id[:15]}...")
            return None
        
        try:
            df = pd.read_csv(files[0])
            
            # Parse timestamps - handle different column names
            timestamp_col = None
            for col in ['date', 'timestamp']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], format='ISO8601')
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading {signal_type} for {subject_id[:15]}...: {e}")
            return None
    
    def load_annotations(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        Load experimental annotations for a subject.
        
        Args:
            subject_id: Subject ID
        
        Returns:
            DataFrame with annotations or None if not found
        """
        return self.load_signal(subject_id, 'annotation')
    
    def load_all_signals(self, subject_id: str) -> Dict[str, pd.DataFrame]:
        """
        Load all available signals for a subject.
        
        Args:
            subject_id: Subject ID
        
        Returns:
            Dictionary mapping signal type to DataFrame
        """
        signal_types = [
            'rr_interval',
            'ppg2_green_6',
            'acc',
            'temperature',
            'emography',
            'activity'
        ]
        
        signals = {}
        for signal_type in signal_types:
            data = self.load_signal(subject_id, signal_type)
            if data is not None:
                signals[signal_type] = data
        
        return signals
    
    def parse_stress_labels(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Parse stress condition labels from annotations.
        
        Args:
            annotations: Annotations DataFrame
        
        Returns:
            DataFrame with parsed labels and timestamps
        """
        if annotations is None or len(annotations) == 0:
            return pd.DataFrame(columns=['timestamp', 'event', 'label'])
        
        # Parse timestamps
        annotations['timestamp'] = pd.to_datetime(annotations['timestamp'], 
                                                  format='ISO8601', errors='coerce')
        
        # Extract button names
        if 'Button Name' not in annotations.columns:
            return pd.DataFrame(columns=['timestamp', 'event', 'label'])
        
        events = []
        for _, row in annotations.iterrows():
            button = str(row['Button Name']).lower()
            timestamp = row['timestamp']
            
            # Determine label based on button name
            if 'baseline' in button or 'rest' in button:
                if 'start' in button:
                    label = 'baseline_start'
                elif 'stop' in button:
                    label = 'baseline_stop'
                else:
                    label = 'baseline'
            elif any(stress_type in button for stress_type in ['cognitive', 'social', 'physical']):
                if 'start' in button or 'introduction' in button:
                    label = 'stress_start'
                elif 'stop' in button:
                    label = 'stress_stop'
                else:
                    label = 'stress'
            else:
                label = 'other'
            
            events.append({
                'timestamp': timestamp,
                'event': button,
                'label': label
            })
        
        return pd.DataFrame(events)
    
    def get_stress_periods(self, subject_id: str) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Extract stress and baseline periods from annotations.
        
        Args:
            subject_id: Subject ID
        
        Returns:
            Tuple of (stress_periods, baseline_periods) where each is a list of 
            (start_time, end_time) tuples
        """
        annotations = self.load_annotations(subject_id)
        if annotations is None:
            return [], []
        
        labels = self.parse_stress_labels(annotations)
        
        stress_periods = []
        baseline_periods = []
        
        # Track ongoing periods
        current_stress_start = None
        current_baseline_start = None
        
        for _, row in labels.iterrows():
            label = row['label']
            timestamp = row['timestamp']
            
            if label == 'stress_start':
                current_stress_start = timestamp
            elif label == 'stress_stop' and current_stress_start:
                stress_periods.append((current_stress_start, timestamp))
                current_stress_start = None
            
            elif label == 'baseline_start':
                current_baseline_start = timestamp
            elif label == 'baseline_stop' and current_baseline_start:
                baseline_periods.append((current_baseline_start, timestamp))
                current_baseline_start = None
        
        return stress_periods, baseline_periods
    
    def get_subject_data_summary(self, subject_id: str) -> Dict:
        """
        Get summary statistics for a subject's data.
        
        Args:
            subject_id: Subject ID
        
        Returns:
            Dictionary with summary statistics
        """
        signals = self.load_all_signals(subject_id)
        stress_periods, baseline_periods = self.get_stress_periods(subject_id)
        
        summary = {
            'subject_id': subject_id,
            'available_signals': list(signals.keys()),
            'num_stress_periods': len(stress_periods),
            'num_baseline_periods': len(baseline_periods),
        }
        
        # Add signal-specific info
        for signal_type, data in signals.items():
            if 'timestamp' in data.columns:
                duration = (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600
                summary[f'{signal_type}_duration_hours'] = duration
                summary[f'{signal_type}_samples'] = len(data)
        
        return summary


def load_vitastress_subject(data_path: str, subject_id: str) -> Dict:
    """
    Convenience function to load all data for a subject.
    
    Args:
        data_path: Path to VitaStress data directory
        subject_id: Subject ID
    
    Returns:
        Dictionary containing all signals and metadata
    """
    loader = VitaStressLoader(data_path)
    
    signals = loader.load_all_signals(subject_id)
    stress_periods, baseline_periods = loader.get_stress_periods(subject_id)
    
    return {
        'subject_id': subject_id,
        'signals': signals,
        'stress_periods': stress_periods,
        'baseline_periods': baseline_periods
    }


if __name__ == '__main__':
    # Test the loader
    data_path = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    
    loader = VitaStressLoader(data_path)
    print(f"Found {len(loader.subjects)} subjects")
    
    # Test loading first subject
    if loader.subjects:
        subject_id = loader.subjects[0]
        print(f"\nTesting with subject: {subject_id[:20]}...")
        
        summary = loader.get_subject_data_summary(subject_id)
        print("\nSubject summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")








