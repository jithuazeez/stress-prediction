"""
Label parser for VitaStress stress prediction.

Extracts stress onset times from annotation files for creating prediction labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


# Stress task events that mark the START of a stressful activity
STRESS_START_EVENTS = [
    "Cognitive: Start",
    "Physical: Start",
    "Public Speaking Start",
]

# Baseline/rest events that mark low-stress periods
BASELINE_START_EVENTS = [
    "Baseline Start (Start of Experiment)",
    "Rest: Start",
]

# Events marking the end of periods
STOP_EVENTS = [
    "Cognitive: Stop",
    "Physical: Stop",
    "Public Speaking Stop",
    "Baseline Stop",
    "Rest: Stop",
]


@dataclass
class StressEvent:
    """Represents a stress task event."""
    timestamp: datetime
    event_type: str  # 'cognitive', 'physical', 'public_speaking'
    event_name: str  # Original event name


@dataclass
class ExperimentPeriod:
    """Represents a period in the experiment."""
    start_time: datetime
    end_time: Optional[datetime]
    period_type: str  # 'baseline', 'stress_cognitive', 'stress_physical', 'stress_public_speaking', 'rest'
    

@dataclass
class SubjectAnnotations:
    """Container for all parsed annotations for a subject."""
    subject_id: str
    stress_onsets: List[StressEvent]
    baseline_periods: List[ExperimentPeriod]
    stress_periods: List[ExperimentPeriod]
    rest_periods: List[ExperimentPeriod]
    experiment_start: Optional[datetime]
    experiment_end: Optional[datetime]


def parse_annotation_file(annotation_path: Path) -> pd.DataFrame:
    """
    Load and parse an annotation CSV file.
    
    Args:
        annotation_path: Path to annotation CSV file
    
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(annotation_path)
    
    # Handle different timestamp column names
    timestamp_col = None
    for col in ['timestamp', 'date']:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {annotation_path}")
    
    df['timestamp'] = pd.to_datetime(df[timestamp_col], format='ISO8601')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def extract_stress_onsets(df: pd.DataFrame) -> List[StressEvent]:
    """
    Extract stress task onset times from annotations.
    
    Args:
        df: Annotations DataFrame
    
    Returns:
        List of StressEvent objects
    """
    stress_onsets = []
    
    for _, row in df.iterrows():
        button_name = str(row.get('Button Name', ''))
        timestamp = row['timestamp']
        
        # Check for stress start events
        for event in STRESS_START_EVENTS:
            if event.lower() in button_name.lower():
                # Determine event type
                if 'cognitive' in button_name.lower():
                    event_type = 'cognitive'
                elif 'physical' in button_name.lower():
                    event_type = 'physical'
                elif 'public speaking' in button_name.lower() or 'speech' in button_name.lower():
                    event_type = 'public_speaking'
                else:
                    event_type = 'unknown'
                
                stress_onsets.append(StressEvent(
                    timestamp=timestamp,
                    event_type=event_type,
                    event_name=button_name
                ))
                break
    
    return stress_onsets


def extract_periods(df: pd.DataFrame) -> Tuple[List[ExperimentPeriod], List[ExperimentPeriod], List[ExperimentPeriod]]:
    """
    Extract baseline, stress, and rest periods from annotations.
    
    Args:
        df: Annotations DataFrame
    
    Returns:
        Tuple of (baseline_periods, stress_periods, rest_periods)
    """
    baseline_periods = []
    stress_periods = []
    rest_periods = []
    
    # Track current period states
    current_baseline_start = None
    current_stress_start = None
    current_stress_type = None
    current_rest_start = None
    
    for _, row in df.iterrows():
        button_name = str(row.get('Button Name', '')).lower()
        timestamp = row['timestamp']
        
        # Baseline handling
        if 'baseline start' in button_name:
            current_baseline_start = timestamp
        elif 'baseline stop' in button_name and current_baseline_start:
            baseline_periods.append(ExperimentPeriod(
                start_time=current_baseline_start,
                end_time=timestamp,
                period_type='baseline'
            ))
            current_baseline_start = None
        
        # Stress handling
        if 'cognitive: start' in button_name:
            current_stress_start = timestamp
            current_stress_type = 'stress_cognitive'
        elif 'physical: start' in button_name:
            current_stress_start = timestamp
            current_stress_type = 'stress_physical'
        elif 'public speaking start' in button_name:
            current_stress_start = timestamp
            current_stress_type = 'stress_public_speaking'
        elif ('cognitive: stop' in button_name or 
              'physical: stop' in button_name or 
              'public speaking stop' in button_name):
            if current_stress_start:
                stress_periods.append(ExperimentPeriod(
                    start_time=current_stress_start,
                    end_time=timestamp,
                    period_type=current_stress_type or 'stress'
                ))
                current_stress_start = None
                current_stress_type = None
        
        # Rest handling
        if 'rest: start' in button_name:
            current_rest_start = timestamp
        elif 'rest: stop' in button_name and current_rest_start:
            rest_periods.append(ExperimentPeriod(
                start_time=current_rest_start,
                end_time=timestamp,
                period_type='rest'
            ))
            current_rest_start = None
    
    return baseline_periods, stress_periods, rest_periods


def get_experiment_bounds(df: pd.DataFrame) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get experiment start and end times from annotations.
    
    Args:
        df: Annotations DataFrame
    
    Returns:
        Tuple of (experiment_start, experiment_end)
    """
    experiment_start = None
    experiment_end = None
    
    for _, row in df.iterrows():
        button_name = str(row.get('Button Name', '')).lower()
        timestamp = row['timestamp']
        
        if 'start of experiment' in button_name or 'baseline start' in button_name:
            if experiment_start is None:
                experiment_start = timestamp
        
        if 'end of experiment' in button_name or 'recovery stop' in button_name:
            experiment_end = timestamp
    
    # Fallback to first and last timestamps
    if experiment_start is None:
        experiment_start = df['timestamp'].min()
    if experiment_end is None:
        experiment_end = df['timestamp'].max()
    
    return experiment_start, experiment_end


def parse_subject_annotations(annotation_path: Path, subject_id: str) -> SubjectAnnotations:
    """
    Parse all annotations for a single subject.
    
    Args:
        annotation_path: Path to annotation CSV file
        subject_id: Subject identifier
    
    Returns:
        SubjectAnnotations object with all parsed data
    """
    df = parse_annotation_file(annotation_path)
    
    stress_onsets = extract_stress_onsets(df)
    baseline_periods, stress_periods, rest_periods = extract_periods(df)
    experiment_start, experiment_end = get_experiment_bounds(df)
    
    return SubjectAnnotations(
        subject_id=subject_id,
        stress_onsets=stress_onsets,
        baseline_periods=baseline_periods,
        stress_periods=stress_periods,
        rest_periods=rest_periods,
        experiment_start=experiment_start,
        experiment_end=experiment_end
    )


def create_window_labels(
    window_end_time: datetime,
    stress_onsets: List[StressEvent],
    horizons_minutes: List[int] = [3, 5, 10]
) -> Dict[str, int]:
    """
    Create prediction labels for a window based on stress onsets.
    
    A window gets label=1 if a stress onset occurs within the prediction horizon
    after the window ends.
    
    Args:
        window_end_time: End time of the feature window
        stress_onsets: List of stress onset events
        horizons_minutes: List of prediction horizons in minutes
    
    Returns:
        Dictionary mapping horizon names to binary labels
        e.g., {'label_3min': 0, 'label_5min': 1, 'label_10min': 1}
    """
    labels = {}
    
    for horizon in horizons_minutes:
        horizon_end = window_end_time + timedelta(minutes=horizon)
        
        # Check if any stress onset falls within [window_end, window_end + horizon]
        label = 0
        for onset in stress_onsets:
            if window_end_time <= onset.timestamp <= horizon_end:
                label = 1
                break
        
        labels[f'label_{horizon}min'] = label
    
    return labels


def get_window_context(
    window_start: datetime,
    window_end: datetime,
    annotations: SubjectAnnotations
) -> str:
    """
    Determine the experimental context of a window.
    
    Args:
        window_start: Window start time
        window_end: Window end time
        annotations: Subject annotations
    
    Returns:
        Context string: 'baseline', 'rest', 'stress_cognitive', 'stress_physical', 
                       'stress_public_speaking', 'pre_stress', 'post_stress', 'unknown'
    """
    window_center = window_start + (window_end - window_start) / 2
    
    # Check if during baseline
    for period in annotations.baseline_periods:
        if period.start_time <= window_center <= (period.end_time or window_center):
            return 'baseline'
    
    # Check if during rest
    for period in annotations.rest_periods:
        if period.start_time <= window_center <= (period.end_time or window_center):
            return 'rest'
    
    # Check if during stress
    for period in annotations.stress_periods:
        if period.start_time <= window_center <= (period.end_time or window_center):
            return period.period_type
    
    # Check if pre-stress (within 10 minutes before stress onset)
    for onset in annotations.stress_onsets:
        if onset.timestamp - timedelta(minutes=10) <= window_center < onset.timestamp:
            return f'pre_{onset.event_type}'
    
    return 'unknown'


class LabelParser:
    """
    Parser for extracting stress labels from VitaStress annotations.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize label parser.
        
        Args:
            data_path: Path to VitaStress data directory
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
    
    def get_subject_folders(self) -> List[str]:
        """Get list of subject folder names."""
        subjects = [d.name for d in self.data_path.iterdir()
                   if d.is_dir() and d.name.startswith('id_')]
        return sorted(subjects)
    
    def parse_subject(self, subject_folder: str) -> Optional[SubjectAnnotations]:
        """
        Parse annotations for a single subject.
        
        Args:
            subject_folder: Subject folder name (e.g., 'id_0a73ef1b-...')
        
        Returns:
            SubjectAnnotations or None if annotation file not found
        """
        subject_path = self.data_path / subject_folder
        subject_id = subject_folder.replace('id_', '')
        
        # Find annotation file
        annotation_files = list(subject_path.glob('*annotation*.csv'))
        
        if not annotation_files:
            print(f"No annotation file found for {subject_folder[:20]}...")
            return None
        
        return parse_subject_annotations(annotation_files[0], subject_id)
    
    def parse_all_subjects(self) -> Dict[str, SubjectAnnotations]:
        """
        Parse annotations for all subjects.
        
        Returns:
            Dictionary mapping subject folder to SubjectAnnotations
        """
        all_annotations = {}
        
        for subject_folder in self.get_subject_folders():
            annotations = self.parse_subject(subject_folder)
            if annotations:
                all_annotations[subject_folder] = annotations
        
        return all_annotations
    
    def get_stress_summary(self) -> pd.DataFrame:
        """
        Get summary of stress events across all subjects.
        
        Returns:
            DataFrame with stress event summary
        """
        all_annotations = self.parse_all_subjects()
        
        rows = []
        for subject_folder, annotations in all_annotations.items():
            for onset in annotations.stress_onsets:
                rows.append({
                    'subject_id': annotations.subject_id[:8],
                    'event_type': onset.event_type,
                    'timestamp': onset.timestamp,
                    'event_name': onset.event_name
                })
        
        return pd.DataFrame(rows)


if __name__ == '__main__':
    # Test the label parser
    data_path = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    
    parser = LabelParser(data_path)
    subjects = parser.get_subject_folders()
    print(f"Found {len(subjects)} subjects")
    
    # Test on first subject
    if subjects:
        annotations = parser.parse_subject(subjects[0])
        if annotations:
            print(f"\nSubject: {annotations.subject_id[:8]}...")
            print(f"Experiment: {annotations.experiment_start} to {annotations.experiment_end}")
            print(f"Stress onsets: {len(annotations.stress_onsets)}")
            for onset in annotations.stress_onsets:
                print(f"  - {onset.event_type}: {onset.timestamp}")
            print(f"Baseline periods: {len(annotations.baseline_periods)}")
            print(f"Stress periods: {len(annotations.stress_periods)}")
            print(f"Rest periods: {len(annotations.rest_periods)}")
    
    # Get overall summary
    print("\n" + "="*50)
    print("Stress events summary:")
    summary = parser.get_stress_summary()
    print(summary.to_string(index=False))

