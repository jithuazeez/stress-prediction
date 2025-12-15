"""
Window creator for VitaStress stress prediction.

Creates non-overlapping time windows with prediction labels for multiple horizons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Window:
    """Represents a single analysis window."""
    window_id: int
    start_time: datetime
    end_time: datetime
    center_time: datetime
    duration_sec: int
    
    # Prediction labels for different horizons
    label_3min: int  # 1 if stress within 3 min of window end
    label_5min: int  # 1 if stress within 5 min of window end
    label_10min: int  # 1 if stress within 10 min of window end
    
    # Context information
    context: str  # 'baseline', 'rest', 'pre_stress', 'during_stress', 'unknown'
    subject_id: str


@dataclass 
class WindowData:
    """Container for window with associated data slices."""
    window: Window
    aligned_df: Optional[pd.DataFrame]  # 1Hz aligned data for this window
    ppg_values: Optional[np.ndarray]  # Raw PPG values for this window


def create_windows(
    start_time: datetime,
    end_time: datetime,
    window_size_sec: int = 120,
    overlap_ratio: float = 0.0
) -> List[Tuple[datetime, datetime]]:
    """
    Create list of window time boundaries.
    
    Args:
        start_time: Start of the recording
        end_time: End of the recording
        window_size_sec: Window size in seconds
        overlap_ratio: Overlap between windows (0.0 = no overlap, 0.5 = 50% overlap)
    
    Returns:
        List of (window_start, window_end) tuples
    """
    windows = []
    window_delta = timedelta(seconds=window_size_sec)
    step_delta = timedelta(seconds=int(window_size_sec * (1 - overlap_ratio)))
    
    current_start = start_time
    
    while current_start + window_delta <= end_time:
        window_end = current_start + window_delta
        windows.append((current_start, window_end))
        current_start = current_start + step_delta
    
    return windows


def assign_window_labels(
    window_end: datetime,
    stress_onsets: List[datetime],
    horizons_minutes: List[int] = [3, 5, 10]
) -> Dict[str, int]:
    """
    Assign prediction labels to a window based on stress onsets.
    
    A window gets label=1 for a given horizon if any stress onset falls within
    [window_end, window_end + horizon].
    
    Args:
        window_end: End time of the window
        stress_onsets: List of stress onset timestamps
        horizons_minutes: List of prediction horizons in minutes
    
    Returns:
        Dictionary mapping label names to binary values
    """
    labels = {}
    
    for horizon in horizons_minutes:
        horizon_end = window_end + timedelta(minutes=horizon)
        
        label = 0
        for onset in stress_onsets:
            if window_end <= onset <= horizon_end:
                label = 1
                break
        
        labels[f'label_{horizon}min'] = label
    
    return labels


def get_window_context(
    window_start: datetime,
    window_end: datetime,
    stress_onsets: List[datetime],
    stress_periods: List[Tuple[datetime, datetime]],
    baseline_periods: List[Tuple[datetime, datetime]],
    rest_periods: List[Tuple[datetime, datetime]]
) -> str:
    """
    Determine the experimental context of a window.
    
    Args:
        window_start: Window start time
        window_end: Window end time
        stress_onsets: List of stress onset times
        stress_periods: List of (start, end) tuples for stress periods
        baseline_periods: List of (start, end) tuples for baseline periods
        rest_periods: List of (start, end) tuples for rest periods
    
    Returns:
        Context string describing the window's position in the experiment
    """
    window_center = window_start + (window_end - window_start) / 2
    
    # Check if during baseline
    for start, end in baseline_periods:
        if start <= window_center <= end:
            return 'baseline'
    
    # Check if during rest
    for start, end in rest_periods:
        if start <= window_center <= end:
            return 'rest'
    
    # Check if during stress task
    for start, end in stress_periods:
        if start <= window_center <= end:
            return 'during_stress'
    
    # Check if pre-stress (within 15 minutes before any stress onset)
    for onset in stress_onsets:
        if onset - timedelta(minutes=15) <= window_center < onset:
            return 'pre_stress'
    
    # Check if post-stress (within 10 minutes after stress)
    for start, end in stress_periods:
        if end <= window_center <= end + timedelta(minutes=10):
            return 'post_stress'
    
    return 'unknown'


class WindowCreator:
    """
    Creates analysis windows with prediction labels for VitaStress data.
    """
    
    def __init__(
        self,
        window_size_sec: int = 120,
        overlap_ratio: float = 0.0,
        horizons_minutes: List[int] = [3, 5, 10]
    ):
        """
        Initialize window creator.
        
        Args:
            window_size_sec: Size of each window in seconds
            overlap_ratio: Overlap between consecutive windows (0.0 = no overlap)
            horizons_minutes: Prediction horizons to create labels for
        """
        self.window_size_sec = window_size_sec
        self.overlap_ratio = overlap_ratio
        self.horizons_minutes = horizons_minutes
    
    def create_subject_windows(
        self,
        subject_id: str,
        time_grid: pd.DatetimeIndex,
        stress_onsets: List[datetime],
        stress_periods: List[Tuple[datetime, datetime]] = None,
        baseline_periods: List[Tuple[datetime, datetime]] = None,
        rest_periods: List[Tuple[datetime, datetime]] = None
    ) -> List[Window]:
        """
        Create all windows for a subject with labels.
        
        Args:
            subject_id: Subject identifier
            time_grid: 1Hz time grid covering the data
            stress_onsets: List of stress onset timestamps
            stress_periods: Optional list of stress period (start, end) tuples
            baseline_periods: Optional list of baseline period tuples
            rest_periods: Optional list of rest period tuples
        
        Returns:
            List of Window objects with labels
        """
        if len(time_grid) == 0:
            return []
        
        stress_periods = stress_periods or []
        baseline_periods = baseline_periods or []
        rest_periods = rest_periods or []
        
        start_time = time_grid[0]
        end_time = time_grid[-1]
        
        # Create window boundaries
        window_bounds = create_windows(
            start_time,
            end_time,
            self.window_size_sec,
            self.overlap_ratio
        )
        
        # Create Window objects
        windows = []
        for i, (w_start, w_end) in enumerate(window_bounds):
            # Get labels for each horizon
            labels = assign_window_labels(w_end, stress_onsets, self.horizons_minutes)
            
            # Get context
            context = get_window_context(
                w_start, w_end,
                stress_onsets,
                stress_periods,
                baseline_periods,
                rest_periods
            )
            
            window = Window(
                window_id=i,
                start_time=w_start,
                end_time=w_end,
                center_time=w_start + (w_end - w_start) / 2,
                duration_sec=self.window_size_sec,
                label_3min=labels.get('label_3min', 0),
                label_5min=labels.get('label_5min', 0),
                label_10min=labels.get('label_10min', 0),
                context=context,
                subject_id=subject_id
            )
            
            windows.append(window)
        
        return windows
    
    def extract_window_data(
        self,
        window: Window,
        aligned_df: pd.DataFrame,
        ppg_df: Optional[pd.DataFrame] = None
    ) -> WindowData:
        """
        Extract data slices for a specific window.
        
        Args:
            window: Window object
            aligned_df: 1Hz aligned DataFrame
            ppg_df: Optional raw PPG DataFrame
        
        Returns:
            WindowData object with data slices
        """
        # Extract aligned data for this window
        window_aligned = None
        if aligned_df is not None and 'timestamp' in aligned_df.columns:
            mask = (aligned_df['timestamp'] >= window.start_time) & \
                   (aligned_df['timestamp'] < window.end_time)
            window_aligned = aligned_df.loc[mask].copy()
        
        # Extract PPG data for this window
        window_ppg = None
        if ppg_df is not None and 'timestamp' in ppg_df.columns:
            mask = (ppg_df['timestamp'] >= window.start_time) & \
                   (ppg_df['timestamp'] < window.end_time)
            window_ppg = ppg_df.loc[mask, 'value'].values.astype(float)
        
        return WindowData(
            window=window,
            aligned_df=window_aligned,
            ppg_values=window_ppg
        )
    
    def iterate_windows(
        self,
        windows: List[Window],
        aligned_df: pd.DataFrame,
        ppg_df: Optional[pd.DataFrame] = None
    ) -> Iterator[WindowData]:
        """
        Iterate over windows yielding WindowData objects.
        
        Args:
            windows: List of Window objects
            aligned_df: 1Hz aligned DataFrame
            ppg_df: Optional raw PPG DataFrame
        
        Yields:
            WindowData objects for each window
        """
        for window in windows:
            yield self.extract_window_data(window, aligned_df, ppg_df)
    
    def windows_to_dataframe(self, windows: List[Window]) -> pd.DataFrame:
        """
        Convert list of Windows to a DataFrame.
        
        Args:
            windows: List of Window objects
        
        Returns:
            DataFrame with window metadata and labels
        """
        rows = []
        for w in windows:
            rows.append({
                'subject_id': w.subject_id,
                'window_id': w.window_id,
                'window_start': w.start_time,
                'window_end': w.end_time,
                'window_center': w.center_time,
                'duration_sec': w.duration_sec,
                'label_3min': w.label_3min,
                'label_5min': w.label_5min,
                'label_10min': w.label_10min,
                'context': w.context
            })
        
        return pd.DataFrame(rows)
    
    def get_label_distribution(self, windows: List[Window]) -> Dict[str, Dict[int, int]]:
        """
        Get distribution of labels across windows.
        
        Args:
            windows: List of Window objects
        
        Returns:
            Dictionary with label distributions
        """
        distributions = {
            'label_3min': {0: 0, 1: 0},
            'label_5min': {0: 0, 1: 0},
            'label_10min': {0: 0, 1: 0}
        }
        
        for w in windows:
            distributions['label_3min'][w.label_3min] += 1
            distributions['label_5min'][w.label_5min] += 1
            distributions['label_10min'][w.label_10min] += 1
        
        return distributions


def filter_windows_by_context(
    windows: List[Window],
    exclude_contexts: List[str] = ['during_stress', 'unknown']
) -> List[Window]:
    """
    Filter windows by context.
    
    Args:
        windows: List of Window objects
        exclude_contexts: List of contexts to exclude
    
    Returns:
        Filtered list of windows
    """
    return [w for w in windows if w.context not in exclude_contexts]


def balance_windows(
    windows: List[Window],
    label_column: str = 'label_5min',
    strategy: str = 'undersample'
) -> List[Window]:
    """
    Balance windows by label.
    
    Args:
        windows: List of Window objects
        label_column: Which label to balance on
        strategy: 'undersample' or 'oversample'
    
    Returns:
        Balanced list of windows
    """
    # Group by label
    positive = [w for w in windows if getattr(w, label_column) == 1]
    negative = [w for w in windows if getattr(w, label_column) == 0]
    
    if len(positive) == 0 or len(negative) == 0:
        return windows
    
    if strategy == 'undersample':
        # Undersample majority class
        if len(positive) < len(negative):
            np.random.shuffle(negative)
            negative = negative[:len(positive)]
        else:
            np.random.shuffle(positive)
            positive = positive[:len(negative)]
    
    balanced = positive + negative
    np.random.shuffle(balanced)
    
    return balanced


if __name__ == '__main__':
    # Test the window creator
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.label_parser import LabelParser
    from src.data.time_aligner import TimeAligner
    
    data_path = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    
    # Initialize components
    label_parser = LabelParser(data_path)
    time_aligner = TimeAligner(data_path)
    window_creator = WindowCreator(window_size_sec=120)
    
    # Get first subject
    subjects = label_parser.get_subject_folders()
    
    if subjects:
        subject = subjects[0]
        print(f"Testing with subject: {subject[:20]}...")
        
        # Parse annotations
        annotations = label_parser.parse_subject(subject)
        if annotations:
            # Align data
            aligned = time_aligner.align_subject_data(subject)
            
            if aligned:
                # Get stress onsets as datetime list
                stress_onsets = [onset.timestamp for onset in annotations.stress_onsets]
                
                # Get periods as tuples
                stress_periods = [(p.start_time, p.end_time) for p in annotations.stress_periods]
                baseline_periods = [(p.start_time, p.end_time) for p in annotations.baseline_periods]
                rest_periods = [(p.start_time, p.end_time) for p in annotations.rest_periods]
                
                # Create windows
                windows = window_creator.create_subject_windows(
                    subject_id=annotations.subject_id,
                    time_grid=aligned.time_grid,
                    stress_onsets=stress_onsets,
                    stress_periods=stress_periods,
                    baseline_periods=baseline_periods,
                    rest_periods=rest_periods
                )
                
                print(f"\nCreated {len(windows)} windows")
                
                # Show label distribution
                dist = window_creator.get_label_distribution(windows)
                print("\nLabel distribution:")
                for label_name, counts in dist.items():
                    total = counts[0] + counts[1]
                    pos_pct = counts[1] / total * 100 if total > 0 else 0
                    print(f"  {label_name}: {counts[0]} neg, {counts[1]} pos ({pos_pct:.1f}% positive)")
                
                # Show context distribution
                contexts = {}
                for w in windows:
                    contexts[w.context] = contexts.get(w.context, 0) + 1
                print("\nContext distribution:")
                for ctx, count in sorted(contexts.items()):
                    print(f"  {ctx}: {count}")
                
                # Show first few windows
                print("\nFirst 5 windows:")
                for w in windows[:5]:
                    print(f"  Window {w.window_id}: {w.start_time.strftime('%H:%M:%S')} - "
                          f"{w.end_time.strftime('%H:%M:%S')} | "
                          f"labels: 3m={w.label_3min}, 5m={w.label_5min}, 10m={w.label_10min} | "
                          f"context: {w.context}")

