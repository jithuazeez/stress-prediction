"""
Windowing utilities for creating labeled windows from aligned data.

IMPORTANT: This module focuses on EMOTIONAL stress prediction.
- Cognitive stress (mental tasks) → STRESS (label=1)
- Public speaking stress (social anxiety) → STRESS (label=1)  
- Physical stress (exercise) → NO STRESS (label=0)

This helps the model learn to distinguish:
- "High HR + sitting" → Emotional stress
- "High HR + running" → Just exercise, not emotional stress
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta


def parse_stress_events(annotation_df: Optional[pd.DataFrame],
                        stress_start_events: List[str],
                        stress_stop_events: List[str],
                        baseline_events: List[str],
                        emotional_stress_start_events: Optional[List[str]] = None,
                        emotional_stress_stop_events: Optional[List[str]] = None,
                        physical_stress_start_events: Optional[List[str]] = None,
                        physical_stress_stop_events: Optional[List[str]] = None) -> Dict:
    """
    Parse annotation file to extract stress onset times and periods.
    
    Distinguishes between:
    - EMOTIONAL stress (cognitive, public speaking) → labeled as STRESS
    - PHYSICAL stress (exercise) → labeled as NO STRESS
    
    Args:
        annotation_df: DataFrame with 'timestamp' and 'annotation' columns
        stress_start_events: List of ALL stress start events (for context)
        stress_stop_events: List of ALL stress stop events
        baseline_events: List of baseline event names
        emotional_stress_start_events: Events for EMOTIONAL stress (labeled as 1)
        emotional_stress_stop_events: Stop events for emotional stress
        physical_stress_start_events: Events for PHYSICAL stress (labeled as 0)
        physical_stress_stop_events: Stop events for physical stress
    
    Returns:
        Dictionary with:
        - emotional_stress_onsets: List of emotional stress onset timestamps
        - physical_stress_onsets: List of physical stress onset timestamps (NOT labeled as stress)
        - stress_periods: List of (start, stop, stress_type) tuples
        - baseline_periods: List of (start, stop) tuples
    """
    result = {
        "emotional_stress_onsets": [],      # These ARE stress (1)
        "physical_stress_onsets": [],        # These are NOT stress (0)
        "stress_onsets": [],                 # ALL stress onsets (for backwards compat)
        "stress_periods": [],
        "emotional_stress_periods": [],
        "physical_stress_periods": [],
        "baseline_periods": [],
        "all_events": []
    }
    
    if annotation_df is None or len(annotation_df) == 0:
        return result
    
    # Default to all stress events if emotional/physical not specified
    if emotional_stress_start_events is None:
        emotional_stress_start_events = [e for e in stress_start_events if "Physical" not in e]
    if emotional_stress_stop_events is None:
        emotional_stress_stop_events = [e for e in stress_stop_events if "Physical" not in e]
    if physical_stress_start_events is None:
        physical_stress_start_events = [e for e in stress_start_events if "Physical" in e]
    if physical_stress_stop_events is None:
        physical_stress_stop_events = [e for e in stress_stop_events if "Physical" in e]
    
    # Find annotation column
    ann_col = None
    for col in annotation_df.columns:
        if col.lower() in ["annotation", "event", "name", "label"]:
            ann_col = col
            break
    
    if ann_col is None:
        for col in annotation_df.columns:
            if col != "timestamp" and annotation_df[col].dtype == object:
                ann_col = col
                break
    
    if ann_col is None:
        return result
    
    # Store all events
    for _, row in annotation_df.iterrows():
        event_name = str(row[ann_col]).strip()
        event_time = row["timestamp"]
        result["all_events"].append((event_time, event_name))
    
    # Find EMOTIONAL stress onsets (these ARE labeled as stress)
    for _, row in annotation_df.iterrows():
        event_name = str(row[ann_col]).strip()
        event_time = row["timestamp"]
        
        for start_event in emotional_stress_start_events:
            if start_event.lower() in event_name.lower():
                result["emotional_stress_onsets"].append(event_time)
                result["stress_onsets"].append(event_time)  # backwards compat
                break
    
    # Find PHYSICAL stress onsets (NOT labeled as stress - just for context)
    for _, row in annotation_df.iterrows():
        event_name = str(row[ann_col]).strip()
        event_time = row["timestamp"]
        
        for start_event in physical_stress_start_events:
            if start_event.lower() in event_name.lower():
                result["physical_stress_onsets"].append(event_time)
                break
    
    # Find emotional stress periods
    for start_event, stop_event in zip(emotional_stress_start_events, emotional_stress_stop_events):
        start_times = []
        stop_times = []
        
        for _, row in annotation_df.iterrows():
            event_name = str(row[ann_col]).strip()
            event_time = row["timestamp"]
            
            if start_event.lower() in event_name.lower():
                start_times.append(event_time)
            elif stop_event.lower() in event_name.lower():
                stop_times.append(event_time)
        
        for start in start_times:
            for stop in stop_times:
                if stop > start:
                    result["emotional_stress_periods"].append((start, stop, "emotional"))
                    result["stress_periods"].append((start, stop))
                    break
    
    # Find physical stress periods (exercise - NOT labeled as stress)
    for start_event, stop_event in zip(physical_stress_start_events, physical_stress_stop_events):
        start_times = []
        stop_times = []
        
        for _, row in annotation_df.iterrows():
            event_name = str(row[ann_col]).strip()
            event_time = row["timestamp"]
            
            if start_event.lower() in event_name.lower():
                start_times.append(event_time)
            elif stop_event.lower() in event_name.lower():
                stop_times.append(event_time)
        
        for start in start_times:
            for stop in stop_times:
                if stop > start:
                    result["physical_stress_periods"].append((start, stop, "physical"))
                    break
    
    # Find baseline periods
    for baseline_event in baseline_events:
        for _, row in annotation_df.iterrows():
            event_name = str(row[ann_col]).strip()
            event_time = row["timestamp"]
            
            if baseline_event.lower() in event_name.lower():
                if "start" in event_name.lower():
                    end_time = event_time + timedelta(minutes=5)
                    result["baseline_periods"].append((event_time, end_time))
    
    return result


def get_window_context(window_center: pd.Timestamp,
                       stress_periods: List[Tuple],
                       baseline_periods: List[Tuple],
                       stress_onsets: List,
                       emotional_stress_periods: List[Tuple] = None,
                       physical_stress_periods: List[Tuple] = None) -> str:
    """
    Determine the context of a window.
    
    Distinguishes between:
    - during_emotional_stress: cognitive/social stress (labeled as 1)
    - during_physical_activity: exercise (labeled as 0)
    - pre_emotional_stress: before cognitive/social stress onset
    - baseline: confirmed rest period
    
    Args:
        window_center: Center timestamp of the window
        stress_periods: List of (start, stop) stress period tuples
        baseline_periods: List of (start, stop) baseline period tuples
        stress_onsets: List of stress onset timestamps
        emotional_stress_periods: List of (start, stop, type) emotional stress periods
        physical_stress_periods: List of (start, stop, type) physical stress periods
    
    Returns:
        Context string
    """
    # Check if during EMOTIONAL stress (labeled as 1)
    if emotional_stress_periods:
        for start, stop, _ in emotional_stress_periods:
            if start <= window_center <= stop:
                return "during_emotional_stress"
    
    # Check if during PHYSICAL stress/exercise (NOT labeled as stress)
    if physical_stress_periods:
        for start, stop, _ in physical_stress_periods:
            if start <= window_center <= stop:
                return "during_physical_activity"
    
    # Fallback for any stress period
    for start, stop in stress_periods:
        if start <= window_center <= stop:
            return "during_stress"
    
    # Check if during baseline
    for start, stop in baseline_periods:
        if start <= window_center <= stop:
            return "baseline"
    
    # Check if pre-stress (within 15 minutes before stress onset)
    for onset in stress_onsets:
        time_to_stress = (onset - window_center).total_seconds() / 60
        if 0 < time_to_stress <= 15:
            return "pre_stress"
    
    # Check if post-stress (within 10 minutes after stress stop)
    for start, stop in stress_periods:
        time_after_stress = (window_center - stop).total_seconds() / 60
        if 0 < time_after_stress <= 10:
            return "post_stress"
    
    return "unknown"


def create_labeled_windows(aligned_df: pd.DataFrame,
                           event_info: Dict,
                           window_size_sec: int = 120,
                           overlap_ratio: float = 0.0,
                           horizons_minutes: List[int] = [3, 5, 10],
                           skip_first_minutes: int = 5) -> List[Dict]:
    """
    Create labeled windows from aligned data.
    
    IMPORTANT: Only EMOTIONAL stress is labeled as STRESS (1).
    Physical stress (exercise) is labeled as NO STRESS (0).
    
    This helps the model distinguish:
    - Emotional stress (sitting + elevated HR)
    - Physical activity (moving + elevated HR) → NOT stress
    
    Args:
        aligned_df: DataFrame with aligned 1Hz data
        event_info: Dictionary from parse_stress_events()
        window_size_sec: Window size in seconds (default 120)
        overlap_ratio: Window overlap ratio (default 0.0)
        horizons_minutes: List of prediction horizons in minutes
        skip_first_minutes: Minutes to skip at the start
    
    Returns:
        List of window dictionaries
    """
    windows = []
    
    if aligned_df is None or len(aligned_df) == 0:
        return windows
    
    # Use EMOTIONAL stress onsets for labeling (NOT physical)
    emotional_stress_onsets = event_info.get("emotional_stress_onsets", [])
    physical_stress_onsets = event_info.get("physical_stress_onsets", [])
    stress_periods = event_info.get("stress_periods", [])
    emotional_stress_periods = event_info.get("emotional_stress_periods", [])
    physical_stress_periods = event_info.get("physical_stress_periods", [])
    baseline_periods = event_info.get("baseline_periods", [])
    
    # Calculate step size
    step_size = int(window_size_sec * (1 - overlap_ratio))
    if step_size < 1:
        step_size = 1
    
    # Get time range
    start_time = aligned_df["timestamp"].iloc[0]
    end_time = aligned_df["timestamp"].iloc[-1]
    
    # Skip first N minutes
    start_time = start_time + timedelta(minutes=skip_first_minutes)
    
    # Create windows
    current_start = start_time
    window_id = 0
    
    while current_start + timedelta(seconds=window_size_sec) <= end_time:
        window_start = current_start
        window_end = current_start + timedelta(seconds=window_size_sec)
        window_center = current_start + timedelta(seconds=window_size_sec / 2)
        
        # Extract window data
        mask = (aligned_df["timestamp"] >= window_start) & (aligned_df["timestamp"] < window_end)
        window_data = aligned_df.loc[mask].copy()
        
        if len(window_data) < window_size_sec * 0.5:  # Skip if less than 50% data
            logger.warning(f"Window {window_id} has less than 50% data, skipping")
            current_start = current_start + timedelta(seconds=step_size)
            continue
        
        # Create labels for each horizon
        # ONLY label EMOTIONAL stress as 1
        labels = {}
        for horizon in horizons_minutes:
            horizon_end = window_end + timedelta(minutes=horizon)
            
            # Check if any EMOTIONAL stress onset falls within prediction window
            label = 0
            for onset in emotional_stress_onsets:
                if window_end <= onset <= horizon_end:
                    label = 1
                    break
            
            labels[f"label_{horizon}min"] = label
        
        # Get context (includes physical activity detection)
        context = get_window_context(
            window_center, 
            stress_periods, 
            baseline_periods, 
            emotional_stress_onsets,  # Use emotional only
            emotional_stress_periods,
            physical_stress_periods
        )
        
        # Create window dictionary
        window_dict = {
            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,
            "window_center": window_center,
            "duration_sec": window_size_sec,
            "window_data": window_data,
            "context": context,
            **labels
        }
        
        windows.append(window_dict)
        window_id += 1
        current_start = current_start + timedelta(seconds=step_size)
    
    return windows


if __name__ == "__main__":
    # Test windowing
    from raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
    from alignment import align_to_1hz
    from config import DEFAULT_CONFIG
    
    print("Testing EMOTIONAL stress labeling")
    print("="*60)
    print("Physical stress (exercise) → NO STRESS (0)")
    print("Cognitive/Social stress → STRESS (1)")
    print("="*60)
    
    subjects = get_all_subjects(DEFAULT_CONFIG.data_path)
    
    if subjects:
        signals = load_raw_signals(subjects[0])
        start, end = get_experiment_time_range(signals)
        aligned = align_to_1hz(signals, start, end)
        
        # Parse events with emotional/physical separation
        event_info = parse_stress_events(
            signals.get("annotation"),
            DEFAULT_CONFIG.stress_start_events,
            DEFAULT_CONFIG.stress_stop_events,
            DEFAULT_CONFIG.baseline_events,
            emotional_stress_start_events=DEFAULT_CONFIG.emotional_stress_start_events,
            emotional_stress_stop_events=DEFAULT_CONFIG.emotional_stress_stop_events,
            physical_stress_start_events=DEFAULT_CONFIG.physical_stress_start_events,
            physical_stress_stop_events=DEFAULT_CONFIG.physical_stress_stop_events
        )
        
        print(f"\nEMOTIONAL stress onsets (labeled as 1): {len(event_info['emotional_stress_onsets'])}")
        print(f"PHYSICAL stress onsets (NOT labeled): {len(event_info['physical_stress_onsets'])}")
        
        # Create windows
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=DEFAULT_CONFIG.window_size_sec,
            skip_first_minutes=DEFAULT_CONFIG.skip_first_minutes
        )
        
        print(f"\nCreated {len(windows)} windows")
        
        # Count labels
        for horizon in DEFAULT_CONFIG.horizons_minutes:
            label_col = f"label_{horizon}min"
            positive = sum(1 for w in windows if w.get(label_col, 0) == 1)
            print(f"  {label_col}: {positive} EMOTIONAL stress ({100*positive/len(windows):.1f}%)")
        
        # Show context distribution
        contexts = {}
        for w in windows:
            ctx = w.get("context", "unknown")
            contexts[ctx] = contexts.get(ctx, 0) + 1
        print(f"\nContext distribution:")
        for ctx, count in sorted(contexts.items()):
            is_stress = "STRESS" if "emotional" in ctx else "no stress" if "physical" in ctx else "-"
            print(f"  {ctx}: {count} [{is_stress}]")
