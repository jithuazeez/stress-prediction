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
import logging
from shared.hrv_extractor import extract_hr_timeseries_from_ppg
from shared.alignment import align_signals
logger = logging.getLogger(__name__)


def parse_stress_events(annotation_df: Optional[pd.DataFrame],
                        stress_start_events: List[str],
                        stress_stop_events: List[str]) -> Dict:
    """
     Parse annotation file to extract stress onset times and periods.
    
    For binary classification:
    - STRESS (1): Emotional stress events (cognitive, public speaking)
    - NOT STRESS (0): Everything else
    
    Args:
        annotation_df: DataFrame with 'timestamp' and annotation columns
        stress_start_events: List of stress start event names (e.g., "Cognitive: Start")
        stress_stop_events: List of stress stop event names (e.g., "Cognitive: Stop")
    
    Returns:
        Dictionary with:
        - stress_onsets: List of stress onset timestamps (for labeling)
        - stress_periods: List of (start, stop) tuples (for context)
    """
    result = {
        "stress_onsets": [],
        "stress_periods": []
    }
    
    # if annotation_df is None or len(annotation_df) == 0:
    #     return result
    
    # # Default to all stress events if emotional/physical not specified
    # if emotional_stress_start_events is None:
    #     emotional_stress_start_events = [e for e in stress_start_events if "Physical" not in e]
    # if emotional_stress_stop_events is None:
    #     emotional_stress_stop_events = [e for e in stress_stop_events if "Physical" not in e]
    # if physical_stress_start_events is None:
    #     physical_stress_start_events = [e for e in stress_start_events if "Physical" in e]
    # if physical_stress_stop_events is None:
    #     physical_stress_stop_events = [e for e in stress_stop_events if "Physical" in e]
    
    # # Find annotation column
    # ann_col = None
    # for col in annotation_df.columns:
    #     if col.lower() in ["annotation", "event", "name", "label"]:
    #         ann_col = col
    #         break
    
    # if ann_col is None:
    #     for col in annotation_df.columns:
    #         if col != "timestamp" and annotation_df[col].dtype == object:
    #             ann_col = col
    #             break
    
    # if ann_col is None:
    #     return result
    
    # # Store all events
    # for _, row in annotation_df.iterrows():
    #     event_name = str(row[ann_col]).strip()
    #     event_time = row["timestamp"]
    #     result["all_events"].append((event_time, event_name))
    
    # # Find EMOTIONAL stress onsets (these ARE labeled as stress)
    # for _, row in annotation_df.iterrows():
    #     event_name = str(row[ann_col]).strip()
    #     event_time = row["timestamp"]
        
    #     for start_event in emotional_stress_start_events:
    #         if start_event.lower() in event_name.lower():
    #             result["emotional_stress_onsets"].append(event_time)
    #             result["stress_onsets"].append(event_time)  # backwards compat
    #             break
    
    # # Find PHYSICAL stress onsets (NOT labeled as stress - just for context)
    # # for _, row in annotation_df.iterrows():
    # #     event_name = str(row[ann_col]).strip()
    # #     event_time = row["timestamp"]
        
    # #     for start_event in physical_stress_start_events:
    # #         if start_event.lower() in event_name.lower():
    # #             result["physical_stress_onsets"].append(event_time)
    # #             break
    
    # # Find emotional stress periods
    # for start_event, stop_event in zip(emotional_stress_start_events, emotional_stress_stop_events):
    #     start_times = []
    #     stop_times = []
        
    #     for _, row in annotation_df.iterrows():
    #         event_name = str(row[ann_col]).strip()
    #         event_time = row["timestamp"]
            
    #         if start_event.lower() in event_name.lower():
    #             start_times.append(event_time)
    #         elif stop_event.lower() in event_name.lower():
    #             stop_times.append(event_time)
        
    #     for start in start_times:
    #         for stop in stop_times:
    #             if stop > start:
    #                 result["emotional_stress_periods"].append((start, stop, "emotional"))
    #                 result["stress_periods"].append((start, stop))
    #                 break
    
    # # Find physical stress periods (exercise - NOT labeled as stress)
    # for start_event, stop_event in zip(physical_stress_start_events, physical_stress_stop_events):
    #     start_times = []
    #     stop_times = []
        
    #     for _, row in annotation_df.iterrows():
    #         event_name = str(row[ann_col]).strip()
    #         event_time = row["timestamp"]
            
    #         if start_event.lower() in event_name.lower():
    #             start_times.append(event_time)
    #         elif stop_event.lower() in event_name.lower():
    #             stop_times.append(event_time)
        
    #     for start in start_times:
    #         for stop in stop_times:
    #             if stop > start:
    #                 result["physical_stress_periods"].append((start, stop, "physical"))
    #                 break
    
    # # Find baseline periods
    # for start_event, stop_event in zip(baseline_start_events, baseline_stop_events):
    #     start_times = []
    #     stop_times = []
        
    #     for _, row in annotation_df.iterrows():
    #         event_name = str(row[ann_col]).strip()
    #         event_time = row["timestamp"]
            
    #         if start_event.lower() in event_name.lower():
    #             start_times.append(event_time)
    #         elif stop_event.lower() in event_name.lower():
    #             stop_times.append(event_time)
        
    #     for start in start_times:
    #         for stop in stop_times:
    #             if stop > start:
    #                 result["baseline_periods"].append((start, stop))
    #                 break

    if annotation_df is None or len(annotation_df) == 0:
        return result
    
    # Find annotation column
    ann_col = None
    for col in annotation_df.columns:
        if col.lower() in ["annotation", "event", "name", "label", "button name"]:
            ann_col = col
            break
    
    if ann_col is None:
        # Try to find any string column that's not timestamp
        for col in annotation_df.columns:
            if col != "timestamp" and annotation_df[col].dtype == object:
                ann_col = col
                break
    
    if ann_col is None:
        return result
    
    # Find stress onsets
    for _, row in annotation_df.iterrows():
        event_name = str(row[ann_col]).strip()
        event_time = row["timestamp"]
        
        for start_event in stress_start_events:
            if start_event.lower() in event_name.lower():
                result["stress_onsets"].append(event_time)
                break
    
    # Find stress periods (start to stop)
    for start_event, stop_event in zip(stress_start_events, stress_stop_events):
        start_times = []
        stop_times = []
        
        for _, row in annotation_df.iterrows():
            event_name = str(row[ann_col]).strip()
            event_time = row["timestamp"]
            
            if start_event.lower() in event_name.lower():
                start_times.append(event_time)
            elif stop_event.lower() in event_name.lower():
                stop_times.append(event_time)
        
        # Pair each start with its next stop
        for start in start_times:
            for stop in stop_times:
                if stop > start:
                    result["stress_periods"].append((start, stop))
                    break
    
    return result






def compute_subject_stats(aligned_df: pd.DataFrame,
                          channels: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute per-subject statistics for subject-wise normalization.
    
    These statistics should be computed on the ENTIRE subject's data
    (before windowing) to enable proper subject-wise z-score normalization.
    
    Args:
        aligned_df: DataFrame with aligned 1Hz data for ONE subject
        channels: List of channel names to compute stats for.
                  If None, uses default channels.
    
    Returns:
        Dictionary mapping channel names to their mean and std:
        {
            "acc_magnitude": {"mean": 1.0, "std": 0.1},
            "skin_temp": {"mean": 32.5, "std": 1.2},
            ...
        }
    """
    if channels is None:
        # Default channels used across experiments
        channels = [
            col for col in aligned_df.columns 
            if col != "timestamp" and pd.api.types.is_numeric_dtype(aligned_df[col])
        ]
    
    stats = {}
    
    for channel in channels:
        if channel in aligned_df.columns:
            values = aligned_df[channel].values
            # Remove NaN for stats computation
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                stats[channel] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values))
                }
            else:
                # No valid data - use default (0, 1) to avoid division by zero
                stats[channel] = {"mean": 0.0, "std": 1.0}
        else:
            # Channel not in data - use default
            stats[channel] = {"mean": 0.0, "std": 1.0}
    
    return stats


def create_labeled_windows(aligned_df: pd.DataFrame,
                           event_info: Dict,
                           window_size_sec: int = 120,
                           overlap_ratio: float = 0.0,
                           horizons_minutes: List[int] = [3, 5, 10],
                           skip_first_minutes: int = 5,
                           subject_stats: Dict[str, Dict[str, float]] = None) -> List[Dict]:
    """
    Create labeled windows from aligned data with optional per-window HR extraction.
    
    IMPORTANT: Only EMOTIONAL stress is labeled as STRESS (1).
    Physical stress (exercise) is labeled as NO STRESS (0).
    
    This helps the model distinguish:
    - Emotional stress (sitting + elevated HR)
    - Physical activity (moving + elevated HR) → NOT stress
    
    Args:
        aligned_df: DataFrame with aligned data (without HR - simple sensors only)
        event_info: Dictionary from parse_stress_events()
        window_size_sec: Window size in seconds (default 120)
        overlap_ratio: Window overlap ratio (default 0.0)
        horizons_minutes: List of prediction horizons in minutes
        skip_first_minutes: Minutes to skip at the start
        subject_stats: Pre-computed subject-level statistics for normalization.
                       If provided, will be attached to each window for subject-wise normalization.
                       Should be computed using compute_subject_stats() on the full aligned data.

    
    Returns:
        List of window dictionaries with 'subject_stats' key if provided.
        If ppg_data provided, each window will include hr_bpm and rmssd columns.
    """

 
    
    if aligned_df is None or len(aligned_df) == 0:
        return []
    
    # Use EMOTIONAL stress onsets for labeling (NOT physical)
    emotional_stress_onsets = event_info.get("stress_onsets", [])
    
    # Calculate step size
    step_size = int(window_size_sec * (1 - overlap_ratio))
    step_size_sec = max(1, step_size)
    
    # Get time range
    start_time = aligned_df["timestamp"].iloc[0] + timedelta(minutes=skip_first_minutes)
    end_time = aligned_df["timestamp"].iloc[-1]
    
    window_starts = pd.date_range(
        start=start_time,
        end=end_time - timedelta(seconds=window_size_sec),
        freq=f"{step_size_sec}S"
    )
    

    windows = []
    rejected_count = 0
    # rejected_hr_extraction = 0  # Track HR extraction failures
    # rejected_hr_coverage = 0     # Track insufficient HR coverage  
    # rejected_insufficient_data = 0  # Track insufficient window data

    for window_id, window_start in enumerate(window_starts):
        window_end = window_start + timedelta(seconds=window_size_sec)
        window_center = window_start + timedelta(seconds=window_size_sec / 2)

        window_data = aligned_df[(aligned_df["timestamp"] >= window_start ) & (aligned_df["timestamp"] < window_end)].copy()
        
        # Extract HR from raw PPG for this window
        # # hr_extraction_success = False
        # if ppg_data is not None:
        #     from shared.hrv_extractor import extract_hr_timeseries_from_ppg
        #     from shared.alignment import resample_signal_scipy
            
        #     hr_data = extract_hr_timeseries_from_ppg(ppg_data, window_start, window_end, sample_rate=64.0)
            
        #     if len(hr_data['timestamps']) > 0:
        #         try:
        #             # Create time grid for this window at target_hz
        #             period_ms = int(1000 / target_hz)
        #             time_grid = pd.date_range(window_start, window_end, freq=f"{period_ms}ms", inclusive='left')
                    
        # #             # Interpolate HR to the window's time grid
        # #             # Convert timestamps to float (nanoseconds since epoch)
        #             hr_ts = pd.to_datetime(hr_data['timestamps']).values.astype('datetime64[ns]').astype(float)
        #             grid_ts = time_grid.values.astype('datetime64[ns]').astype(float)
                    
        #             hr_bpm_aligned = resample_signal_scipy(hr_ts, hr_data['hr_bpm'], grid_ts, method="linear")
        #             rmssd_aligned = resample_signal_scipy(hr_ts, hr_data['rmssd'], grid_ts, method="linear")
                    
        #             # Check if we have sufficient HR coverage (at least 50% non-NaN)
        #             hr_valid_count = np.sum(~np.isnan(hr_bpm_aligned))
        #             hr_coverage = hr_valid_count / len(hr_bpm_aligned) if len(hr_bpm_aligned) > 0 else 0
                    
        #             if hr_coverage >= 0.5:  # Require at least 50% HR coverage
        #                 # Add HR columns to window_data
        #                 window_data["hr_bpm"] = hr_bpm_aligned
        #                 window_data["rmssd"] = rmssd_aligned
        #                 hr_extraction_success = True
        #             # else: Insufficient coverage, will be rejected below
                    
                # except Exception as e:
                #     print(f"Interpolation failed: {e}")
                #     raise e
        #             pass
        
        # # If HR extraction failed, reject this window (skip it entirely)
        # if not hr_extraction_success:
        #     rejected_count += 1
        #     rejected_hr_extraction += 1
        #     continue

        # # Reject if insufficient sensor data in window
        # if len(window_data) < window_size_sec * 0.5:
        #     rejected_count += 1
        #     rejected_insufficient_data += 1
        #     continue

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
        
        window_dict = {
            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,
            "window_center": window_center,
            "duration_sec": window_size_sec,
            "window_data": window_data,
            **labels
        }
        
        # Add subject-level statistics for normalization if provided
        if subject_stats is not None:
            window_dict["subject_stats"] = subject_stats
        
        windows.append(window_dict)
    # Log windowing statistics
    total_accepted = len(windows)
    total_attempted = len(window_starts)
    
    if total_attempted > 0:
        acceptance_rate = 100 * total_accepted / total_attempted
        rejection_rate = 100 * rejected_count / total_attempted
        
        # rejection_details = []
        # if rejected_hr_extraction > 0:
        #     rejection_details.append(f"HR extraction: {rejected_hr_extraction}")
        # if rejected_hr_coverage > 0:
        #     rejection_details.append(f"HR coverage: {rejected_hr_coverage}")
        # if rejected_insufficient_data > 0:
        #     rejection_details.append(f"Insufficient data: {rejected_insufficient_data}")
        
        # details_str = ", ".join(rejection_details) if rejection_details else "None"
        
        logger.debug(
            f"Windowing: {total_accepted}/{total_attempted} windows accepted "
            f"({acceptance_rate:.1f}%), {rejected_count} rejected ({rejection_rate:.1f}%) "
            # f"[Reasons: {details_str}]"
        )
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
        
        # Compute subject-level statistics for normalization
        subject_stats = compute_subject_stats(aligned)
        print(f"\nSubject-wise statistics computed:")
        for channel, stats in subject_stats.items():
            if stats["std"] != 1.0:  # Only print channels with real data
                print(f"  {channel}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
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
        
        # Create windows with subject stats
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=DEFAULT_CONFIG.window_size_sec,
            skip_first_minutes=DEFAULT_CONFIG.skip_first_minutes,
            subject_stats=subject_stats  # Pass subject stats for normalization
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
        
        # Verify subject stats are in windows
        if windows and "subject_stats" in windows[0]:
            print(f"\n✓ Subject-wise normalization stats attached to all {len(windows)} windows")
