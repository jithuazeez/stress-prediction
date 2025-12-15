"""
VitaStress Feature Extraction Pipeline.

End-to-end preprocessing pipeline that:
1. Loads data for each subject
2. Aligns modalities to common time grid
3. Creates 120s windows with prediction labels
4. Extracts 39 features per window
5. Combines all subjects into final dataset

Usage:
    python -m src.pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.label_parser import LabelParser, SubjectAnnotations
from src.data.time_aligner import TimeAligner, AlignedData
from src.data.window_creator import WindowCreator, Window
from src.features.feature_extractor import MasterFeatureExtractor, FEATURE_NAMES


class VitaStressPipeline:
    """
    End-to-end feature extraction pipeline for VitaStress dataset.
    """
    
    def __init__(
        self,
        data_path: str,
        window_size_sec: int = 120,
        skip_first_minutes: int = 5,
        horizons_minutes: List[int] = [3, 5, 10],
        ppg_sample_rate: float = 64.0,
        max_hrv_rejection_rate: float = 0.5
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to VitaStress data directory
            window_size_sec: Size of analysis windows in seconds
            skip_first_minutes: Minutes to skip at start (sensor settling)
            horizons_minutes: Prediction horizons for labels
            ppg_sample_rate: Expected PPG sampling rate
            max_hrv_rejection_rate: Max HeartPy rejection rate to accept
        """
        self.data_path = Path(data_path)
        self.window_size_sec = window_size_sec
        self.skip_first_minutes = skip_first_minutes
        self.horizons_minutes = horizons_minutes
        self.ppg_sample_rate = ppg_sample_rate
        self.max_hrv_rejection_rate = max_hrv_rejection_rate
        
        # Initialize components
        self.label_parser = LabelParser(data_path)
        self.time_aligner = TimeAligner(data_path)
        self.window_creator = WindowCreator(
            window_size_sec=window_size_sec,
            overlap_ratio=0.0,
            horizons_minutes=horizons_minutes
        )
        self.feature_extractor = MasterFeatureExtractor(ppg_sample_rate=ppg_sample_rate)
        
        # Get subjects
        self.subjects = self.label_parser.get_subject_folders()
    
    def process_subject(
        self,
        subject_folder: str,
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Process a single subject and extract features for all windows.
        
        Args:
            subject_folder: Subject folder name
            verbose: Print progress info
        
        Returns:
            DataFrame with features for all windows, or None if processing fails
        """
        subject_id = subject_folder.replace('id_', '')[:8]
        
        if verbose:
            print(f"Processing {subject_id}...", end=" ")
        
        # Step 1: Parse annotations
        annotations = self.label_parser.parse_subject(subject_folder)
        if annotations is None:
            if verbose:
                print("❌ No annotations")
            return None
        
        # Step 2: Align data
        aligned = self.time_aligner.align_subject_data(
            subject_folder,
            skip_first_minutes=self.skip_first_minutes
        )
        if aligned is None:
            if verbose:
                print("❌ Alignment failed")
            return None
        
        # Update PPG sample rate if available
        if 'ppg' in aligned.sample_rates:
            self.feature_extractor = MasterFeatureExtractor(
                ppg_sample_rate=aligned.sample_rates['ppg']
            )
        
        # Step 3: Create windows with labels
        stress_onsets = [onset.timestamp for onset in annotations.stress_onsets]
        stress_periods = [(p.start_time, p.end_time) for p in annotations.stress_periods]
        baseline_periods = [(p.start_time, p.end_time) for p in annotations.baseline_periods]
        rest_periods = [(p.start_time, p.end_time) for p in annotations.rest_periods]
        
        windows = self.window_creator.create_subject_windows(
            subject_id=annotations.subject_id,
            time_grid=aligned.time_grid,
            stress_onsets=stress_onsets,
            stress_periods=stress_periods,
            baseline_periods=baseline_periods,
            rest_periods=rest_periods
        )
        
        if len(windows) == 0:
            if verbose:
                print("❌ No windows created")
            return None
        
        # Step 4: Get combined aligned DataFrame
        aligned_df = self.time_aligner.get_aligned_dataframe(aligned)
        
        # Step 5: Extract features for each window
        rows = []
        n_success = 0
        n_hrv_failed = 0
        
        for window in windows:
            # Extract PPG values for this window
            ppg_values = None
            ppg_quality = None
            
            if aligned.ppg_df is not None:
                mask = (aligned.ppg_df['timestamp'] >= window.start_time) & \
                       (aligned.ppg_df['timestamp'] < window.end_time)
                window_ppg = aligned.ppg_df.loc[mask]
                if len(window_ppg) > 0:
                    ppg_values = window_ppg['value'].values.astype(float)
                    if 'quality' in window_ppg.columns:
                        ppg_quality = window_ppg['quality'].values >= 3
            
            # Extract aligned data for this window
            window_aligned = None
            if aligned_df is not None and 'timestamp' in aligned_df.columns:
                mask = (aligned_df['timestamp'] >= window.start_time) & \
                       (aligned_df['timestamp'] < window.end_time)
                window_aligned = aligned_df.loc[mask].copy()
            
            # Extract features
            features = self.feature_extractor.extract_from_window(
                ppg_values, window_aligned, ppg_quality
            )
            
            # Check HRV quality
            hrv_valid = not np.isnan(features.get('hr_bpm', np.nan))
            if hrv_valid:
                n_success += 1
            else:
                n_hrv_failed += 1
            
            # Build row with metadata
            row = {
                'subject_id': window.subject_id,
                'window_id': window.window_id,
                'window_start': window.start_time,
                'window_end': window.end_time,
                'window_center': window.center_time,
                'duration_sec': window.duration_sec,
                'label_3min': window.label_3min,
                'label_5min': window.label_5min,
                'label_10min': window.label_10min,
                'context': window.context,
                'hrv_valid': hrv_valid
            }
            row.update(features)
            rows.append(row)
        
        if verbose:
            hrv_rate = n_success / len(windows) * 100 if len(windows) > 0 else 0
            print(f"✅ {len(windows)} windows, HRV valid: {n_success}/{len(windows)} ({hrv_rate:.0f}%)")
        
        return pd.DataFrame(rows)
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run the full pipeline on all (or specified) subjects.
        
        Args:
            subjects: List of subject folders to process (None = all)
            verbose: Print progress info
        
        Returns:
            Combined DataFrame with features for all subjects
        """
        if subjects is None:
            subjects = self.subjects
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"VitaStress Feature Extraction Pipeline")
            print(f"{'='*60}")
            print(f"Subjects: {len(subjects)}")
            print(f"Window size: {self.window_size_sec}s")
            print(f"Skip first: {self.skip_first_minutes} minutes")
            print(f"Prediction horizons: {self.horizons_minutes} minutes")
            print(f"{'='*60}\n")
        
        all_dfs = []
        successful = 0
        failed = 0
        
        for i, subject in enumerate(subjects):
            if verbose:
                print(f"[{i+1:2d}/{len(subjects)}] ", end="")
            
            df = self.process_subject(subject, verbose=verbose)
            
            if df is not None:
                all_dfs.append(df)
                successful += 1
            else:
                failed += 1
        
        if len(all_dfs) == 0:
            if verbose:
                print("\n❌ No subjects processed successfully!")
            return pd.DataFrame()
        
        # Combine all subjects
        combined = pd.concat(all_dfs, ignore_index=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Subjects processed: {successful}/{len(subjects)}")
            print(f"Total windows: {len(combined)}")
            print(f"Total features: {len(FEATURE_NAMES)}")
            
            # Label distribution
            print(f"\nLabel distribution:")
            for horizon in self.horizons_minutes:
                col = f'label_{horizon}min'
                if col in combined.columns:
                    pos = combined[col].sum()
                    neg = len(combined) - pos
                    pct = pos / len(combined) * 100 if len(combined) > 0 else 0
                    print(f"  {col}: {neg} neg, {pos} pos ({pct:.1f}% positive)")
            
            # HRV validity
            if 'hrv_valid' in combined.columns:
                valid = combined['hrv_valid'].sum()
                pct = valid / len(combined) * 100 if len(combined) > 0 else 0
                print(f"\nHRV valid: {valid}/{len(combined)} ({pct:.1f}%)")
            
            # Missing data
            print(f"\nFeature completeness:")
            for group_name, group_features in self.feature_extractor.get_feature_groups().items():
                missing_rates = []
                for feat in group_features:
                    if feat in combined.columns:
                        missing_rate = combined[feat].isna().mean() * 100
                        missing_rates.append(missing_rate)
                avg_missing = np.mean(missing_rates) if missing_rates else 100
                print(f"  {group_name}: {100 - avg_missing:.1f}% complete")
        
        return combined
    
    def save_results(
        self,
        df: pd.DataFrame,
        output_path: str,
        include_timestamps: bool = False
    ) -> None:
        """
        Save processed features to CSV.
        
        Args:
            df: DataFrame with features
            output_path: Path to save CSV
            include_timestamps: Include timestamp columns in output
        """
        output_df = df.copy()
        
        # Optionally drop timestamp columns (can cause issues with some ML pipelines)
        if not include_timestamps:
            timestamp_cols = ['window_start', 'window_end', 'window_center']
            for col in timestamp_cols:
                if col in output_df.columns:
                    output_df[col] = output_df[col].astype(str)
        
        output_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved to: {output_path}")
        print(f"   Shape: {output_df.shape}")


def normalize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    method: str = 'subject_zscore'
) -> pd.DataFrame:
    """
    Normalize features (per-subject or global).
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to normalize
        method: 'subject_zscore' or 'global_zscore'
    
    Returns:
        DataFrame with normalized features
    """
    df = df.copy()
    
    if method == 'subject_zscore':
        # Per-subject z-score normalization
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            # Convert to float, handling any object types
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                continue
            
            # Check if column is all NaN
            if df[col].isna().all():
                continue
            
            # Compute per-subject stats
            for subject in df['subject_id'].unique():
                mask = df['subject_id'] == subject
                subject_values = df.loc[mask, col].values.astype(float)
                
                # Get valid (non-NaN) values
                valid_idx = ~np.isnan(subject_values)
                if valid_idx.sum() > 1:
                    valid_values = subject_values[valid_idx]
                    mean = np.mean(valid_values)
                    std = np.std(valid_values)
                    if std > 0:
                        df.loc[mask, col] = (df.loc[mask, col] - mean) / std
    
    elif method == 'global_zscore':
        # Global z-score normalization
        for col in feature_columns:
            if col not in df.columns:
                continue
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                continue
            
            if df[col].isna().all():
                continue
                
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
    
    return df


def drop_high_missing_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features with high missing rate.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to check
        threshold: Maximum allowed missing rate (0.5 = 50%)
    
    Returns:
        Tuple of (filtered DataFrame, list of dropped columns)
    """
    dropped = []
    
    for col in feature_columns:
        if col in df.columns:
            missing_rate = df[col].isna().mean()
            if missing_rate > threshold:
                dropped.append(col)
    
    df_filtered = df.drop(columns=dropped, errors='ignore')
    
    return df_filtered, dropped


if __name__ == '__main__':
    # Run the pipeline
    DATA_PATH = '/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data'
    OUTPUT_PATH = '/Users/jithuazeez/Documents/Msc/Dissertation/vitastress_features_120s.csv'
    
    # Initialize pipeline
    pipeline = VitaStressPipeline(
        data_path=DATA_PATH,
        window_size_sec=120,
        skip_first_minutes=5,
        horizons_minutes=[3, 5, 10]
    )
    
    # Run on all subjects
    df = pipeline.run(verbose=True)
    
    if len(df) > 0:
        # Drop features with >50% missing
        df_filtered, dropped = drop_high_missing_features(df, FEATURE_NAMES, threshold=0.5)
        if dropped:
            print(f"\nDropped {len(dropped)} features with >50% missing:")
            for col in dropped:
                print(f"  - {col}")
        
        # Normalize features (per-subject z-score)
        remaining_features = [f for f in FEATURE_NAMES if f in df_filtered.columns]
        df_normalized = normalize_features(df_filtered, remaining_features, method='subject_zscore')
        
        # Save
        pipeline.save_results(df_normalized, OUTPUT_PATH)
        
        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"{'='*60}")

