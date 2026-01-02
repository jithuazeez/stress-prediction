#!/usr/bin/env python3
"""
Fixed Respiratory Rate Extraction from PPG using RRest v3.0

Key fixes:
1. Correct RRest output file path (subdirectory structure)
2. Proper error reporting (no silent failures)
3. Better cleanup of temporary files
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Matlab engine
import matlab.engine

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "shared"))

from raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from windowing import parse_stress_events
from config import DEFAULT_CONFIG


class RRestMatlabWrapper:
    """Wrapper to call RRest v3.0 via Matlab Engine with correct file paths."""
    
    def __init__(self, rrest_path: Path, window_length: int = 60):
        """
        Initialize RRest wrapper.
        
        Args:
            rrest_path: Path to RRest v3.0 directory
            window_length: RRest window length in seconds (default: 60)
        """
        self.rrest_path = rrest_path
        self.window_length = window_length
        self.eng = None
        self.temp_data_path = Path("/tmp/rrest_temp_data")
        self.temp_data_path.mkdir(exist_ok=True)
        
        print("Initializing Matlab engine...")
        try:
            self.eng = matlab.engine.start_matlab()
            # Suppress MATLAB output
            self.eng.eval('warning off all', nargout=0)
            print("✓ Matlab engine started")
            
            # Add RRest to path
            self.eng.addpath(str(self.rrest_path), nargout=0)
            self.eng.addpath(str(self.rrest_path / 'Algorithms'), nargout=0)
            print(f"✓ Added RRest to Matlab path: {self.rrest_path}")
            
        except Exception as e:
            print(f"❌ Failed to start Matlab engine: {e}")
            raise
    
    def extract_rr_from_window(self, ppg_values: np.ndarray, fs: float = 64.0) -> dict:
        """
        Extract respiratory rate from PPG window using RRest v3.0.
        
        Args:
            ppg_values: 1D array of PPG values
            fs: Sampling rate in Hz
        
        Returns:
            Dictionary with RR estimates or None if failed
        """
        if len(ppg_values) < fs * 30:  # Need at least 30 seconds
            return None
        
        dataset_name = "temp_ppg_data"
        mat_file = self.temp_data_path / f"{dataset_name}_data.mat"
        
        try:
            # Save PPG data in format expected by RRest
            self.eng.workspace['ppg_values'] = matlab.double(ppg_values.tolist())
            self.eng.workspace['fs'] = int(fs)
            self.eng.workspace['mat_file'] = str(mat_file)
            
            # Create and save data structure
            # RRest expects: data(n).ppg.v (row vector), data(n).ppg.fs (int32), data(n).group
            self.eng.eval("""
            % Create data structure for RRest
            data = struct();
            data(1).ppg.v = ppg_values(:)';  % Row vector
            data(1).ppg.fs = int32(fs);
            data(1).group = 'ppg';  % Required field
            
            % Save to mat file
            save(mat_file, 'data');
            """, nargout=0)
            
            # Run RRest on the saved data
            # RRest calls setup_universal_params internally which creates subdirectories
            self.eng.eval(f"RRest('{dataset_name}');", nargout=0)
            
            # CRITICAL FIX: RRest saves results to subdirectory structure
            # Path: root_folder/dataset_name/Analysis_files/Component_Data/dataset_name_win_data.mat
            results_file = (self.temp_data_path / dataset_name / "Analysis_files" / 
                          "Component_Data" / f"{dataset_name}_win_data.mat")
            
            if results_file.exists():
                # Load the results using MATLAB
                # win_data.rrEst contains RR estimates (columns: fused RR, and individual algorithm RRs)
                rr_data = self.eng.eval(f"load('{str(results_file)}'); win_data.rrEst;", nargout=1)
                rr_estimates = np.array(rr_data)
                
                # Get the first column (fused RR estimate)
                if len(rr_estimates.shape) > 1:
                    rr_estimates = rr_estimates[:, 0]  # First column is the fused estimate
                rr_estimates = rr_estimates.flatten()
                
                if len(rr_estimates) > 0:
                    # Filter physiologically valid estimates (4-60 bpm)
                    valid_rr = rr_estimates[(rr_estimates >= 4) & (rr_estimates <= 60) & (~np.isnan(rr_estimates))]
                    
                    if len(valid_rr) > 0:
                        return {
                            'rr_mean': float(np.mean(valid_rr)),
                            'rr_std': float(np.std(valid_rr)) if len(valid_rr) > 1 else 0.0,
                            'rr_min': float(np.min(valid_rr)),
                            'rr_max': float(np.max(valid_rr)),
                            'rr_trend': float(valid_rr[-1] - valid_rr[0]) if len(valid_rr) > 1 else 0.0,
                            'n_estimates': len(valid_rr),
                            'quality': min(1.0, len(valid_rr) / 3.0)
                        }
            else:
                print(f"⚠ Results file not found at: {results_file}")
                return None
            
        except Exception as e:
            print(f"❌ RRest extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temp files
            if mat_file.exists():
                mat_file.unlink()
            # Clean up RRest generated directory structure
            dataset_dir = self.temp_data_path / dataset_name
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir, ignore_errors=True)
        
        return None
    
    def close(self):
        """Close Matlab engine."""
        if self.eng:
            print("\nClosing Matlab engine...")
            self.eng.quit()
            print("✓ Matlab engine closed")


def extract_rr_for_subject(subject_folder: Path, 
                          rrest_wrapper,
                          config,
                          window_size_sec: int = 120,
                          ppg_sampling_rate: float = 64.0,
                          min_ppg_coverage: float = 0.5) -> pd.DataFrame:
    """
    Extract RR features for all 120s windows in a subject.
    
    Uses 60s sub-windows for RRest, then aggregates to 120s windows.
    """
    signals = load_raw_signals(subject_folder)
    subject_id = signals["subject_id"]
    
    # Check PPG availability
    ppg_df = signals.get("ppg")
    if ppg_df is None or len(ppg_df) == 0:
        return pd.DataFrame()
    
    # Get time range
    try:
        start, end = get_experiment_time_range(signals)
    except ValueError:
        return pd.DataFrame()
    
    # Create 120s windows
    overlap_ratio = config.overlap_ratio
    overlap_sec = int(window_size_sec * overlap_ratio)
    step_sec = window_size_sec - overlap_sec
    
    # Generate window starts
    skip_sec = config.skip_first_minutes * 60
    window_starts = []
    current_time = start + pd.Timedelta(seconds=skip_sec)
    
    while current_time + pd.Timedelta(seconds=window_size_sec) <= end:
        window_starts.append(current_time)
        current_time += pd.Timedelta(seconds=step_sec)
    
    # Extract RR for each window
    rr_rows = []
    
    for window_start in window_starts:
        window_end = window_start + pd.Timedelta(seconds=window_size_sec)
        
        # Extract PPG for this 120s window
        ppg_mask = (ppg_df["timestamp"] >= window_start) & (ppg_df["timestamp"] < window_end)
        ppg_window = ppg_df[ppg_mask]
        
        # Check coverage
        expected_samples = window_size_sec * ppg_sampling_rate
        if len(ppg_window) < min_ppg_coverage * expected_samples:
            continue
        
        # Get PPG values and remove NaN/zeros
        ppg_values = ppg_window["value"].values
        ppg_values = ppg_values[(~np.isnan(ppg_values)) & (ppg_values != 0)]
        
        if len(ppg_values) < min_ppg_coverage * expected_samples:
            continue
        
        # Split into two 60s sub-windows for RRest
        mid_point = len(ppg_values) // 2
        sub_window_1 = ppg_values[:mid_point]
        sub_window_2 = ppg_values[mid_point:]
        
        # Extract RR from each sub-window
        rr_1 = rrest_wrapper.extract_rr_from_window(sub_window_1, fs=ppg_sampling_rate)
        rr_2 = rrest_wrapper.extract_rr_from_window(sub_window_2, fs=ppg_sampling_rate)
        
        # Aggregate if we have at least one estimate
        valid_estimates = []
        if rr_1: valid_estimates.append(rr_1['rr_mean'])
        if rr_2: valid_estimates.append(rr_2['rr_mean'])
        
        if len(valid_estimates) > 0:
            row = {
                'subject_id': subject_id,
                'window_start': window_start,
                'window_end': window_end,
                'rr_mean': np.mean(valid_estimates),
                'rr_std': np.std(valid_estimates) if len(valid_estimates) > 1 else 0.0,
                'rr_min': np.min(valid_estimates),
                'rr_max': np.max(valid_estimates),
                'rr_trend': valid_estimates[-1] - valid_estimates[0] if len(valid_estimates) > 1 else 0.0,
                'n_sub_windows': len(valid_estimates),
                'quality': len(valid_estimates) / 2.0,  # Both sub-windows = quality 1.0
                'ppg_coverage': len(ppg_values) / expected_samples
            }
            rr_rows.append(row)
    
    return pd.DataFrame(rr_rows)


def main():
    # Paths
    RREST_PATH = Path(__file__).parent.parent / "experiments" / "shared" / "RRest" / "RRest" / "RRest_v3.0"
    OUTPUT_PATH = Path(__file__).parent.parent / "reports" / "rr_features_from_ppg.csv"
    
    # Processing parameters
    config = DEFAULT_CONFIG
    WINDOW_SIZE_SEC = 120  # 120-second stress prediction windows
    RREST_WINDOW_SEC = 60  # 60-second RRest sub-windows (recommended)
    PPG_SAMPLING_RATE = 64.0  # Hz
    MIN_PPG_COVERAGE = 0.5  # Require 50% PPG data in window
    
    print("="*70)
    print("RESPIRATORY RATE EXTRACTION FROM PPG USING RREST v3.0 (FIXED)")
    print("="*70)
    print(f"RRest path: {RREST_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Window size: {WINDOW_SIZE_SEC}s")
    print(f"RRest sub-window: {RREST_WINDOW_SEC}s")
    print()
    
    # Initialize RRest
    rrest = RRestMatlabWrapper(RREST_PATH, window_length=RREST_WINDOW_SEC)
    
    # Get all subjects
    subjects = get_all_subjects(config.data_path)
    print(f"Found {len(subjects)} subjects")
    print()
    
    # Extract RR for all subjects
    all_rr_data = []
    failed_subjects = []
    
    for subject_folder in tqdm(subjects, desc="Processing subjects"):
        try:
            rr_df = extract_rr_for_subject(
                subject_folder, rrest, config,
                window_size_sec=WINDOW_SIZE_SEC,
                ppg_sampling_rate=PPG_SAMPLING_RATE,
                min_ppg_coverage=MIN_PPG_COVERAGE
            )
            
            if len(rr_df) > 0:
                all_rr_data.append(rr_df)
                print(f"✓ {subject_folder.name[:15]}: {len(rr_df)} windows extracted")
            else:
                failed_subjects.append(subject_folder.name)
                print(f"⚠ {subject_folder.name[:15]}: No windows extracted")
        except Exception as e:
            print(f"\n❌ Error processing {subject_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            failed_subjects.append(subject_folder.name)
    
    # Close Matlab engine
    rrest.close()
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {len(all_rr_data)} subjects")
    print(f"Failed: {len(failed_subjects)} subjects")
    if failed_subjects:
        print(f"Failed subjects: {', '.join([s[:12] for s in failed_subjects[:5]])}...")
    
    # Combine and save
    if all_rr_data:
        rr_df = pd.concat(all_rr_data, ignore_index=True)
        
        # Save to CSV
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        rr_df.to_csv(OUTPUT_PATH, index=False)
        
        print(f"\n✓ Saved to: {OUTPUT_PATH}")
        print(f"  Total windows: {len(rr_df)}")
        print(f"  Subjects: {rr_df['subject_id'].nunique()}")
        print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    else:
        print("\n❌ No data extracted!")


if __name__ == "__main__":
    main()

