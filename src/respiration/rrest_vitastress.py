"""
Respiratory Rate Estimation from VitaStress PPG Data using RRest Toolbox.

This script:
1. Loads PPG infrared data from VitaStress subjects
2. Converts to MATLAB structure format expected by RRest
3. Configures RRest with optimal settings for PPG (Charlton 2016)
4. Runs respiratory rate estimation via MATLAB Engine API
5. Extracts and saves results

Configuration based on:
- Charlton et al. 2016 "An assessment of algorithms to estimate respiratory
  rate from the electrocardiogram and photoplethysmogram"
- Feature-based extraction with AM, FM, BW modulations
- Time-domain RR estimation (CtO, CtA, PKS, ZeX, PZX)
- Smart Fusion (SFu) for combining estimates

Author: Generated for VitaStress Dissertation Project
Date: November 2024
"""

from doctest import DocFileCase
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from scipy import signal as scipy_signal
from fractions import Fraction

# from vitastress_draft import values
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_ppg_missing_values(df: pd.DataFrame, csv_path: Path) -> Dict[str, any]:
    """
    Check for missing values in PPG signal.
    
    Missing values include:
    - NaN values
    - Infinite values
    - Zero values (physiologically impossible for PPG)
    
    Args:
        ppg_values: Array of PPG values
        csv_path: Path to the source file (for error messages)
    
    Returns:
        Dictionary with missing value statistics
        
    Raises:
        ValueError: If missing values are found
    """
    total_samples = len(df['ppg'].dropna())
    
    # Check for different types of missing/invalid values
    nan_count = df['ppg'].isna().sum()
    # nan_count = np.sum(np.isnan(ppg_values))
    zero_count = df['ppg'].eq(0).sum()
    
    # Total missing (NaN, Inf, or Zero)
    total_missing = nan_count  + zero_count
    missing_pct = (total_missing / total_samples) * 100
    
    stats = {
        "total_samples": total_samples,
        "nan_count": nan_count,
        "zero_count": zero_count,
        "total_missing": total_missing,
        "missing_percentage": missing_pct,
        "has_missing": total_missing > 0
    }
    
    # Log the findings
    logger.info(f"  Missing value check:")
    logger.info(f"    NaN values: {nan_count:,}")
    logger.info(f"    Zero values: {zero_count:,}")
    logger.info(f"    Total missing: {total_missing:,} ({missing_pct:.2f}%)")
    
    # Raise error if missing values found
    if total_missing > 0:
        error_msg = (
            f"Missing values detected in PPG signal from {csv_path.name}!\n"
            f"  - NaN values: {nan_count:,}\n"
            f"  - Zero values: {zero_count:,} (zeros are invalid for PPG)\n"
            f"  - Total missing: {total_missing:,} ({missing_pct:.2f}%)\n"
            f"\nPPG signals cannot contain missing values for respiratory rate extraction.\n"
            f"Please clean the data before processing."
        )
        logger.error(error_msg)
    # df[df['value']== 0] = np.nan
    # print(ppg_values.columns)
    # exit()
    df.set_index('date', inplace = True)
    print(df.head())
    ppg_clean = df.loc[:,'ppg']
    ppg_clean = ppg_clean.interpolate(method = 'time', limit_area = 'inside')
    return ppg_clean
    
    

def load_ppg_from_csv(csv_path: Path, check_missing: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Load PPG data from a VitaStress CSV file.
    
    Args:
        csv_path: Path to the *_ppg2_infra_red_22.csv file
        check_missing: If True, check for missing values and raise error if found
    
    Returns:
        Tuple of (ppg_values, sampling_rate_hz, quality_flags)
        
    Raises:
        ValueError: If check_missing=True and missing values are found
        
    Example:
        >>> ppg, fs, quality = load_ppg_from_csv(Path("subject_ppg2_infra_red_22.csv"))
        >>> print(f"Loaded {len(ppg)} samples at {fs:.2f} Hz")
    """
    logger.info(f"Loading PPG data from: {csv_path.name}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    # df=df[df['quality']!=-1]
    # ppg_cleaned = check_ppg_missing_values(ppg_values, csv_path)
    
    # Extract PPG values
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    time_diffs = df["date"].diff().dt.total_seconds().dropna()
    median_interval = time_diffs.median()
    sampling_rate = 1.0 / median_interval
    ppg_cleaned = check_ppg_missing_values(df, csv_path)
    ppg_values = ppg_cleaned.values.astype(np.float64)
    
    
    # Extract quality flags (useful for later filtering)
    # quality_flags = df["quality"].values
    
    # Calculate sampling rate from timestamps



    
    logger.info(f"  Samples: {len(ppg_values):,}")
    logger.info(f"  Duration: {len(ppg_values) / sampling_rate / 60:.1f} minutes")
    logger.info(f"  Sampling rate: {sampling_rate:.2f} Hz")
    # logger.info(f"  Quality range: {quality_flags.min()} - {quality_flags.max()}")
    
    # Check for missing values (NaN, Inf, zeros)
    
    
    target_fs = 125
    # sampling_rate = target_fs # RRest expects 125 Hz
    
    upsampled_signal, new_fs = resample_ppg(ppg_cleaned, sampling_rate, target_fs)
    logger.info(f"Upsampled signal from {sampling_rate:.2f} Hz to {new_fs:.2f} Hz") 
    logger.info(f"Upsampled signal length: {len(upsampled_signal)}")
    return upsampled_signal, target_fs #quality_flags
    
def resample_ppg(ppg_values: np.ndarray, original_fs: float, target_fs: float = 125.0) -> Tuple[np.ndarray, float]:
    """
    Resample PPG signal to a target sampling rate.
    
    Uses scipy.signal.resample_poly for high-quality resampling with
    automatic anti-aliasing filter.
    
    Args:
        ppg_values: Original PPG signal array
        original_fs: Original sampling rate in Hz
        target_fs: Target sampling rate in Hz (default 65 Hz for RRest compatibility)
    
    Returns:
        Tuple of (resampled_signal, actual_new_fs)
        
    Example:
        >>> ppg_resampled, new_fs = resample_ppg(ppg, 62.5, 65.0)
        >>> print(f"Resampled from 62.5 Hz to {new_fs} Hz")
    """
    
    # Find rational approximation for resampling ratio
    # target_fs / original_fs = up / down
    ratio = Fraction(target_fs / original_fs).limit_denominator(1000)
    up = ratio.numerator
    down = ratio.denominator
    
    logger.info(f"  Resampling: {original_fs:.2f} Hz → {target_fs:.2f} Hz (ratio: {up}/{down})")
    
    # Resample using polyphase filtering (high quality)
    resampled = scipy_signal.resample_poly(ppg_values, up, down)
    
    # Calculate actual new sampling rate
    actual_new_fs = original_fs * up / down
    
    logger.info(f"  Samples: {len(ppg_values):,} → {len(resampled):,}")
    logger.info(f"  Actual new fs: {actual_new_fs:.4f} Hz")

    return resampled, actual_new_fs
def find_subject_ppg_files(data_root: Path, num_subjects: Optional[int] = None) -> List[Dict]:
    """
    Find PPG infrared files for all (or first N) subjects.
    
    Args:
        data_root: Path to VitaStress/data folder
        num_subjects: Optional limit on number of subjects to load
    
    Returns:
        List of dicts with 'subject_id', 'folder', and 'ppg_file' keys
    """
    subject_folders = sorted(data_root.glob("id_*"))
    
    if num_subjects is not None:
        subject_folders = subject_folders[:num_subjects]
    
    subjects = []
    for folder in subject_folders:
        subject_id = folder.name.replace("id_", "")
        ppg_files = list(folder.glob("*_ppg3.csv"))
        
        if ppg_files:
            subjects.append({
                "subject_id": subject_id,
                "folder": folder,
                "ppg_file": ppg_files[0]
            })
        else:
            logger.warning(f"No PPG infrared file found for subject {subject_id}")
    
    logger.info(f"Found {len(subjects)} subjects with PPG data")
    return subjects


def prepare_rrest_data(
    subjects: List[Dict],
    output_dir: Path
) -> Tuple[Path, List[str]]:
    """
    Prepare VitaStress PPG data in RRest format and save as .mat file.
    
    RRest expects a MATLAB structure array 'data' with fields:
    - data(n).ppg.fs: sampling rate in Hz
    - data(n).ppg.v: row vector of PPG values
    - data(n).group: group name (optional)
    - data(n).ref.params.rr.v: reference RR values (empty if not available)
    - data(n).ref.params.rr.t: reference RR times (empty if not available)
    
    Args:
        subjects: List of subject dicts from find_subject_ppg_files
        output_dir: Directory to save the .mat file
    
    Returns:
        Tuple of (path to .mat file, list of subject IDs)
    """
    try:
        import matlab.engine
    except ImportError:
        raise ImportError(
            "MATLAB Engine API not found. Install with:\n"
            "  cd /path/to/matlab/extern/engines/python\n"
            "  python setup.py install"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    mat_path = output_dir / "vitastress_ppg_data.mat"
    
    logger.info("Starting MATLAB engine for data preparation...")
    eng = matlab.engine.start_matlab()
    
    try:
        subject_ids = []
        
        for idx, subj in enumerate(subjects):
            logger.info(f"\nProcessing subject {idx + 1}/{len(subjects)}: {subj['subject_id'][:8]}...")
            
            # Load PPG data
            ppg_values, fs = load_ppg_from_csv(subj["ppg_file"])

            subject_ids.append(subj["subject_id"])
            
            # Convert to MATLAB format (row vector)
            # MATLAB expects nested list for row vector: [[val1, val2, ...]]
            ppg_matlab = matlab.double([ppg_values.tolist()])
            
            # Store in MATLAB workspace (1-indexed)
            matlab_idx = idx + 1
            eng.workspace["temp_ppg_v"] = ppg_matlab
            eng.workspace["temp_fs"] = float(fs)
            eng.workspace["temp_group"] = "vitastress"
            
            # Build data structure
            eng.eval(f"data({matlab_idx}).ppg.fs = temp_fs;", nargout=0)
            eng.eval(f"data({matlab_idx}).ppg.v = temp_ppg_v;", nargout=0)
            eng.eval(f"data({matlab_idx}).group = temp_group;", nargout=0)
            
            # Add empty reference (required by some RRest functions)
            eng.eval(f"data({matlab_idx}).ref.params.rr.v = [];", nargout=0)
            eng.eval(f"data({matlab_idx}).ref.params.rr.t = [];", nargout=0)
        
        # Save to .mat file
        eng.eval(f"save('{mat_path}', 'data');", nargout=0)
        logger.info(f"\nSaved data to: {mat_path}")
        
        # Save subject ID mapping for later use
        subject_mapping = pd.DataFrame({
            "matlab_idx": range(1, len(subject_ids) + 1),
            "subject_id": subject_ids
        })
        mapping_path = output_dir / "subject_id_mapping.csv"
        subject_mapping.to_csv(mapping_path, index=False)
        logger.info(f"Saved subject ID mapping to: {mapping_path}")
        
    finally:
        eng.quit()
        logger.info("MATLAB engine closed (data preparation)")
    
    return mat_path, subject_ids


def configure_rrest_params(eng) -> None:
    """
    Configure RRest universal parameters for optimal PPG respiratory rate estimation.
    
    Configuration based on Charlton 2016 best practices:
    - Feature-based extraction only (ppg_feat)
    - IMS beat detection
    - AM, FM, BW feature measurement
    - Cubic spline resampling with bandpass (cubB)
    - Time-domain RR estimation (CtO, CtA, PKS, ZeX, PZX)
    - Smart Fusion (SFu)
    
    Args:
        eng: MATLAB engine instance
    """
    logger.info("Configuring RRest parameters...")
    
    # ================================================================
    # ALGORITHM CONFIGURATION
    # ================================================================
    
    # Specify the stages of the algorithms
    # Using all three stages: extraction, estimation, and fusion
    eng.eval("up.al.key_components = {'extract_resp_sig', 'estimate_rr', 'fuse_rr'};", nargout=0)
    
    # ================================================================
    # STAGE 1: RESPIRATORY SIGNAL EXTRACTION
    # ================================================================
    
    # Use ONLY feature-based extraction for PPG (no filter-based)
    eng.eval("up.al.options.extract_resp_sig = {'ppg_feat'};", nargout=0)
    
    # Specify all components for feature-based extraction
    eng.eval("up.al.sub_components.ppg_feat = {'EHF', 'PDt', 'FPt', 'FMe', 'RS', 'ELF'};", nargout=0)
    
    # Beat Detection: IMS = Incremental-Merge Segmentation
    eng.eval("up.al.options.PDt = {'IMS'};", nargout=0)
    
    # Feature Measurement: AM, FM, BW (the "big three" modulations)
    eng.eval("up.al.options.FMe = {'am', 'fm', 'bw'};", nargout=0)
    
    # Resampling: Cubic spline with bandpass filtering
    eng.eval("up.al.options.RS = {'cubB'};", nargout=0)
    
    # ================================================================
    # STAGE 2: RESPIRATORY RATE ESTIMATION
    # ================================================================
    
    # Time-domain techniques (outperform frequency-domain per Charlton 2016)
    eng.eval("up.al.options.estimate_rr = {'CtO', 'CtA', 'PKS', 'ZeX', 'PZX'};", nargout=0)
    
    # ================================================================
    # STAGE 3: FUSION
    # ================================================================
    
    # Use modulation fusion
    eng.eval("up.al.options.fuse_rr = {'fus_mod'};", nargout=0)
    
    # Smart Fusion - checks agreement, rejects outliers, takes median
    eng.eval("up.al.sub_components.fus_mod = {'SFu'};", nargout=0)
    
    eng.eval("up.al.sub_components.fus_temp = {};", nargout=0)
    # ================================================================
    # ADDITIONAL PARAMETERS
    # ================================================================
    
    # Window duration for RR estimation (32 seconds is standard)
    # eng.eval("up.paramSet.winLeng = 32;", nargout=0)
    
    # # Overlap between consecutive windows (0 = no overlap)
    # eng.eval("up.paramSet.winOverlap = 0;", nargout=0)
    
    # # Expected respiratory rate range (breaths per minute)
    # eng.eval("up.paramSet.rr_range = [4, 60];", nargout=0)
    
    logger.info("RRest parameters configured successfully")



def setup_rrest_directory_structure(output_dir: Path, dataset_name: str) -> Path:
    """
    Create the directory structure that RRest expects.
    
    RRest expects:
    - root_folder/dataset_name/Analysis_files/Data_for_Analysis/dataset_name_data.mat
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
    
    Returns:
        Path to the data file location
    """
    # Create RRest expected directory structure
    dataset_dir = output_dir / dataset_name
    analysis_dir = dataset_dir / "Analysis_files"
    data_dir = analysis_dir / "Data_for_Analysis"
    component_dir = analysis_dir / "Component_Data"
    
    for d in [dataset_dir, analysis_dir, data_dir, component_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created RRest directory structure at: {dataset_dir}")
    
    return data_dir / f"{dataset_name}_data.mat"


def run_rrest_analysis(
    rrest_path: Path,
    data_mat_path: Path,
    output_dir: Path
) -> Path:
    """
    Run RRest analysis on prepared VitaStress data.
    
    Args:
        rrest_path: Path to RRest_v3.0 folder
        data_mat_path: Path to the prepared .mat file
        output_dir: Directory for RRest output
    
    Returns:
        Path to results directory
    """
    try:
        import matlab.engine
    except ImportError:
        raise ImportError("MATLAB Engine API not found")
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING RREST ANALYSIS")
    logger.info("=" * 60)
    
    # Dataset name
    dataset_name = "vitastress_ppg"
    
    # Step 1: Modify RRest setup file to use correct paths
    # logger.info("\n[1/4] Configuring RRest paths...")
    # modify_rrest_setup_file(rrest_path, output_dir)
    
    # Step 2: Create directory structure RRest expects
    logger.info("\n[2/4] Creating directory structure...")
    proper_data_path = setup_rrest_directory_structure(output_dir, dataset_name)
    
    # Start MATLAB engine
    logger.info("\n[3/4] Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    configure_rrest_params(eng)
    
    try:
        # Add RRest to MATLAB path
        logger.info(f"Adding RRest to path: {rrest_path}")
        eng.addpath(eng.genpath(str(rrest_path)), nargout=0)
        
        # Load the prepared data
        logger.info(f"Loading data from: {data_mat_path}")
        eng.eval(f"load('{data_mat_path}');", nargout=0)
        
        # Save data to the location RRest expects
        logger.info(f"Saving data to RRest location: {proper_data_path}")
        eng.eval(f"save('{proper_data_path}', 'data');", nargout=0)
        
        # Configure RRest parameters BEFORE running
        logger.info("\n[4/4] Running RRest analysis...")
        
        
        logger.info(f"Executing RRest('{dataset_name}')...")
        logger.info("This may take several minutes depending on data size...")
        logger.info("-" * 40)
        
        try:
            # Run the analysis
            eng.RRest(dataset_name, nargout=0)
            
            logger.info("-" * 40)
            logger.info("RRest analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during RRest execution: {e}")
            logger.info("Attempting manual respiratory rate extraction...")
            
            # Alternative: Run the analysis manually without RRest
            # run_rrest_manual(eng, output_dir, dataset_name)
            raise e
        results_dir = output_dir / dataset_name / "Analysis_files"
        return results_dir
        
    finally:
        eng.quit()
        logger.info("MATLAB engine closed")


def run_rrest_manual(eng, output_dir: Path, dataset_name: str) -> None:
    """
    Run simplified respiratory rate extraction if RRest fails.
    
    This uses basic signal processing to extract respiratory rate:
    1. Bandpass filter PPG to respiratory frequencies (0.1-0.5 Hz)
    2. Find peaks in filtered signal
    3. Calculate respiratory rate from peak intervals
    
    Args:
        eng: MATLAB engine instance
        output_dir: Output directory
        dataset_name: Name of the dataset
    """
    logger.info("Running manual respiratory rate extraction...")
    logger.info("Using bandpass filtering + peak detection method")
    
    # This is a simplified but functional respiratory rate extraction
    eng.eval("""
        % Get number of subjects
        num_subj = length(data);
        
        % Parameters
        win_length_sec = 32;  % Window length in seconds
        rr_low = 0.1;         % Lower resp freq (6 bpm)
        rr_high = 0.5;        % Upper resp freq (30 bpm)
        
        % Initialize results
        results = struct();
        
        for subj_idx = 1:num_subj
            fprintf('Processing subject %d/%d...\\n', subj_idx, num_subj);
            
            % Get PPG signal
            ppg_sig = double(data(subj_idx).ppg.v(:)');  % Ensure row vector
            fs = data(subj_idx).ppg.fs;
            
            % Store basic info
            results(subj_idx).subject_idx = subj_idx;
            results(subj_idx).fs = fs;
            results(subj_idx).signal_length = length(ppg_sig);
            results(subj_idx).duration_min = length(ppg_sig) / fs / 60;
            
            fprintf('  Signal length: %d samples (%.1f min)\\n', ...
                length(ppg_sig), results(subj_idx).duration_min);
            
            % Design bandpass filter for respiratory frequencies
            try
                [b, a] = butter(4, [rr_low, rr_high] / (fs/2), 'bandpass');
                
                % Filter the signal
                ppg_filt = filtfilt(b, a, ppg_sig);
                
                % Calculate window parameters
                win_samples = round(win_length_sec * fs);
                num_windows = floor(length(ppg_sig) / win_samples);
                
                % Extract RR for each window
                rr_estimates = zeros(1, num_windows);
                window_times = zeros(1, num_windows);
                
                for w = 1:num_windows
                    start_idx = (w-1) * win_samples + 1;
                    end_idx = w * win_samples;
                    
                    win_sig = ppg_filt(start_idx:end_idx);
                    
                    % Find peaks
                    [~, peak_locs] = findpeaks(win_sig, 'MinPeakDistance', round(fs/rr_high));
                    
                    if length(peak_locs) >= 2
                        % Calculate RR from peak intervals
                        peak_intervals = diff(peak_locs) / fs;  % in seconds
                        mean_interval = mean(peak_intervals);
                        rr_estimates(w) = 60 / mean_interval;   % breaths per minute
                    else
                        rr_estimates(w) = NaN;
                    end
                    
                    window_times(w) = (start_idx + end_idx) / 2 / fs;  % center time in seconds
                end
                
                % Store results
                results(subj_idx).rr_estimates = rr_estimates;
                results(subj_idx).window_times = window_times;
                results(subj_idx).num_windows = num_windows;
                results(subj_idx).mean_rr = nanmean(rr_estimates);
                results(subj_idx).std_rr = nanstd(rr_estimates);
                
                fprintf('  Extracted RR for %d windows\\n', num_windows);
                fprintf('  Mean RR: %.1f bpm (std: %.1f)\\n', ...
                    results(subj_idx).mean_rr, results(subj_idx).std_rr);
                
            catch ME
                fprintf('  Error processing: %s\\n', ME.message);
                results(subj_idx).rr_estimates = [];
                results(subj_idx).error = ME.message;
            end
        end
        
        fprintf('\\nManual RR extraction complete.\\n');
    """, nargout=0)
    
    # Save results
    results_path = output_dir / f"{dataset_name}_results.mat"
    eng.eval(f"save('{results_path}', 'results');", nargout=0)
    logger.info(f"Results saved to: {results_path}")
    
    # Also save as CSV for easy access
    csv_path = output_dir / f"{dataset_name}_rr_estimates.csv"
    eng.eval(f"""
        % Export to CSV
        fid = fopen('{csv_path}', 'w');
        fprintf(fid, 'subject_idx,window_idx,time_sec,rr_bpm\\n');
        for subj_idx = 1:length(results)
            if isfield(results(subj_idx), 'rr_estimates') && ~isempty(results(subj_idx).rr_estimates)
                for w = 1:length(results(subj_idx).rr_estimates)
                    fprintf(fid, '%d,%d,%.2f,%.2f\\n', ...
                        subj_idx, w, ...
                        results(subj_idx).window_times(w), ...
                        results(subj_idx).rr_estimates(w));
                end
            end
        end
        fclose(fid);
    """, nargout=0)
    logger.info(f"CSV results saved to: {csv_path}")


def combine_rrest_results(
    results_dir: Path,
    subject_mapping_path: Path,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Combine RRest respiratory rate estimates from all subjects into one DataFrame.
    
    This function:
    1. Loads the subject ID mapping (matlab_idx -> original subject_id)
    2. Loads RR estimates from each subject's *_rrEsts.mat file
    3. Extracts Smart Fusion (SFu) estimates
    4. Combines into a single DataFrame with original subject IDs
    
    Args:
        results_dir: Path to Component_Data folder with RRest outputs
        subject_mapping_path: Path to subject_id_mapping.csv
        output_path: Optional path to save combined results CSV
    
    Returns:
        DataFrame with columns: subject_id, window_idx, window_center_time, rr_bpm, algorithm
    """
    import scipy.io as sio
    
    logger.info("=" * 60)
    logger.info("COMBINING RREST RESULTS FROM ALL SUBJECTS")
    logger.info("=" * 60)
    
    # Load subject ID mapping
    if not subject_mapping_path.exists():
        raise FileNotFoundError(f"Subject mapping not found: {subject_mapping_path}")
    
    subject_mapping = pd.read_csv(subject_mapping_path)
    logger.info(f"Loaded subject mapping: {len(subject_mapping)} subjects")
    
    all_results = []
    subjects_processed = 0
    subjects_failed = 0
    
    for _, row in subject_mapping.iterrows():
        matlab_idx = row["matlab_idx"]
        subject_id = row["subject_id"]
        
        # Paths to subject files
        rr_file = results_dir / f"{matlab_idx}_rrEsts.mat"
        wins_file = results_dir / f"{matlab_idx}_wins.mat"
        
        if not rr_file.exists():
            logger.warning(f"Subject {subject_id} (idx {matlab_idx}): No results found")
            subjects_failed += 1
            continue
        
        try:
            # Load RR estimates
            rr_data = sio.loadmat(str(rr_file))
            
            # Find all Smart Fusion variables (ending in _SFu)
            sfu_keys = [k for k in rr_data.keys() if k.endswith('_SFu') and not k.startswith('_')]
            
            if not sfu_keys:
                logger.warning(f"Subject {subject_id}: No SFu (Smart Fusion) estimates found")
                subjects_failed += 1
                continue
            
            # Load window timing if available
            window_times = None
            if wins_file.exists():
                wins_data = sio.loadmat(str(wins_file))
                # Try to extract window times (varies by RRest version)
                if 't' in wins_data:
                    window_times = wins_data['t'].flatten()
                elif 'timings' in wins_data:
                    window_times = wins_data['timings'].flatten()
            
            # Process each SFu algorithm
            for alg_key in sfu_keys:
                rr_values = rr_data[alg_key].flatten()
                
                # Create DataFrame for this subject/algorithm
                n_windows = len(rr_values)
                
                subj_df = pd.DataFrame({
                    'subject_id': subject_id,
                    'matlab_idx': matlab_idx,
                    'window_idx': range(1, n_windows + 1),
                    'rr_bpm': rr_values,
                    'algorithm': alg_key
                })
                
                # Add window center times if available
                if window_times is not None and len(window_times) == n_windows:
                    subj_df['window_center_time_sec'] = window_times
                else:
                    # Estimate based on 32-second windows (RRest default)
                    subj_df['window_center_time_sec'] = (subj_df['window_idx'] - 0.5) * 32
                
                all_results.append(subj_df)
            
            subjects_processed += 1
            valid_count = np.sum(~np.isnan(rr_values))
            nan_pct = (np.sum(np.isnan(rr_values)) / len(rr_values)) * 100
            logger.info(
                f"Subject {subject_id}: {len(sfu_keys)} algorithms, "
                f"{n_windows} windows, {valid_count} valid ({nan_pct:.1f}% NaN)"
            )
            
        except Exception as e:
            logger.error(f"Subject {subject_id}: Error loading results - {e}")
            subjects_failed += 1
            continue
    
    # Combine all results
    if not all_results:
        logger.error("No results could be loaded from any subject!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("COMBINATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Subjects processed: {subjects_processed}/{len(subject_mapping)}")
    logger.info(f"Subjects failed: {subjects_failed}")
    logger.info(f"Total windows: {len(combined_df):,}")
    logger.info(f"Valid RR estimates: {combined_df['rr_bpm'].notna().sum():,}")
    logger.info(f"NaN rate: {(combined_df['rr_bpm'].isna().sum() / len(combined_df) * 100):.1f}%")
    logger.info(f"Mean RR: {combined_df['rr_bpm'].mean():.1f} bpm")
    logger.info(f"Std RR: {combined_df['rr_bpm'].std():.1f} bpm")
    logger.info(f"RR range: {combined_df['rr_bpm'].min():.1f} - {combined_df['rr_bpm'].max():.1f} bpm")
    
    # Per-algorithm summary
    logger.info("\nPer-algorithm summary:")
    for alg in combined_df['algorithm'].unique():
        alg_data = combined_df[combined_df['algorithm'] == alg]
        nan_pct = (alg_data['rr_bpm'].isna().sum() / len(alg_data)) * 100
        logger.info(f"  {alg}: {nan_pct:.1f}% NaN, mean={alg_data['rr_bpm'].mean():.1f} bpm")
    
    # Save if output path provided
    if output_path:
        combined_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved combined results to: {output_path}")
    
    return combined_df


def extract_results(results_dir: Path) -> pd.DataFrame:
    """
    Extract respiratory rate results from RRest output.
    
    Args:
        results_dir: Path to RRest results directory
    
    Returns:
        DataFrame with respiratory rate estimates
    """
    logger.info(f"Extracting results from: {results_dir}")
    
    # Look for result files
    result_files = list(results_dir.glob("*.mat")) if results_dir.exists() else []
    
    if not result_files:
        logger.warning("No result files found")
        return pd.DataFrame()
    
    logger.info(f"Found {len(result_files)} result file(s)")
    
    # For now, return info about what was generated
    results_info = []
    for f in result_files:
        results_info.append({
            "file": f.name,
            "size_kb": f.stat().st_size / 1024
        })
    
    return pd.DataFrame(results_info)


def main():
    """
    Main function to run respiratory rate estimation on VitaStress data.
    """
    # ================================================================
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
    # ================================================================
    
    VITASTRESS_DATA = Path("/Users/jithuazeez/Documents/Msc/Dissertation/Datasets/VitaStress/data")
    OUTPUT_DIR = Path("/Users/jithuazeez/Documents/Msc/Dissertation/rrest_output")
    
    # Check multiple possible RRest locations
    RREST_POSSIBLE_PATHS = [
        Path("/Users/jithuazeez/Documents/Msc/Dissertation/RRest/RRest_v3.0"),
        Path.home() / "RRest" / "RRest_v3.0",
        Path.cwd() / "RRest" / "RRest_v3.0",
    ]
    
    RREST_PATH = None
    for path in RREST_POSSIBLE_PATHS:
        if path.exists():
            RREST_PATH = path
            break
    
    # Number of subjects to process (None for all, or specify a number)
    NUM_SUBJECTS = None  # Process ALL 21 subjects
    
    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    
    logger.info("=" * 60)
    logger.info("VITASTRESS RESPIRATORY RATE ESTIMATION")
    logger.info("Using RRest Toolbox (Charlton et al. 2016)")
    logger.info("=" * 60)
    
    # Step 1: Find subject PPG files
    logger.info("\n[STEP 1] Finding subject PPG files...")
    subjects = find_subject_ppg_files(VITASTRESS_DATA, NUM_SUBJECTS)
    
    if not subjects:
        logger.error("No subjects found! Check the data path.")
        return
    
    # Step 2: Prepare data in RRest format
    logger.info("\n[STEP 2] Preparing data in RRest format...")
    mat_path, subject_ids = prepare_rrest_data(subjects, OUTPUT_DIR)
    
    # Step 3: Check if RRest path exists
    if RREST_PATH is None or not RREST_PATH.exists():
        logger.warning("\n" + "=" * 60)
        logger.warning("RREST TOOLBOX NOT FOUND")
        logger.warning("=" * 60)
        logger.warning("\nTo download RRest, run these commands:")
        logger.warning("  cd /Users/jithuazeez/Documents/Msc/Dissertation")
        logger.warning("  git clone https://github.com/peterhcharlton/RRest.git")
        logger.warning("\nAfter downloading, re-run this script.")
        logger.info("\n" + "-" * 60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("-" * 60)
        logger.info(f"\nData has been prepared and saved to:\n  {mat_path}")
        logger.info("\nYou can run RRest manually in MATLAB:")
        logger.info("  1. Open MATLAB")
        logger.info("  2. Add RRest to path:")
        logger.info("     >> addpath(genpath('/path/to/RRest/RRest_v3.0'))")
        logger.info("  3. Navigate to output directory:")
        logger.info(f"     >> cd('{OUTPUT_DIR}')")
        logger.info("  4. Load and configure (see matlabsettings.md for config):")
        logger.info(f"     >> load('{mat_path}')")
        logger.info("  5. Run analysis:")
        logger.info("     >> RRest('vitastress_ppg')")
        return mat_path, subject_ids
    
    # Step 4: Run RRest analysis
    logger.info("\n[STEP 3] Running RRest analysis...")
    results_dir = run_rrest_analysis(RREST_PATH, mat_path, OUTPUT_DIR)
    
    # Step 5: Combine results from all subjects
    logger.info("\n[STEP 4] Combining respiratory rate estimates from all subjects...")
    component_data_dir = OUTPUT_DIR / "vitastress_ppg" / "Analysis_files" / "Component_Data"
    subject_mapping_path = OUTPUT_DIR / "subject_id_mapping.csv"
    combined_output_path = OUTPUT_DIR / "vitastress_all_subjects_rr_estimates.csv"
    
    combined_df = combine_rrest_results(
        results_dir=component_data_dir,
        subject_mapping_path=subject_mapping_path,
        output_path=combined_output_path
    )
    
    if not combined_df.empty:
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLE OF COMBINED RESULTS (First 20 rows)")
        logger.info("=" * 60)
        print(combined_df.head(20).to_string(index=False))
        
        logger.info("\n" + "=" * 60)
        logger.info("PER-SUBJECT SUMMARY")
        logger.info("=" * 60)
        subject_summary = combined_df.groupby('subject_id').agg({
            'rr_bpm': ['count', 'mean', 'std', lambda x: x.isna().sum()]
        }).round(2)
        subject_summary.columns = ['Total Windows', 'Mean RR (bpm)', 'Std RR (bpm)', 'NaN Count']
        print(subject_summary.to_string())
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Subjects processed: {len(subject_ids)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Combined results saved to: {combined_output_path}")
    
    return combined_df, subject_ids


if __name__ == "__main__":
    main()

