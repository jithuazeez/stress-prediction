#!/usr/bin/env python3
"""
Quick test script to verify RRest extraction works on a single window.
This helps confirm the fix before processing all subjects.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matlab.engine

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "shared"))
from raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from config import DEFAULT_CONFIG

print("="*70)
print("TESTING RREST EXTRACTION ON SINGLE WINDOW")
print("="*70)

# Paths
RREST_PATH = Path(__file__).parent.parent / "experiments" / "shared" / "RRest" / "RRest" / "RRest_v3.0"
TEMP_PATH = Path("/tmp/rrest_temp_data")
TEMP_PATH.mkdir(exist_ok=True)

print(f"\n1. Initializing MATLAB engine...")
eng = matlab.engine.start_matlab()
eng.eval('warning off all', nargout=0)
print("   ✓ MATLAB engine started")

print(f"\n2. Adding RRest to MATLAB path...")
eng.addpath(str(RREST_PATH), nargout=0)
eng.addpath(str(RREST_PATH / 'Algorithms'), nargout=0)
print(f"   ✓ RRest path: {RREST_PATH}")

print(f"\n3. Loading test subject data...")
config = DEFAULT_CONFIG
subjects = get_all_subjects(config.data_path)
test_subject = subjects[0]
print(f"   Using: {test_subject.name}")

signals = load_raw_signals(test_subject)
ppg_df = signals["ppg"]
start, end = get_experiment_time_range(signals)

# Get first 60-second window
window_start = start + pd.Timedelta(seconds=60)
window_end = window_start + pd.Timedelta(seconds=60)
ppg_mask = (ppg_df["timestamp"] >= window_start) & (ppg_df["timestamp"] < window_end)
ppg_window = ppg_df[ppg_mask]
ppg_values = ppg_window["value"].values
ppg_values = ppg_values[(~np.isnan(ppg_values)) & (ppg_values != 0)]

print(f"   ✓ Extracted {len(ppg_values)} PPG samples ({len(ppg_values)/64:.1f}s)")

print(f"\n4. Creating MATLAB data structure...")
dataset_name = "temp_ppg_data"
mat_file = TEMP_PATH / f"{dataset_name}_data.mat"

eng.workspace['ppg_values'] = matlab.double(ppg_values.tolist())
eng.workspace['fs'] = 64
eng.workspace['mat_file'] = str(mat_file)

eng.eval("""
data = struct();
data(1).ppg.v = ppg_values(:)';
data(1).ppg.fs = int32(fs);
data(1).group = 'ppg';
save(mat_file, 'data');
""", nargout=0)

print(f"   ✓ Saved to: {mat_file}")
print(f"   ✓ File exists: {mat_file.exists()}")

print(f"\n5. Running RRest (this may take 30-60 seconds)...")
try:
    eng.eval(f"RRest('{dataset_name}');", nargout=0, timeout=60.0)
    print("   ✓ RRest completed successfully")
except Exception as e:
    print(f"   ❌ RRest failed: {e}")
    eng.quit()
    sys.exit(1)

print(f"\n6. Checking for output file...")
# Check multiple possible locations
possible_paths = [
    TEMP_PATH / dataset_name / "Analysis_files" / "Component_Data" / f"{dataset_name}_win_data.mat",
    TEMP_PATH / f"{dataset_name}_win_data.mat",
    TEMP_PATH / dataset_name / f"{dataset_name}_win_data.mat",
]

results_file = None
for i, path in enumerate(possible_paths, 1):
    print(f"   Checking path {i}: {path}")
    if path.exists():
        results_file = path
        print(f"   ✓ FOUND at path {i}!")
        break
    else:
        print(f"     ✗ Not found")

if not results_file:
    print("\n   ❌ ERROR: Results file not found at any expected location!")
    print("\n   Directory structure:")
    import os
    for root, dirs, files in os.walk(TEMP_PATH):
        level = root.replace(str(TEMP_PATH), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    eng.quit()
    sys.exit(1)

print(f"\n7. Loading and parsing results...")
rr_data = eng.eval(f"load('{str(results_file)}'); win_data.rrEst;", nargout=1)
rr_estimates = np.array(rr_data)

print(f"   ✓ Loaded RR estimates")
print(f"   Shape: {rr_estimates.shape}")
print(f"   Values: {rr_estimates.flatten()}")

if len(rr_estimates.shape) > 1:
    rr_fused = rr_estimates[:, 0]
else:
    rr_fused = rr_estimates.flatten()

valid_rr = rr_fused[(rr_fused >= 4) & (rr_fused <= 60) & (~np.isnan(rr_fused))]

print(f"\n8. Results:")
if len(valid_rr) > 0:
    print(f"   ✓ SUCCESS!")
    print(f"   Mean RR: {np.mean(valid_rr):.1f} breaths/min")
    print(f"   Std RR:  {np.std(valid_rr):.1f} breaths/min")
    print(f"   Range:   {np.min(valid_rr):.1f} - {np.max(valid_rr):.1f} breaths/min")
    print(f"   N estimates: {len(valid_rr)}")
else:
    print(f"   ⚠ WARNING: No valid RR estimates found")

print(f"\n9. Cleaning up...")
eng.quit()
if mat_file.exists():
    mat_file.unlink()
dataset_dir = TEMP_PATH / dataset_name
if dataset_dir.exists():
    import shutil
    shutil.rmtree(dataset_dir, ignore_errors=True)
print("   ✓ Cleanup complete")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print("="*70)

if len(valid_rr) > 0:
    print("\n✓ RRest extraction is working correctly!")
    print("  You can now run the full extraction on all subjects.")
else:
    print("\n⚠ RRest ran but produced no valid estimates.")
    print("  Check PPG signal quality or RRest parameters.")

