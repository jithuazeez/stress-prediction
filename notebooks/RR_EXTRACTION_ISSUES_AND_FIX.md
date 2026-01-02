# Respiratory Rate Extraction Issues and Fixes

## Summary
All 21 subjects were failing PPG extraction due to **incorrect output file path** in the RRest wrapper. The code was looking for results in the wrong directory.

---

## Critical Issues Found

### 1. **Wrong Results File Path** (Main Issue) ❌

**Problem:**
The notebook looked for RRest output at:
```
/tmp/rrest_temp_data/temp_ppg_data_win_data.mat
```

**Actual Location:**
According to RRest's directory structure (defined in `setup_universal_params.m`), results are saved to:
```
/tmp/rrest_temp_data/temp_ppg_data/Analysis_files/Component_Data/temp_ppg_data_win_data.mat
```

**RRest Directory Structure:**
```
root_folder/
└── dataset_name/
    └── Analysis_files/
        ├── Data_for_Analysis/        # Input data copied here
        ├── Component_Data/            # Results saved here ✓
        │   └── temp_ppg_data_win_data.mat
        └── Results/                   # Final results
```

**Fix:**
```python
# OLD (WRONG):
results_file = self.temp_data_path / f"{dataset_name}_win_data.mat"

# NEW (CORRECT):
results_file = (self.temp_data_path / dataset_name / "Analysis_files" / 
              "Component_Data" / f"{dataset_name}_win_data.mat")
```

---

### 2. **Silent Error Suppression** ⚠️

**Problem:**
The original code had:
```python
try:
    self.eng.eval(f"RRest('{dataset_name}');", nargout=0, timeout=30.0)
    ...
except Exception as e:
    # RRest might fail silently
    return None  # No error message!
```

This made debugging impossible because errors were hidden.

**Fix:**
```python
try:
    self.eng.eval(f"RRest('{dataset_name}');", nargout=0, timeout=60.0)
    ...
except Exception as e:
    print(f"❌ RRest extraction error: {e}")
    import traceback
    traceback.print_exc()  # Show full error trace
    return None
```

---

### 3. **Timeout Too Short** ⏱️

**Problem:**
- Original timeout: 30 seconds
- RRest with AM/FM/BW features + 5 time-domain estimators can take 40-60 seconds

**Fix:**
```python
timeout=60.0  # Increased to 60 seconds
```

---

## Reference Signal Requirement ✓

**From RRest Wiki (Input-Data.md):**
> "It must also contain reference respiratory data..."

**However:**
The custom `setup_universal_params.m` (lines 352-371) already handles missing reference data:
```matlab
if isfield(data(1), 'ref')
    % ... determine ref method ...
else
    % No reference data - skip reference method (for temp Python data)
    up.paramSet.ref_method = 'none';
end
```

This means **reference signal is NOT required** for our use case ✓

---

## RRest Data Structure (Verified)

According to RRest wiki and our implementation:

### Input Format:
```matlab
data(1).ppg.v = [array];     % Row vector of PPG values
data(1).ppg.fs = 64;         % Sampling rate (int32)
data(1).group = 'ppg';       % Group name (string)
```

### Output Format:
```matlab
% File: dataset_name_win_data.mat
win_data.rrEst               % Matrix of RR estimates
                            % Column 1: Fused RR estimate
                            % Other columns: Individual algorithm estimates
```

Our implementation correctly:
1. Creates the input structure ✓
2. Saves to `temp_ppg_data_data.mat` ✓
3. Calls `RRest('temp_ppg_data')` ✓
4. Reads from correct output path ✓ (after fix)

---

## How to Use the Fixed Version

### Option 1: Run the Fixed Python Script
```bash
cd /Users/jithuazeez/Documents/Msc/Dissertation/notebooks
python extract_rr_fixed.py
```

This script includes:
- ✓ Correct file paths
- ✓ Proper error reporting
- ✓ Better progress messages
- ✓ Comprehensive cleanup

### Option 2: Update the Notebook

Apply these changes to `extract_respiratory_rate.ipynb`:

1. **Fix the results file path** (line ~293):
```python
# Change:
results_file = self.temp_data_path / f"{dataset_name}_win_data.mat"

# To:
results_file = (self.temp_data_path / dataset_name / "Analysis_files" / 
              "Component_Data" / f"{dataset_name}_win_data.mat")
```

2. **Add error reporting** (line ~323):
```python
# Change:
except Exception as e:
    return None

# To:
except Exception as e:
    print(f"❌ RRest extraction error: {e}")
    import traceback
    traceback.print_exc()
    return None
```

3. **Increase timeout** (line ~290):
```python
# Change:
timeout=30.0

# To:
timeout=60.0
```

---

## Expected Results After Fix

With the correct paths, you should see:

```
Processing subjects: 100%|████████| 21/21 [XX:XX<00:00,  X.XXs/it]
✓ id_0a73ef1b-...: 45 windows extracted
✓ id_3e775b57-...: 52 windows extracted
...

EXTRACTION COMPLETE
======================================================================
Successful: 21 subjects
Failed: 0 subjects

✓ Saved to: /Users/.../reports/rr_features_from_ppg.csv
  Total windows: 1200
  Subjects: 21
  File size: 89.3 KB
```

---

## Verification Steps

To verify RRest is working correctly:

1. **Check temporary directory structure:**
```bash
ls -R /tmp/rrest_temp_data/temp_ppg_data/
```

Expected:
```
Analysis_files/
├── Component_Data/
│   └── temp_ppg_data_win_data.mat  ← This file!
├── Data_for_Analysis/
│   └── temp_ppg_data_data.mat
└── Results/
```

2. **Inspect output file in MATLAB:**
```matlab
load('/tmp/rrest_temp_data/temp_ppg_data/Analysis_files/Component_Data/temp_ppg_data_win_data.mat')
whos win_data
win_data.rrEst  % Should show RR estimates matrix
```

---

## Root Cause Analysis

The bug occurred because:
1. RRest's documentation doesn't clearly specify the subdirectory structure
2. The `setup_universal_params.m` creates these subdirectories automatically
3. The original code assumed a flat file structure
4. Silent error handling prevented early detection

**Lesson:** When interfacing with external tools, always verify:
- Actual file locations (not assumed locations)
- Enable verbose error reporting during development
- Check tool documentation for directory structure

---

## Related Files

- **Fixed Script:** `notebooks/extract_rr_fixed.py`
- **Original Notebook:** `notebooks/extract_respiratory_rate.ipynb`
- **RRest Config:** `experiments/shared/RRest/RRest/RRest_v3.0/Algorithms/setup_universal_params.m`
- **RRest Wiki:** `experiments/shared/RRest/RRest.wiki/Input-Data.md`

---

## Date
January 2, 2026

## Status
✓ Issue identified
✓ Fix implemented
⏳ Testing required

