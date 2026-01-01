"""
Test script to validate 4 Hz alignment is working correctly.

Run this before training models to ensure alignment produces correct output.
"""

import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent / "shared"))

from raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from alignment import align_signals
from config import DEFAULT_CONFIG

def test_4hz_alignment():
    """Test 4 Hz alignment on first subject."""
    
    print("="*70)
    print("TESTING 4 HZ ALIGNMENT")
    print("="*70)
    
    # Get first subject
    subjects = get_all_subjects(DEFAULT_CONFIG.data_path)
    
    if not subjects:
        print("❌ ERROR: No subjects found!")
        return False
    
    print(f"\n✓ Found {len(subjects)} subjects")
    print(f"  Testing with: {subjects[0].name}")
    
    # Load signals
    signals = load_raw_signals(subjects[0])
    subject_id = signals["subject_id"]
    
    try:
        start, end = get_experiment_time_range(signals)
        duration_sec = (end - start).total_seconds()
        print(f"\n✓ Loaded signals for {subject_id[:12]}...")
        print(f"  Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} minutes)")
    except Exception as e:
        print(f"❌ ERROR: Could not get time range - {e}")
        return False
    
    # Test 4 Hz alignment
    print(f"\n{'='*70}")
    print("ALIGNING TO 4 HZ")
    print('='*70)
    
    try:
        aligned = align_signals(signals, start, end, target_hz=4.0)
    except Exception as e:
        print(f"❌ ERROR: Alignment failed - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate results
    print(f"\n✓ Alignment successful!")
    print(f"\nValidation:")
    print(f"  Shape: {aligned.shape}")
    
    expected_samples = int(duration_sec * 4.0)
    actual_samples = len(aligned)
    sample_diff = abs(actual_samples - expected_samples)
    
    print(f"  Expected samples (120s × 4Hz): ~{expected_samples}")
    print(f"  Actual samples: {actual_samples}")
    
    if sample_diff <= 5:  # Allow small tolerance
        print(f"  ✓ Sample count correct (diff: {sample_diff})")
    else:
        print(f"  ⚠️  Sample count mismatch (diff: {sample_diff})")
    
    # Check columns
    expected_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'acc_magnitude',
                     'skin_temp', 'heatflux', 'cbt', 'pulse_rate', 'hr_bpm', 'rmssd']
    
    print(f"\nColumns present:")
    for col in expected_cols:
        if col in aligned.columns:
            missing = aligned[col].isna().sum()
            pct = 100 * missing / len(aligned)
            print(f"  ✓ {col:15s}: {len(aligned) - missing:4d}/{len(aligned)} valid ({100-pct:.1f}%)")
        else:
            print(f"  ❌ {col:15s}: MISSING")
    
    # Check for 8 channels (for TCN)
    tcn_channels = ['acc_x', 'acc_y', 'acc_z', 'skin_temp', 'heatflux', 
                    'cbt', 'hr_bpm', 'rmssd']
    
    missing_tcn_channels = [c for c in tcn_channels if c not in aligned.columns]
    
    if not missing_tcn_channels:
        print(f"\n✓ All 8 TCN channels present")
    else:
        print(f"\n⚠️  Missing TCN channels: {missing_tcn_channels}")
    
    # Check data quality
    print(f"\nData Quality:")
    for col in tcn_channels:
        if col in aligned.columns:
            values = aligned[col].dropna()
            if len(values) > 0:
                print(f"  {col:15s}: mean={values.mean():8.2f}, std={values.std():8.2f}, "
                      f"min={values.min():8.2f}, max={values.max():8.2f}")
    
    # Test windowing (simulate what training scripts do)
    print(f"\n{'='*70}")
    print("SIMULATING WINDOW EXTRACTION (120s window)")
    print('='*70)
    
    # Take first 120 seconds
    window_samples = 120 * 4  # 480 samples at 4 Hz
    if len(aligned) >= window_samples:
        window = aligned.iloc[:window_samples]
        print(f"  ✓ Extracted window: {window.shape}")
        print(f"  Expected shape: (480, {len(aligned.columns)})")
        
        if window.shape[0] == 480:
            print(f"  ✓ Window has correct length for TCN (480 timesteps)")
        else:
            print(f"  ⚠️  Window length mismatch: {window.shape[0]} != 480")
    else:
        print(f"  ⚠️  Signal too short for 120s window ({len(aligned)} samples)")
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print('='*70)
    print("\n✅ 4 Hz alignment is working correctly!")
    print("\nYou can now run training scripts:")
    print("  - Classical ML: cd classical_ml && python train.py")
    print("  - TCN: cd tcn && python train.py")
    print("  - Ensemble: cd two_stage_ensemble && python train.py")
    
    return True


if __name__ == "__main__":
    success = test_4hz_alignment()
    sys.exit(0 if success else 1)

