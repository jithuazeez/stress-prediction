# DIAGNOSTIC CELL - Add this to your notebook and run it to debug the failures

print("="*70)
print("DIAGNOSTIC ANALYSIS")
print("="*70)

# Test 1: Check Matlab Engine
print("\n1. Testing Matlab Engine...")
try:
    import matlab.engine
    print("   ✓ matlab.engine imported successfully")
    
    # Try to start engine
    print("   Starting Matlab engine (this may take 30-60 seconds)...")
    eng = matlab.engine.start_matlab()
    print("   ✓ Matlab engine started successfully")
    
    # Test basic operation
    result = eng.sqrt(4.0)
    print(f"   ✓ Matlab computation works (sqrt(4) = {result})")
    
    eng.quit()
    print("   ✓ Matlab engine closed")
    
except ImportError as e:
    print(f"   ❌ Failed to import matlab.engine: {e}")
    print("   → Install with: cd /Applications/MATLAB_R20XXx.app/extern/engines/python && python setup.py install")
except Exception as e:
    print(f"   ❌ Matlab engine error: {e}")

# Test 2: Check RRest Path
print("\n2. Testing RRest Path...")
RREST_PATH = Path.cwd().parent / "experiments" / "shared" / "RRest" / "RRest" / "RRest_v3.0"
print(f"   Path: {RREST_PATH}")
print(f"   Exists: {RREST_PATH.exists()}")

if RREST_PATH.exists():
    algorithms_path = RREST_PATH / "Algorithms"
    print(f"   Algorithms/: {algorithms_path.exists()}")
    
    if algorithms_path.exists():
        rrest_m = algorithms_path / "RRest.m"
        print(f"   RRest.m: {rrest_m.exists()}")
    else:
        print("   ❌ Algorithms directory not found!")
else:
    print("   ❌ RRest path not found!")
    print(f"   Looking for: {RREST_PATH}")

# Test 3: Check Data Path
print("\n3. Testing Data Access...")
subjects = get_all_subjects(config.data_path)
print(f"   Found {len(subjects)} subjects")

if len(subjects) > 0:
    # Test first subject
    test_subject = subjects[0]
    print(f"\n   Testing first subject: {test_subject.name}")
    
    try:
        signals = load_raw_signals(test_subject)
        print(f"   ✓ Loaded signals for {signals['subject_id']}")
        
        # Check PPG
        ppg_df = signals.get("ppg")
        if ppg_df is not None:
            print(f"   ✓ PPG data available: {len(ppg_df)} samples")
            print(f"   ✓ PPG sampling rate: ~{len(ppg_df) / ((ppg_df['timestamp'].max() - ppg_df['timestamp'].min()).total_seconds()):.1f} Hz")
            
            # Check for NaN/zeros
            ppg_values = ppg_df["value"].values
            valid_values = ppg_values[(~np.isnan(ppg_values)) & (ppg_values != 0)]
            print(f"   ✓ Valid PPG samples: {len(valid_values)} / {len(ppg_values)} ({100*len(valid_values)/len(ppg_values):.1f}%)")
            
            # Check time range
            start, end = get_experiment_time_range(signals)
            duration = (end - start).total_seconds()
            print(f"   ✓ Experiment duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
            
            # Estimate windows
            n_windows = int((duration - 60) / 60)  # rough estimate
            print(f"   ✓ Estimated windows: ~{n_windows}")
            
        else:
            print("   ❌ No PPG data found!")
            
    except Exception as e:
        print(f"   ❌ Error loading subject: {e}")
        import traceback
        traceback.print_exc()

# Test 4: Quick RRest Test (if Matlab works)
print("\n4. Testing RRest Functionality...")
try:
    # Get a small PPG sample
    if ppg_df is not None and len(valid_values) > 1920:  # 30 seconds at 64 Hz
        test_ppg = valid_values[:1920]  # First 30 seconds
        
        print(f"   Testing with {len(test_ppg)} PPG samples (30s at 64 Hz)")
        
        # Try minimal RRest wrapper
        print("   Initializing Matlab engine for RRest...")
        eng = matlab.engine.start_matlab()
        eng.addpath(str(RREST_PATH), nargout=0)
        eng.addpath(str(RREST_PATH / 'Algorithms'), nargout=0)
        print("   ✓ Paths added to Matlab")
        
        # Simple configuration
        eng.eval("""
        up = struct();
        up.paths.root_folder = tempdir;
        up.al.key_components = {'extract_resp_sig', 'estimate_rr', 'fuse_rr'};
        up.al.options.extract_resp_sig = {'ppg_feat'};
        up.al.sub_components.ppg_feat = {'EHF', 'PDt', 'FPt', 'FMe', 'RS', 'ELF'};
        up.al.options.PDt = {'IMS'};
        up.al.options.FMe = {'am', 'fm', 'bw'};
        up.al.options.RS = {'cubB'};
        up.al.options.estimate_rr = {'CtO', 'CtA', 'PKS', 'ZeX', 'PZX'};
        up.al.options.fuse_rr = {'fus_mod'};
        up.al.sub_components.fus_mod = {'SFu'};
        up.paramSet.winLeng = 30;
        up.paramSet.winOverlap = 0;
        up.paramSet.rr_range = [4, 60];
        disp('✓ RRest configured');
        """, nargout=0)
        
        # Convert PPG to Matlab format
        ppg_matlab = matlab.double(test_ppg.tolist())
        eng.workspace['ppg_data'] = ppg_matlab
        eng.workspace['fs'] = 64.0
        
        print("   Running RRest on test data...")
        eng.eval("""
        data = struct();
        data.ppg.v = ppg_data;
        data.ppg.fs = fs;
        
        try
            results = RRest(data, up);
            if isfield(results, 'qual')
                fprintf('   ✓ RRest succeeded!\\n');
                fprintf('   RR estimates: ');
                disp(results.qual.rr_ests);
                test_success = 1;
            else
                fprintf('   ⚠ RRest ran but no qual field\\n');
                test_success = 0;
            end
        catch ME
            fprintf('   ❌ RRest failed: %s\\n', ME.message);
            test_success = 0;
        end
        """, nargout=0)
        
        test_success = eng.workspace['test_success']
        if test_success == 1:
            print("   ✓ RRest test PASSED!")
        else:
            print("   ❌ RRest test FAILED!")
        
        eng.quit()
        
except Exception as e:
    print(f"   ❌ RRest test error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE - Check results above")
print("="*70)

