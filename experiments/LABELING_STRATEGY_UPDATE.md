# Labeling Strategy Update: Period Overlap vs Onset Detection

**Date:** 2025-12-21  
**Impact:** Critical - Changes prediction task definition  
**Files Modified:** `experiments/shared/windowing.py`

## Problem Statement

The original labeling strategy only detected when **NEW stress events would START**, not whether a person would **BE stressed** during the prediction window.

### Original Strategy (ONSET-BASED)
```
Question: "Will a NEW stress event START in the next 5 minutes?"
Method: Check if stress onset falls within [window_end, window_end + 5min]
```

### New Strategy (PERIOD-BASED)  
```
Question: "Will I BE stressed ANYTIME in the next 5 minutes?"
Method: Check if [window_end, window_end + 5min] OVERLAPS with any stress period
```

---

## The Issue

### Example Timeline
```
Stress period: 15:26:00 - 15:31:00 (5 minutes of cognitive task)

Window A: Input 15:22-15:24, Predict 15:24-15:29
├─ Prediction window overlaps with stress (15:26-15:29)
├─ OLD label: 1 ✓ (onset at 15:26 is in prediction window)
└─ NEW label: 1 ✓ (prediction window overlaps with stress period)
Status: ✅ Both methods agree

Window B: Input 15:27-15:29, Predict 15:29-15:34
├─ Input is DURING active stress!
├─ Prediction window overlaps with stress (15:29-15:31)
├─ OLD label: 0 ✗ (no NEW onset - stress already started)
└─ NEW label: 1 ✓ (will continue being stressed for 2 more minutes)
Status: ❌ Methods disagree - OLD is WRONG for our use case!
```

**The problem:** When already experiencing stress, the OLD method labeled windows as 0 (no stress) because no NEW stress event was starting. But the person IS stressed and WILL CONTINUE being stressed!

---

## Solution: Overlap Detection

### Mathematical Definition

For a prediction window `[t_end, t_end + horizon]` and stress period `[t_start, t_stop]`:

**Overlap exists if:**
```
overlap_start = max(t_end, t_start)
overlap_end = min(t_end + horizon, t_stop)

if overlap_start < overlap_end:
    label = 1  # Will be stressed during prediction window
else:
    label = 0  # No stress during prediction window
```

### Code Implementation

```python
# NEW: Period-based labeling
for horizon in horizons_minutes:
    horizon_end = window_end + timedelta(minutes=horizon)
    
    label = 0
    for start, stop, _ in emotional_stress_periods:
        overlap_start = max(window_end, start)
        overlap_end = min(horizon_end, stop)
        
        if overlap_start < overlap_end:
            label = 1
            break
    
    labels[f"label_{horizon}min"] = label
```

---

## Impact on Training Data

### Label Distribution Changes

**Before (Onset-based):**
```
Only windows BEFORE stress onset get label=1
Windows during stress get label=0
→ Fewer positive samples
→ Model only learns PRE-stress patterns
```

**After (Period-based):**
```
Windows BEFORE stress onset get label=1
Windows DURING stress (with future stress) get label=1
Windows AFTER stress get label=0
→ More positive samples
→ Model learns BOTH pre-stress AND during-stress patterns
```

### Example Subject Timeline

```
15:12 ────── 15:26 ────── 15:31 ────── 15:40 ────── 15:48
    Baseline    │ Cognitive │   Rest    │  Physical │  Rest
                │   Stress  │           │   Stress  │
                └───────────┘           └───────────┘
                 Emotional=1            Physical=0 (exercise)

Windows with 5-minute prediction horizon:

Time Range        │ OLD Label │ NEW Label │ Reason
─────────────────┼───────────┼───────────┼──────────────────────────
15:20-15:22      │     1     │     1     │ Onset at 15:26 in pred window
15:23-15:25      │     1     │     1     │ Onset at 15:26 in pred window
15:24-15:26      │     1     │     1     │ Onset at 15:26 at boundary
15:25-15:27      │     0     │     1     │ ✓ NEW: Overlaps 15:27-15:31
15:27-15:29      │     0     │     1     │ ✓ NEW: Overlaps 15:29-15:31
15:29-15:31      │     0     │     0     │ Both: Stress ending at 15:31
15:31-15:33      │     0     │     0     │ Both: No stress ahead
```

**Result:** 2 additional positive windows (15:25-15:27, 15:27-15:29)

---

## Benefits

### 1. **Realistic Prediction Task**
```
User asks: "Will I be stressed in the next 5 minutes?"
- If stress starting soon → YES ✓
- If already stressed → YES ✓ (will continue)
- If stress just ended → NO ✓
```

### 2. **Continuous Monitoring**
The model can provide predictions throughout the stress episode, not just before it starts.

### 3. **Richer Training Signal**
```
Pre-stress patterns:
- Subtle HRV changes
- Anticipatory arousal
- Baseline → transition

During-stress patterns:
- Active stress response
- Peak physiological changes
- Sustained arousal

Both are valuable for prediction!
```

### 4. **Better HRV Utilization**
```
Window during stress (input: 15:27-15:29):
- Low HRV (RMSSD reduced)
- High HR (85-90 bpm)
- Elevated EDA
- Label: 1 (will continue being stressed)

This teaches: "When you see ACTIVE stress markers,
               stress will persist in near future"
```

---

## Validation

### Test Cases

```python
# Test 1: Stress period 15:26-15:31
window_end = Timestamp('15:24:00')
horizon = 5  # minutes
# Prediction: 15:24-15:29
# Overlap with 15:26-15:29 (3 minutes)
# Expected: label=1 ✓

# Test 2: During stress
window_end = Timestamp('15:28:00')
horizon = 5  # minutes
# Prediction: 15:28-15:33
# Overlap with 15:28-15:31 (3 minutes)
# Expected: label=1 ✓

# Test 3: After stress
window_end = Timestamp('15:32:00')
horizon = 5  # minutes
# Prediction: 15:32-15:37
# No overlap (stress ended at 15:31)
# Expected: label=0 ✓

# Test 4: Stress starts at boundary
window_end = Timestamp('15:26:00')
horizon = 5  # minutes
# Prediction: 15:26-15:31
# Overlap with 15:26-15:31 (5 minutes)
# Expected: label=1 ✓
```

---

## Clinical Interpretation

### Prediction Horizons

**3-minute horizon:** "Immediate future"
- Can still take preventive action
- Most actionable for intervention
- High confidence needed

**5-minute horizon:** "Near future"  
- Early warning system
- Time for breathing exercises, breaks
- Balance of accuracy and utility

**10-minute horizon:** "Planning ahead"
- Long-term awareness
- Schedule adjustments possible
- Lower confidence acceptable

### Use Cases

1. **Pre-stress warning**
   - Window: Before stress
   - Pattern: Subtle HRV changes
   - Action: "Take a break before the meeting"

2. **During-stress awareness**
   - Window: During stress  
   - Pattern: Active stress response
   - Action: "You're stressed and it will continue for 2 more minutes"

3. **Recovery detection**
   - Window: After stress
   - Pattern: Returning to baseline
   - Action: "Stress ended, entering recovery"

---

## Migration Notes

### Backwards Compatibility

**Impact on existing experiments:**
- ❌ Labels will change for some windows
- ❌ Models need retraining
- ✓ Feature extraction unchanged
- ✓ Window creation logic unchanged

### What to Update

1. **Retrain all models** with new labels
2. **Update documentation** to reflect new task
3. **Re-evaluate baselines** with new label distribution
4. **Update performance metrics** interpretation

### Expected Changes

- **Positive class frequency:** Will increase
- **Model performance:** May improve (more training signal)
- **Prediction interpretation:** More intuitive for end users

---

## Summary

| Aspect | OLD (Onset) | NEW (Overlap) |
|--------|-------------|---------------|
| **Question** | When will stress START? | Will I BE stressed? |
| **Method** | Check onset in window | Check period overlap |
| **Pre-stress** | label=1 ✓ | label=1 ✓ |
| **During stress** | label=0 ✗ | label=1 ✓ |
| **After stress** | label=0 ✓ | label=0 ✓ |
| **Positive samples** | Fewer | More |
| **Clinical utility** | Limited | High |
| **Realistic task** | No | Yes ✓ |

**Conclusion:** The period-based overlap strategy better matches the real-world prediction task and provides more useful training signal for the model.

---

## References

- VitaStress dataset documentation
- HRV-based stress prediction literature
- Temporal overlap algorithms in time series labeling
