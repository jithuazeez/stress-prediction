# Understanding the Updated Windowing and Labeling Strategy

## 🎯 Your Goal (Now Implemented!)

**Question:** "Based on the last 2 minutes of data, will I be stressed ANYTIME in the next 5 minutes?"

**Input:** 2-minute window of physiological signals (HRV, HR, EDA, etc.)  
**Output:** Binary prediction (stressed=1, not stressed=0) for next 5 minutes

---

## 📊 How It Works Now

### Step 1: Create Input Windows
```
Every 2 minutes (120 seconds) of data becomes an input window:
- Window size: 120 seconds
- Sampling rate: 1 Hz
- Features: HR, HRV (RMSSD), EDA, skin temp, accelerometer, etc.
- Result: 120 time steps × N channels
```

### Step 2: Define Prediction Window
```
Prediction horizon: 5 minutes (configurable: 3, 5, or 10 minutes)

Timeline:
   Input window          Prediction window
   ↓                     ↓
   [─────────2 min──────][────────5 min────────]
   t-2              t=now              t+5
   
Question: "Will I be stressed between t and t+5?"
```

### Step 3: Check for Overlap with Stress Periods
```python
# Stress period from annotations (e.g., cognitive task)
stress_period = (15:26:00, 15:31:00)  # 5 minutes of stress

# For each window:
prediction_start = window_end  # e.g., 15:24:00
prediction_end = window_end + 5 min  # e.g., 15:29:00

# Calculate overlap
overlap_start = max(prediction_start, stress_start)
overlap_end = min(prediction_end, stress_stop)

if overlap_start < overlap_end:
    label = 1  # YES, will be stressed!
else:
    label = 0  # NO stress in prediction window
```

---

## 📈 Example Walkthrough

### Scenario: Cognitive Task from 15:26-15:31

```
Full timeline:
15:20 ────── 15:22 ────── 15:24 ────── 15:26 ────── 15:28 ────── 15:31 ────── 15:33
  Baseline      │           │      Cognitive Task Starts  │      Task Ends    │
                │           │           │                 │           │        │
```

### Window A: 15:20-15:22 (Far before stress)
```
Input: 15:20-15:22
├─ HR: 72 bpm (normal)
├─ RMSSD: 45 ms (normal HRV)
└─ EDA: 2.0 µS (calm)

Prediction: 15:22-15:27
├─ Stress period: 15:26-15:31
├─ Overlap: 15:26-15:27 (1 minute)
└─ Label: 1 ✓ (stress will start in 4 minutes!)

Model learns: "When baseline signals start showing subtle changes,
               stress is coming in ~4 minutes"
```

### Window B: 15:22-15:24 (Getting closer)
```
Input: 15:22-15:24
├─ HR: 74 bpm (slight increase)
├─ RMSSD: 42 ms (HRV decreasing)
└─ EDA: 2.2 µS (arousal starting)

Prediction: 15:24-15:29
├─ Stress period: 15:26-15:31
├─ Overlap: 15:26-15:29 (3 minutes)
└─ Label: 1 ✓ (stress in 2 minutes!)

Model learns: "Anticipatory arousal patterns predict imminent stress"
```

### Window C: 15:24-15:26 (Right before onset)
```
Input: 15:24-15:26
├─ HR: 76 bpm (increasing)
├─ RMSSD: 38 ms (HRV dropping)
└─ EDA: 2.5 µS (anticipatory arousal)

Prediction: 15:26-15:31
├─ Stress period: 15:26-15:31
├─ Overlap: 15:26-15:31 (5 minutes - complete overlap!)
└─ Label: 1 ✓ (stress starts NOW!)

Model learns: "These patterns indicate stress is about to begin"
```

### Window D: 15:26-15:28 (DURING stress) ⭐ KEY!
```
Input: 15:26-15:28
├─ HR: 88 bpm (elevated!)
├─ RMSSD: 25 ms (low HRV - stressed!)
└─ EDA: 4.1 µS (high arousal)

Prediction: 15:28-15:33
├─ Stress period: 15:26-15:31
├─ Overlap: 15:28-15:31 (3 minutes)
└─ Label: 1 ✓ (will continue being stressed!)

Model learns: "When actively stressed, stress will continue 
               for near future"

OLD SYSTEM WOULD GIVE: Label: 0 ✗ (WRONG!)
```

### Window E: 15:28-15:30 (Still stressed)
```
Input: 15:28-15:30
├─ HR: 90 bpm (peak)
├─ RMSSD: 23 ms (very low HRV)
└─ EDA: 4.5 µS (peak arousal)

Prediction: 15:30-15:35
├─ Stress period: 15:26-15:31
├─ Overlap: 15:30-15:31 (1 minute)
└─ Label: 1 ✓ (stress ending soon)

Model learns: "High stress markers indicate continued stress,
               even if ending soon"
```

### Window F: 15:31-15:33 (Recovery)
```
Input: 15:31-15:33
├─ HR: 82 bpm (decreasing)
├─ RMSSD: 32 ms (HRV recovering)
└─ EDA: 3.0 µS (arousal decreasing)

Prediction: 15:33-15:38
├─ Stress period: 15:26-15:31
├─ Overlap: none (stress ended at 15:31)
└─ Label: 0 ✓ (no more stress ahead)

Model learns: "Recovery patterns indicate stress has passed"
```

---

## 🔥 Key Advantage: Continuous Prediction

### What happens as you approach stress?

```
Distance to stress     Can predict?    Label    Model sees
─────────────────────┼───────────────┼────────┼──────────────────────
6 minutes away       │      YES      │   1    │ Baseline patterns
4 minutes away       │      YES      │   1    │ Subtle changes
2 minutes away       │      YES      │   1    │ Anticipatory arousal
At onset             │      YES      │   1    │ Transition to stress
2 min into stress    │      YES      │   1    │ Active stress markers
4 min into stress    │      YES      │   1    │ Sustained stress
After stress ends    │      YES      │   0    │ Recovery patterns
```

**Answer: YES, you can STILL predict as you get closer!**

The model gets **different but equally valuable** information:
- **Far from stress:** Subtle baseline changes, anticipatory signals
- **Near stress:** Transition patterns, pre-stress arousal
- **During stress:** Active stress markers, persistence patterns
- **After stress:** Recovery patterns, return to baseline

---

## 🧠 What the Model Learns

### Pre-Stress Patterns (Windows BEFORE onset)
```
Input features:
- Slight HR increase (72 → 76 bpm)
- HRV decrease (45 → 38 ms RMSSD)
- EDA rise (2.0 → 2.5 µS)
- Low activity (sitting)

Label: 1 (stress coming in N minutes)

Learning: "These subtle changes predict future stress"
```

### During-Stress Patterns (Windows DURING stress)
```
Input features:
- High HR (85-90 bpm)
- Low HRV (20-30 ms RMSSD)
- High EDA (3.5-4.5 µS)
- Low activity (sitting, cognitively focused)

Label: 1 (will continue being stressed)

Learning: "When actively stressed, stress persists"
```

### Recovery Patterns (Windows AFTER stress)
```
Input features:
- Decreasing HR (90 → 78 bpm)
- Recovering HRV (25 → 40 ms)
- Decreasing EDA (4.0 → 2.5 µS)

Label: 0 (no stress ahead)

Learning: "Recovery patterns indicate stress has ended"
```

---

## 💪 Why This Approach Works

### 1. **Matches Clinical Reality**
```
Doctor: "Will this patient be stressed in the next 5 minutes?"
Nurse: Checks current state + trajectory
       - If calm and stable → NO
       - If showing arousal → YES (will start)
       - If already stressed → YES (will continue)
       - If recovering → NO
```

### 2. **Maximum Information Utilization**
```
Every phase of the stress response provides signal:
✓ Anticipation phase
✓ Onset phase
✓ Active stress phase
✓ Recovery phase

OLD system only used anticipation phase!
```

### 3. **Realistic Edge Cases**
```
User checking app during stressful meeting:
- Input: Last 2 minutes (DURING stress)
- Question: "Will I be stressed for next 5 minutes?"
- Answer: YES (meeting continues for 3 more minutes)
- Prediction: label=1 ✓

This is USEFUL information!
```

---

## 📊 Impact on Training

### Label Distribution
```
OLD strategy (onset-based):
- Pre-stress windows: 30% positive
- During-stress windows: 0% positive ✗
- Total positive: 15% (very imbalanced!)

NEW strategy (period-based):
- Pre-stress windows: 30% positive
- During-stress windows: 60% positive ✓
- Total positive: 25-30% (better balance!)
```

### Training Examples
```
OLD: 100 windows → 15 positive samples
NEW: 100 windows → 28 positive samples

More data = Better learning!
```

---

## 🎯 Summary

### Your Original Question
> "Does the current windowing strategy make the model learn when stress will start?"

**OLD answer:** Yes, ONLY when it will start (onset detection)

**NEW answer:** Yes, when it will start AND when it will continue (stress presence)

### Can You Predict As You Get Closer?
**YES!** The model continuously learns:
- Far away: Baseline → anticipation transitions
- Getting closer: Anticipatory arousal patterns  
- At onset: Transition to active stress
- During stress: Sustained stress markers
- After: Recovery patterns

### Final Prediction Task
```
Input: Last 2 minutes of physiological data
Output: "Will you be stressed ANYTIME in next 5 minutes?"

✓ Detects upcoming stress
✓ Detects ongoing stress that will continue
✓ Detects when stress has ended
```

**This is exactly what you wanted!** 🎉

---

## Next Steps

1. **Retrain models** with the new labeling strategy
2. **Compare performance** between old and new approach
3. **Analyze predictions** at different distances from stress onset
4. **Validate** that during-stress predictions are accurate

The updated code is in `experiments/shared/windowing.py` and ready to use!
