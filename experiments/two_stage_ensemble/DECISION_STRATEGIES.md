# Two-Stage Ensemble Decision Strategies

This document explains the two available decision strategies for the LR+TCN ensemble model.

---

## Strategy A: **LR-Only Decision + TCN Confidence** (Default)

### Goal
**Maximize recall** - catch all stress events, accept some false positives

### How It Works

```
┌─────────────┐
│     LR      │ Makes decision based on probability threshold
│  (Screener) │ Optimized for high recall (e.g., min 90%)
└──────┬──────┘
       │
       ├─── LR_proba ≥ threshold → PREDICT: STRESS ✓
       │                           CONFIDENCE: TCN_proba
       │
       └─── LR_proba < threshold  → PREDICT: NO STRESS ✗
                                    CONFIDENCE: 1 - TCN_proba

┌─────────────┐
│     TCN     │ Provides confidence scores only
│ (Confidence)│ NEVER vetoes LR's stress detections
└─────────────┘
```

### Key Points
- ✅ **LR makes ALL decisions** - threshold set for high recall (90%+)
- ✅ **TCN provides confidence** - helps interpret/rank predictions
- ✅ **No stress is vetoed** - if LR says stress, it's stress
- ✅ **Best for recall** - catches maximum stress events
- ⚠️ **Lower precision** - will have more false alarms

### When to Use
- Your priority is **not missing any stress events**
- You can tolerate false positives (better safe than sorry)
- You want confidence scores for post-processing/ranking
- Clinical/safety-critical applications

### Configuration
```python
results = loso_cross_validation(
    windows_by_subject,
    config,
    device,
    logger,
    min_recall_lr=0.90,      # LR must catch 90% of stress
    max_far_tcn=0.25,        # Not used in this mode
    n_epochs_tcn=100,
    decision_strategy="lr_only"  # ← This activates Strategy A
)
```

### Output
- **Predictions**: Binary (0 or 1) from LR only
- **Confidence**: Float [0, 1] from TCN
  - For stress predictions: confidence = TCN agreement (high = confident)
  - For no-stress predictions: confidence = 1 - TCN_proba (high = confident)

---

## Strategy B: **Hard AND Cascade**

### Goal
**Minimize false alarms** - only predict stress when both models agree

### How It Works

```
┌─────────────┐
│     LR      │ Stage 1: Screening
│  (Screener) │ Catches potential stress candidates
└──────┬──────┘
       │
       ├─── LR_proba < threshold → PREDICT: NO STRESS ✗
       │                           (Skip TCN)
       │
       └─── LR_proba ≥ threshold → Go to Stage 2
                                    ↓
                        ┌─────────────────┐
                        │      TCN        │ Stage 2: Confirmation
                        │  (Confirmation) │ Must agree for stress
                        └────────┬────────┘
                                 │
                                 ├─── TCN_proba ≥ threshold → PREDICT: STRESS ✓
                                 │
                                 └─── TCN_proba < threshold  → PREDICT: NO STRESS ✗
```

### Key Points
- ✅ **Both models must agree** - AND logic
- ✅ **High precision** - fewer false alarms
- ⚠️ **Lower recall** - TCN's poor recall will cause missed stress
- ⚠️ **TCN can veto LR** - even when LR correctly detects stress

### When to Use
- Your priority is **avoiding false alarms**
- You can tolerate missing some stress events
- You want high precision and specificity
- Resource-limited applications (reduce unnecessary interventions)

### Configuration
```python
results = loso_cross_validation(
    windows_by_subject,
    config,
    device,
    logger,
    min_recall_lr=0.75,      # LR screening threshold
    max_far_tcn=0.25,        # TCN must keep FAR under 25%
    n_epochs_tcn=100,
    decision_strategy="and_cascade"  # ← This activates Strategy B
)
```

### Output
- **Predictions**: Binary (0 or 1) requiring both LR AND TCN agreement
- **No confidence scores** - decision is binary

---

## Comparison Table

| Metric | LR-Only (A) | AND Cascade (B) |
|--------|-------------|-----------------|
| **Recall** | ⭐⭐⭐⭐⭐ Very High (driven by LR) | ⭐⭐ Low (limited by TCN) |
| **Precision** | ⭐⭐ Low (same as LR) | ⭐⭐⭐⭐ High (both must agree) |
| **Specificity** | ⭐⭐ Low (same as LR) | ⭐⭐⭐⭐⭐ Very High (TCN gates) |
| **False Alarms** | ⚠️ Higher | ✅ Lower |
| **Missed Stress** | ✅ Lower | ⚠️ Higher |
| **Confidence Scores** | ✅ Yes (from TCN) | ❌ No |
| **Complexity** | Simple | More complex |

---

## Recommended Strategy

### For Your Use Case: **Strategy A (LR-Only)** ✅

**Reason**: You stated "**catching all stress events is my priority**"

With Strategy A:
1. **LR's high recall (85-90%)** is fully preserved
2. **TCN provides interpretable confidence** for each prediction
3. **No true stress events are lost** due to TCN's poor recall
4. You can **post-filter by confidence** if needed (e.g., only show high-confidence alerts)

Example:
```
LR detects stress (proba=0.82)
├─ TCN confidence: 0.91 → ✅ High confidence stress
└─ Final: STRESS (confidence: 91%)

LR detects stress (proba=0.78)
├─ TCN confidence: 0.23 → ⚠️ Low confidence stress
└─ Final: STRESS (confidence: 23%) - might be false alarm
```

---

## How to Switch Between Strategies

**Step 1**: Edit `experiments/two_stage_ensemble/train.py`

**Step 2**: Find the `loso_cross_validation` call in `main()` (around line 625)

**Step 3**: Change the `decision_strategy` parameter:

```python
# For Strategy A (LR-Only):
decision_strategy="lr_only"

# For Strategy B (AND Cascade):
decision_strategy="and_cascade"
```

**Step 4**: Adjust thresholds based on strategy:

```python
# Strategy A (LR-Only) - maximize recall
min_recall_lr=0.90,  # Very high recall
max_far_tcn=0.25,    # Not used

# Strategy B (AND Cascade) - minimize FAR
min_recall_lr=0.75,  # Moderate recall
max_far_tcn=0.25,    # Strict FAR control
```

**Step 5**: Run training:
```bash
python experiments/two_stage_ensemble/train.py
```

---

## Understanding the Output

### Strategy A (LR-Only)

Training log will show:
```
Decision Strategy: lr_only
  LR: Makes all decisions (min recall=90%)
  TCN: Provides confidence scores only

Fold 1/17 (S1...):
  LR threshold: 0.2341 (Recall=0.920)
  TCN threshold: 0.6789 (not used for decision)
  Test metrics:
    Recall:     0.8750  ← High (from LR)
    Precision:  0.3214  ← Lower (from LR)
    ...
  TCN Confidence:
    Overall:    0.6234
    Stress:     0.5821 (n=84)  ← TCN agrees with LR 58% of time
    No-stress:  0.7142 (n=156) ← TCN agrees with LR 71% of time
```

### Strategy B (AND Cascade)

Training log will show:
```
Decision Strategy: and_cascade
  LR: Screening (min recall=75%)
  TCN: Confirmation gate (max FAR=25%)

Fold 1/17 (S1...):
  LR threshold: 0.3241 (Recall=0.820)
  TCN threshold: 0.6789 (FAR=0.198)
  Test metrics:
    Recall:     0.4375  ← Much lower (TCN vetoes many)
    Precision:  0.6786  ← Higher (both must agree)
    ...
```

---

## Summary

| Priority | Recommended Strategy | Expected Recall | Expected Precision |
|----------|---------------------|-----------------|-------------------|
| Catch all stress | **A: LR-Only** | 85-90% | 30-40% |
| Minimize false alarms | **B: AND Cascade** | 40-50% | 60-70% |

Choose **Strategy A** if you can't afford to miss stress events (your stated priority).

Choose **Strategy B** if false alarms are more costly than missed detections.

