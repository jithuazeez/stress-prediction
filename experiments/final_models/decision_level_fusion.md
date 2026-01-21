# Decision-Level Fusion Strategies

**Combining Classical ML and Deep Learning for Improved Stress Prediction**

---

## Table of Contents

1. [Overview](#1-overview)
2. [What is Decision-Level Fusion?](#2-what-is-decision-level-fusion)
3. [Why Decision-Level (Not Feature or Model-Level)?](#3-why-decision-level-not-feature-or-model-level)
4. [Fusion Strategy 1: Logical OR](#4-fusion-strategy-1-logical-or)
5. [Fusion Strategy 2: Cascade (Sequential Screening)](#5-fusion-strategy-2-cascade-sequential-screening)
6. [Fusion Strategy 3: Stacked Generalization](#6-fusion-strategy-3-stacked-generalization)
7. [Implementation Details](#7-implementation-details)
8. [Expected Trade-offs](#8-expected-trade-offs)
9. [Connection to Research Question 3](#9-connection-to-research-question-3)

---

## 1. Overview

### 1.1 Motivation

**Research Question 3 (RQ3):**
> *Can decision-level fusion strategies, such as logical operators and stacked generalisation, improve the balance between stress prediction recall and false alarm rate in subject-independent settings?*

**The Challenge:**
- **Logistic Regression (LR):** High interpretability, fast inference, but limited capacity
- **TCN (Deep Learning):** Complex temporal patterns, but black box
- **Goal:** Combine their complementary strengths to achieve better recall-FAR trade-off

**Hypothesis:**
- LR and TCN may make different types of errors
- LR may excel on certain physiological patterns (e.g., clear HRV drops)
- TCN may excel on complex temporal sequences (e.g., gradual autonomic shifts)
- **Fusion can leverage both strengths**

---

### 1.2 Fusion Approaches Evaluated

| Strategy | Type | Description | Complexity |
|----------|------|-------------|------------|
| **Logical OR** | Rule-based | Predict stress if **either** model predicts stress | Low |
| **Cascade** | Rule-based | LR screens → TCN confirms | Low |
| **Stacked Generalization** | Learned | Meta-model learns optimal combination | High |

All three strategies are applied at the **decision level** using probability outputs from base models.

---

## 2. What is Decision-Level Fusion?

### 2.1 Definition

**Decision-level fusion** combines predictions from multiple independent models **after** they have each made their individual decisions.

**Key Characteristics:**
1. **Base models trained independently** (LR and TCN don't see each other's features)
2. **Fusion operates on outputs** (probabilities, not internal representations)
3. **Late fusion** (happens at final decision stage)

### 2.2 Fusion Pipeline

```
Raw Data (480 timesteps × 8 channels)
   │
   ├─────────────────────┬─────────────────────┐
   ↓                     ↓                     ↓
Feature Extraction    Keep Raw           Keep Raw
(39 features)         Sequences          Sequences
   ↓                     ↓                     ↓
Logistic             TCN                (Other models...)
Regression           Model
   ↓                     ↓                     ↓
P(stress|LR)         P(stress|TCN)       P(stress|...)
= 0.65               = 0.72              = ...
   │                     │                     │
   └─────────────────────┴─────────────────────┘
                         ↓
                  FUSION LAYER
              (OR / Cascade / Stacked)
                         ↓
                 Final Prediction
                  P(stress) or {0, 1}
```

**Inputs to Fusion:**
- `lr_proba`: Probability of stress from Logistic Regression (range: [0, 1])
- `tcn_proba`: Probability of stress from TCN (range: [0, 1])

**Outputs from Fusion:**
- **Rule-based (OR, Cascade):** Binary prediction {0, 1}
- **Stacked:** Meta-model probability [0, 1] → thresholded to {0, 1}

---

## 3. Why Decision-Level (Not Feature or Model-Level)?

### 3.1 Fusion Level Comparison

#### **Option A: Feature-Level Fusion** (NOT USED)

```
Features from LR (39) + Raw sequences from TCN (8×480)
   ↓
Concatenated feature vector (39 + 3,840 = 3,879 dimensions)
   ↓
Single unified model
```

**Why NOT used:**
- ❌ Incompatible representations: 39 scalars vs. 3,840 temporal sequences
- ❌ Dimensionality mismatch: How to meaningfully combine?
- ❌ Loses model independence: Can't leverage pre-trained models
- ❌ More complex training: Single massive model to train

#### **Option B: Model-Level Fusion (Ensemble Learning)** (NOT USED)

```
Average predictions:
P_final = (P_LR + P_TCN) / 2
```

**Why NOT used:**
- ❌ Naive averaging: Assumes equal model quality
- ❌ No adaptability: Fixed weights (0.5, 0.5)
- ❌ Ignores complementary strengths: Treats all predictions equally

#### **Option C: Decision-Level Fusion** (OUR APPROACH ✓)

```
LR outputs: P(stress|LR) = 0.65
TCN outputs: P(stress|TCN) = 0.72
   ↓
Fusion strategy (OR / Cascade / Stacked)
   ↓
Final decision: stress or no-stress
```

**Why Decision-Level:**
- ✅ **Modularity:** Base models trained independently, can be updated separately
- ✅ **Flexibility:** Multiple fusion strategies easily tested
- ✅ **Interpretability:** Can analyze each model's contribution
- ✅ **Complementarity:** Leverages different error patterns
- ✅ **Practical:** Works with pre-trained models, standard approach in literature

---

### 3.2 Advantages of Decision-Level Fusion

**1. Model Independence:**
- LR and TCN use completely different representations
- LR: 39 hand-crafted features (statistical aggregates)
- TCN: 8×480 raw sequences (temporal patterns)
- Decision-level fusion doesn't require reconciling these representations

**2. Error Diversity:**
```
Example window (subject 5, window 23):

LR prediction: 0.45 (no stress) ← Misses subtle temporal pattern
TCN prediction: 0.78 (stress)   ← Catches gradual HRV decline
Fusion (OR): STRESS ✓ (corrects LR's false negative)

Example window (subject 12, window 67):

LR prediction: 0.82 (stress)    ← Clear HRV drop + fidgeting
TCN prediction: 0.52 (no stress) ← Noisy temporal sequence
Fusion (Cascade): STRESS ✓ (LR's confidence dominates)
```

**Models make different mistakes → fusion can improve!**

**3. Computational Efficiency:**
- Base models already trained (no retraining needed)
- Fusion is lightweight (rule-based or simple meta-LR)
- Inference: Run both models in parallel, combine outputs

**4. Transparency:**
- Can inspect individual model contributions
- Logical rules (OR, Cascade) are fully interpretable
- Even stacked meta-model is simple (just 2 features: LR prob, TCN prob)

---

## 4. Fusion Strategy 1: Logical OR

### 4.1 Concept

**Prediction Logic:**
```
Predict STRESS if:
  (LR probability ≥ LR threshold) OR (TCN probability ≥ TCN threshold)

Otherwise predict NO STRESS
```

**Intuition:** "Trust either model's stress detection"

### 4.2 Mathematical Formulation

```python
def logical_or_fusion(lr_proba, tcn_proba, lr_thr, tcn_thr):
    """
    Logical OR: Predict stress if EITHER model predicts stress.
    
    Args:
        lr_proba: LR probability outputs (n_samples,)
        tcn_proba: TCN probability outputs (n_samples,)
        lr_thr: Threshold for LR (e.g., 0.45)
        tcn_thr: Threshold for TCN (e.g., 0.52)
    
    Returns:
        Binary predictions (n_samples,) - {0, 1}
    """
    lr_pred = (lr_proba >= lr_thr).astype(int)    # LR's decision
    tcn_pred = (tcn_proba >= tcn_thr).astype(int) # TCN's decision
    
    # OR: stress if either predicts stress
    return (lr_pred | tcn_pred).astype(int)
```

### 4.3 When Does OR Fusion Help?

**Scenario 1: LR Catches, TCN Misses**
```
Window 142 (subject 7):
  LR: 0.75 → STRESS (clear HRV drop)
  TCN: 0.42 → No stress (sequence too noisy)
  OR: STRESS ✓ (LR saves the prediction)
```

**Scenario 2: TCN Catches, LR Misses**
```
Window 89 (subject 3):
  LR: 0.38 → No stress (aggregate features look normal)
  TCN: 0.81 → STRESS (subtle gradual autonomic shift)
  OR: STRESS ✓ (TCN saves the prediction)
```

**Scenario 3: Both Agree (No Stress)**
```
Window 205 (subject 14):
  LR: 0.22 → No stress
  TCN: 0.31 → No stress
  OR: No stress ✓ (high confidence no stress)
```

### 4.4 Expected Behavior

**Recall (Sensitivity):**
- ✅ **Maximized:** If **either** model detects stress, predict stress
- Any true stress window caught by LR **or** TCN is correctly labeled
- Formula: `Recall_OR ≥ max(Recall_LR, Recall_TCN)`

**False Alarm Rate:**
- ❌ **Increased:** More liberal (predicts stress more often)
- False alarm if **either** model makes a false alarm
- Formula: `FAR_OR ≥ max(FAR_LR, FAR_TCN)` (worst case)

**Trade-off:**
- Prioritizes **high recall** (catch all stress)
- Accepts **higher FAR** (more false alarms)
- Suitable for safety-critical applications (better false alarm than missed stress)

---

### 4.5 Threshold Selection for OR Fusion

**Per-Fold Procedure (LOSO):**

```python
# For each test subject:
for test_subject in subjects:
    # Split: hold out test subject
    train_data = all_data[subjects != test_subject]
    test_data = all_data[subjects == test_subject]
    
    # Find optimal thresholds on TRAIN data
    lr_thr = find_optimal_threshold(train_lr_proba, train_y, method="gmean")
    tcn_thr = find_optimal_threshold(train_tcn_proba, train_y, method="gmean")
    
    # Apply OR fusion with thresholds to TEST data
    test_pred = (test_lr_proba >= lr_thr) | (test_tcn_proba >= tcn_thr)
    
    # Evaluate on test subject
    evaluate(test_y, test_pred)
```

**Three Threshold Strategies (Ablation B):**

| Strategy | LR Threshold | TCN Threshold | Goal |
|----------|-------------|---------------|------|
| **B1** | Unconstrained G-Mean | Unconstrained G-Mean | Balanced recall-specificity |
| **B2** | Recall ≥ 70% | Recall ≥ 70% | High recall (safety) |
| **B3** | FAR ≤ 30% | FAR ≤ 30% | Low FAR (usability) |

---

## 5. Fusion Strategy 2: Cascade (Sequential Screening)

### 5.1 Concept

**Two-Stage Decision:**

```
Stage 1 (LR Screening):
  If LR probability < LR threshold:
    → Predict NO STRESS (stop)
  
  If LR probability ≥ LR threshold:
    → Proceed to Stage 2

Stage 2 (TCN Confirmation):
  If TCN probability ≥ TCN threshold:
    → Predict STRESS
  Else:
    → Predict NO STRESS
```

**Intuition:** "LR screens, TCN confirms"

### 5.2 Mathematical Formulation

```python
def cascade_fusion(lr_proba, tcn_proba, lr_thr, tcn_thr):
    """
    Cascade: LR screens, TCN confirms.
    
    Stage 1: If LR says "no stress" → final decision = no stress
    Stage 2: If LR says "stress" → check TCN for confirmation
    
    Args:
        lr_proba: LR probability outputs
        tcn_proba: TCN probability outputs
        lr_thr: LR threshold (screener)
        tcn_thr: TCN threshold (confirmer)
    
    Returns:
        Binary predictions {0, 1}
    """
    pred = np.zeros(len(lr_proba), dtype=int)  # Default: no stress
    
    # Stage 1: LR screening
    lr_positive = (lr_proba >= lr_thr)
    
    # Stage 2: For LR positives, check TCN
    pred[lr_positive] = (tcn_proba[lr_positive] >= tcn_thr).astype(int)
    
    return pred
```

### 5.3 When Does Cascade Help?

**Design Philosophy:**
- **LR (Fast, Interpretable):** First-line screener, filters obvious cases
- **TCN (Powerful, Complex):** Second-line confirmer, for uncertain cases

**Scenario 1: LR Filters Out Clear Negatives**
```
Window 312 (subject 18):
  LR: 0.18 → No stress (very confident, skip TCN)
  TCN: Not evaluated (cascade stops at LR)
  Cascade: No stress ✓
  
Benefit: Saves TCN computation for 60-70% of windows!
```

**Scenario 2: LR Uncertain → TCN Confirms**
```
Window 95 (subject 4):
  LR: 0.68 → Potential stress (proceed to TCN)
  TCN: 0.82 → STRESS (confirms LR's suspicion)
  Cascade: STRESS ✓
```

**Scenario 3: LR False Alarm → TCN Corrects**
```
Window 201 (subject 11):
  LR: 0.71 → Potential stress (proceed to TCN)
  TCN: 0.39 → No stress (rejects LR's alarm)
  Cascade: No stress ✓ (TCN filters false alarm)
```

### 5.4 Expected Behavior

**Recall:**
- ✓ **Lower than OR** (two-stage filter is stricter)
- Both LR **and** TCN must agree for stress prediction
- Formula: `Recall_Cascade ≤ min(Recall_LR, Recall_TCN)`

**False Alarm Rate:**
- ✅ **Lower than OR** (TCN filters LR's false alarms)
- Must pass both LR screening **and** TCN confirmation
- Formula: `FAR_Cascade ≤ FAR_LR` (TCN acts as quality gate)

**Trade-off:**
- Prioritizes **low FAR** (reduce false alarms)
- Accepts **lower recall** (may miss some stress)
- Suitable for user-facing applications (avoid alert fatigue)

**Computational Efficiency:**
- LR is fast (~1ms per window)
- If LR filters 70% of windows → skip TCN 70% of the time
- Reduces average inference cost

---

### 5.5 Asymmetric Role Assignment

**Why LR as Screener (Not TCN)?**

```
Option A: LR screens → TCN confirms (OUR CHOICE ✓)
  Pros:
    - LR is 10× faster (1ms vs 10ms)
    - LR is interpretable (can explain screening decisions)
    - TCN adds value for complex cases

Option B: TCN screens → LR confirms
  Cons:
    - TCN is slower (run on all windows)
    - TCN is black box (less transparent screening)
    - LR unlikely to add value after TCN
```

**Result:** LR's speed + interpretability make it ideal first stage.

---

## 6. Fusion Strategy 3: Stacked Generalization

### 6.1 Concept

**Meta-Learning Approach:**
```
Level 0 (Base Models):
  LR and TCN trained independently on raw data

Level 1 (Meta-Model):
  Meta-Logistic Regression learns to combine LR and TCN outputs
  
  Input: [P(stress|LR), P(stress|TCN)]
  Output: P(stress|Meta-LR)
```

**Intuition:** "Learn the optimal fusion rule from data"

### 6.2 Mathematical Formulation

**Base Models (Level 0):**
```
LR:  X (39 features) → P(stress|LR)
TCN: X (8×480 raw) → P(stress|TCN)
```

**Meta-Model (Level 1):**
```
Meta-LR: [P(stress|LR), P(stress|TCN)] → P(stress|Meta)

Logistic Regression on 2D input:
  P(stress|Meta) = σ(w₁·P(LR) + w₂·P(TCN) + b)
  
Where:
  σ(z) = 1 / (1 + exp(-z))  [sigmoid function]
  w₁, w₂: Learned weights for LR and TCN
  b: Learned bias term
```

**What Meta-LR Learns:**
- **Relative confidence:** When to trust LR vs. TCN
- **Interaction patterns:** How probabilities combine (not just average!)
- **Error correction:** Down-weight unreliable predictions

### 6.3 Training Procedure: Nested LOSO

**Critical:** Must avoid data leakage during meta-model training!

**Standard Stacking (WRONG for our case):**
```
1. Train LR and TCN on all training data
2. Use their predictions on training data to train meta-model
❌ Problem: Meta-model sees predictions on data used to train base models
❌ Result: Overfitting, optimistic bias
```

**Nested LOSO (CORRECT for subject-independent evaluation):**

```python
# Outer LOSO loop (same as base models)
for test_subject in all_subjects:
    
    # Hold out test subject
    train_subjects = all_subjects - {test_subject}
    
    # Base model probabilities already computed (from base model training)
    # These are OUT-OF-FOLD predictions for train subjects
    lr_proba_train = load_oof_predictions("LR", train_subjects)
    tcn_proba_train = load_oof_predictions("TCN", train_subjects)
    y_train = load_labels(train_subjects)
    
    # Train meta-model on OUT-OF-FOLD predictions
    X_meta_train = np.column_stack([lr_proba_train, tcn_proba_train])
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(X_meta_train, y_train)
    
    # Test on held-out subject
    lr_proba_test = load_predictions("LR", test_subject)
    tcn_proba_test = load_predictions("TCN", test_subject)
    X_meta_test = np.column_stack([lr_proba_test, tcn_proba_test])
    
    # Meta-model prediction
    meta_proba_test = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # Apply 3 threshold strategies
    for strategy in ["B1", "B2", "B3"]:
        threshold = find_optimal_threshold(meta_proba_train, y_train, ...)
        meta_pred = (meta_proba_test >= threshold).astype(int)
        evaluate(y_test, meta_pred)
```

**Key Safeguards:**
1. ✅ Meta-model trained on **out-of-fold** base predictions (no leakage)
2. ✅ Test subject completely unseen by meta-model
3. ✅ Same LOSO structure as base models (fair comparison)

---

### 6.4 What Can Meta-LR Learn?

**Example Learned Weights:**

```python
# Hypothetical meta-model after training:
meta_model.coef_ = [1.2, 0.8]   # [weight for LR, weight for TCN]
meta_model.intercept_ = -0.5

Interpretation:
  - LR weight (1.2) > TCN weight (0.8)
  - Meta-model trusts LR slightly more than TCN
  - Negative bias (-0.5): Conservative (leans toward no-stress)
```

**Learned Combination Patterns:**

```
Case 1: Both models agree (high confidence)
  LR=0.85, TCN=0.88 → Meta=0.92 (amplifies agreement)

Case 2: Both models disagree (low confidence)
  LR=0.45, TCN=0.52 → Meta=0.38 (cautious, no stress)

Case 3: LR high, TCN low (trust LR's expertise)
  LR=0.82, TCN=0.48 → Meta=0.71 (follows LR)

Case 4: LR low, TCN high (trust TCN's temporal analysis)
  LR=0.41, TCN=0.79 → Meta=0.68 (follows TCN)
```

**Meta-model learns these patterns automatically from training data!**

---

### 6.5 Expected Behavior

**Flexibility:**
- ✅ Adapts to base model strengths
- Can learn arbitrary weighting (not fixed like 0.5/0.5 average)
- Can learn interaction effects

**Performance:**
- ✓ **Often best recall-FAR balance**
- Combines strengths of both models optimally
- More sophisticated than hard-coded rules (OR, Cascade)

**Drawbacks:**
- ❌ Requires training (unlike rule-based methods)
- ❌ Less interpretable (learned weights may be hard to explain)
- ❌ Risk of overfitting if not properly validated (hence nested LOSO)

---

## 7. Implementation Details

### 7.1 Fusion Inputs

**Base Model Probabilities:**

All fusion strategies use the **probability outputs** from LR and TCN:

```python
# After base model training (LOSO):
lr_predictions.csv:
  subject | y_true | y_proba | y_pred
  --------|--------|---------|-------
  sub_01  |   0    |  0.23   |   0
  sub_01  |   1    |  0.78   |   1
  ...

tcn_predictions.csv:
  subject | y_true | y_proba | y_pred
  --------|--------|---------|-------
  sub_01  |   0    |  0.31   |   0
  sub_01  |   1    |  0.82   |   1
  ...
```

**Loading for Fusion:**
```python
# From train_fusion.py lines 329-343:
lr_pred = load_predictions("lr", "b1", results_dir)
tcn_pred = load_predictions("tcn", "b1", results_dir)

# Verify alignment (same windows, same order)
assert all(lr_pred["subjects"] == tcn_pred["subjects"])
assert all(lr_pred["y_true"] == tcn_pred["y_true"])

# Extract probability arrays
lr_proba = lr_pred["y_proba"].values   # (n_windows,)
tcn_proba = tcn_pred["y_proba"].values # (n_windows,)
y_true = lr_pred["y_true"].values
subjects = lr_pred["subjects"].values
```

---

### 7.2 Validation Protocol

**All fusion strategies use LOSO:**

```python
for test_subject in unique_subjects:
    # Split
    train_mask = (subjects != test_subject)
    test_mask = (subjects == test_subject)
    
    # Training data
    lr_proba_train = lr_proba[train_mask]
    tcn_proba_train = tcn_proba[train_mask]
    y_train = y_true[train_mask]
    
    # Test data
    lr_proba_test = lr_proba[test_mask]
    tcn_proba_test = tcn_proba[test_mask]
    y_test = y_true[test_mask]
    
    # Strategy-specific fusion
    if fusion == "OR":
        # Find thresholds on train, apply OR to test
        ...
    elif fusion == "Cascade":
        # Find thresholds on train, apply cascade to test
        ...
    elif fusion == "Stacked":
        # Train meta-LR on train, predict on test
        ...
    
    # Evaluate on test subject
    evaluate(y_test, predictions)
```

**Key Property:**
- Same 21 LOSO folds as base models
- Fair comparison across all approaches
- No subject seen during their own fold's fusion training

---

### 7.3 Threshold Strategies (Ablation B)

**All fusion methods apply 3 threshold strategies:**

```python
# B1: Unconstrained G-Mean
threshold_b1 = find_optimal_threshold(
    y_train, proba_train, 
    method="geometric_mean"
)

# B2: Recall-Constrained (≥70%)
threshold_b2 = find_optimal_threshold(
    y_train, proba_train,
    method="constrained_gmean",
    min_recall=0.70,
    max_fpr=1.0
)

# B3: FAR-Constrained (≤30%)
threshold_b3 = find_optimal_threshold(
    y_train, proba_train,
    method="constrained_gmean",
    min_recall=0.0,
    max_fpr=0.30
)
```

**Application:**

| Fusion | B1 Thresholds | B2 Thresholds | B3 Thresholds |
|--------|--------------|--------------|--------------|
| **OR** | LR_thr, TCN_thr (both G-Mean) | LR_thr, TCN_thr (both Recall≥70%) | LR_thr, TCN_thr (both FAR≤30%) |
| **Cascade** | LR_thr, TCN_thr (both G-Mean) | LR_thr, TCN_thr (both Recall≥70%) | LR_thr, TCN_thr (both FAR≤30%) |
| **Stacked** | Meta_thr (G-Mean on meta-proba) | Meta_thr (Recall≥70%) | Meta_thr (FAR≤30%) |

---

## 8. Expected Trade-offs

### 8.1 Recall vs. FAR Trade-off Spectrum

```
Recall (Sensitivity) vs. False Alarm Rate

High Recall, High FAR:
  ┌─────────────┐
  │ Logical OR  │ ← Most liberal (catch everything)
  └─────────────┘

Balanced:
  ┌─────────────┐
  │  Stacked    │ ← Learned balance
  └─────────────┘

  ┌─────────────┐
  │  Base Models│ ← LR, TCN individually
  └─────────────┘

Low Recall, Low FAR:
  ┌─────────────┐
  │  Cascade    │ ← Most conservative (reduce false alarms)
  └─────────────┘
```

---

### 8.2 Strategy-Specific Trade-offs

#### **Logical OR**

| Aspect | Expected Behavior | Rationale |
|--------|------------------|-----------|
| **Recall** | ⬆️ **Highest** | If **either** model detects stress → predict stress |
| **FAR** | ⬆️ **Highest** | More liberal, predicts stress more often |
| **Specificity** | ⬇️ **Lowest** | Fewer true negatives (more false alarms) |
| **Precision** | ⬇️ **Lower** | More false alarms reduce precision |
| **Use Case** | Safety-critical | Better false alarm than missed stress |

**When OR Improves:**
- Base models have **complementary errors** (LR catches A, TCN catches B)
- High cost of **false negatives** (missing stress is critical)
- Accept higher **alert fatigue** for better **coverage**

**When OR Hurts:**
- Base models make **correlated errors** (both wrong on same cases)
- OR amplifies **false alarms** from both models

---

#### **Cascade**

| Aspect | Expected Behavior | Rationale |
|--------|------------------|-----------|
| **Recall** | ⬇️ **Lower** | Must pass **both** LR and TCN filters |
| **FAR** | ⬇️ **Lower** | TCN filters LR's false alarms |
| **Specificity** | ⬆️ **Higher** | More true negatives (stricter criteria) |
| **Precision** | ⬆️ **Higher** | Fewer false alarms increase precision |
| **Use Case** | User-facing apps | Reduce alert fatigue, maintain usability |

**When Cascade Improves:**
- LR has **high recall but high FAR** (good screener, needs refinement)
- TCN can **filter false alarms** without losing true positives
- **Computational budget** limited (save TCN for uncertain cases)

**When Cascade Hurts:**
- LR **misses many true positives** (cascade stops too early)
- TCN **rejects true positives** flagged by LR

---

#### **Stacked Generalization**

| Aspect | Expected Behavior | Rationale |
|--------|------------------|-----------|
| **Recall** | ✓ **Balanced** | Learns optimal combination for G-mean |
| **FAR** | ✓ **Balanced** | Adaptive weighting of base models |
| **Flexibility** | ⬆️ **Highest** | Can learn arbitrary decision boundaries |
| **Interpretability** | ⬇️ **Lower** | Learned weights less transparent |
| **Use Case** | General-purpose | Best overall performance (if validated properly) |

**When Stacked Improves:**
- Base models have **clear patterns** in errors
- Sufficient **training data** for meta-model (20 subjects in LOSO train)
- Want **adaptive fusion** (not fixed rule)

**When Stacked Hurts:**
- **Overfitting** to training subjects (hence nested LOSO critical!)
- Small dataset: Meta-model may not generalize well

---

### 8.3 Comparative Performance Expectations

**Hypothetical Results (Subject to Empirical Validation):**

| Model | Recall | FAR | Specificity | G-Mean | Interpretation |
|-------|--------|-----|-------------|--------|----------------|
| **LR (Base)** | 0.68 | 0.32 | 0.68 | 0.68 | Balanced baseline |
| **TCN (Base)** | 0.72 | 0.28 | 0.72 | 0.72 | Slightly better than LR |
| **Logical OR** | 0.82 | 0.45 | 0.55 | 0.67 | High recall, high FAR |
| **Cascade** | 0.61 | 0.18 | 0.82 | 0.71 | Low FAR, lower recall |
| **Stacked** | 0.76 | 0.25 | 0.75 | 0.75 | Best balance ✓ |

**Key Insights:**
1. **OR:** Maximizes recall but increases FAR (trade-off accepted)
2. **Cascade:** Minimizes FAR but reduces recall (conservative)
3. **Stacked:** Best of both worlds (if meta-model generalizes)

**These are HYPOTHETICAL** - actual results depend on empirical evaluation!

---

### 8.4 Sensitivity to Threshold Strategies

**Impact of Ablation B on Fusion:**

```
Logical OR with B2 (Recall ≥ 70%):
  LR threshold lowered → more LR positives
  TCN threshold lowered → more TCN positives
  OR: Even more stress predictions
  Result: Very high recall (>80%), very high FAR (>40%)

Logical OR with B3 (FAR ≤ 30%):
  LR threshold raised → fewer LR positives
  TCN threshold raised → fewer TCN positives
  OR: Still more liberal than individual models
  Result: Moderate recall (~70%), moderate FAR (~35%)
```

**Fusion interacts with threshold strategy!**
- B2 (high recall) + OR → extremely liberal
- B3 (low FAR) + Cascade → extremely conservative

---

## 9. Connection to Research Question 3

### 9.1 RQ3 Revisited

**Research Question 3:**
> *Can decision-level fusion strategies, such as logical operators and stacked generalisation, improve the balance between stress prediction recall and false alarm rate in subject-independent settings?*

**Components to Investigate:**

1. **Fusion vs. Individual Models:**
   - Does OR/Cascade/Stacked outperform LR and TCN alone?
   - Under which threshold strategy (B1, B2, or B3)?

2. **Fusion Strategy Comparison:**
   - Does learned fusion (Stacked) beat rule-based (OR, Cascade)?
   - Are simple rules sufficient, or is meta-learning necessary?

3. **Recall-FAR Trade-off:**
   - Can fusion shift the Pareto frontier (better than any individual model)?
   - Which strategy achieves best balance (G-mean)?

4. **Subject Independence:**
   - Do fusion benefits generalize across unseen subjects (LOSO)?
   - Is meta-model overfitting to training subjects?

---

### 9.2 Evaluation Metrics

**Primary Metrics:**
- **Recall:** Proportion of true stress windows correctly identified
- **False Alarm Rate (FAR):** Proportion of no-stress windows incorrectly flagged
- **G-mean:** Geometric mean of recall and specificity (balanced metric)

**Secondary Metrics:**
- **AUROC, PR-AUC:** Threshold-independent performance
- **Precision:** Positive predictive value (for user trust)

**Statistical Analysis:**
- **Wilcoxon Signed-Rank Test:** Compare fusion vs. base models (paired across 21 folds)
- **Cliff's Delta:** Effect size of improvement

---

### 9.3 Expected Research Outcomes

**Hypothesis 1: OR Improves Recall**
- **Prediction:** OR fusion achieves higher recall than LR or TCN alone
- **Trade-off:** At cost of increased FAR
- **Validation:** Wilcoxon test on per-fold recall

**Hypothesis 2: Cascade Reduces FAR**
- **Prediction:** Cascade achieves lower FAR than LR or TCN alone
- **Trade-off:** At cost of reduced recall
- **Validation:** Wilcoxon test on per-fold FAR

**Hypothesis 3: Stacked Optimizes G-Mean**
- **Prediction:** Stacked meta-model achieves best G-mean (balanced performance)
- **Rationale:** Learns optimal LR-TCN combination
- **Validation:** Wilcoxon test on per-fold G-mean

**Hypothesis 4: Subject Independence Maintained**
- **Prediction:** Fusion benefits generalize across all 21 LOSO folds
- **Rationale:** Nested LOSO prevents meta-model overfitting
- **Validation:** Low std across folds, no outlier subjects

---

### 9.4 Practical Implications

**Deployment Scenario Selection:**

| Application | Recommended Fusion | Rationale |
|-------------|-------------------|-----------|
| **Emergency Alert System** | Logical OR (B2) | Maximize recall, accept false alarms |
| **Wellness App** | Cascade (B3) | Minimize false alarms, user trust |
| **Clinical Decision Support** | Stacked (B1) | Balanced accuracy, interpretable probabilities |
| **Research Tool** | All strategies + Ablation B | Explore full trade-off space |

**Interpretability Considerations:**
- **OR:** "Alert if either LR or TCN detects stress" (fully transparent)
- **Cascade:** "LR screens, TCN confirms" (clear two-stage logic)
- **Stacked:** "Learned combination" (requires explanation: "Meta-model trusts LR 60%, TCN 40%")

---

## 10. Summary

### 10.1 Key Takeaways

**Decision-Level Fusion:**
- ✅ Combines **independent models** at output stage (probabilities)
- ✅ **Modular** (base models trained separately, fusion added post-hoc)
- ✅ **Interpretable** (can analyze individual model contributions)

**Three Strategies:**

| Strategy | Type | Recall | FAR | Use Case |
|----------|------|--------|-----|----------|
| **Logical OR** | Rule-based | ⬆️ High | ⬆️ High | Safety-critical (maximize detection) |
| **Cascade** | Rule-based | ⬇️ Lower | ⬇️ Low | User-facing (minimize false alarms) |
| **Stacked** | Learned | ✓ Balanced | ✓ Balanced | General-purpose (optimal combination) |

**Validation:**
- ✅ All strategies use **LOSO** (subject-independent)
- ✅ **Nested LOSO** for stacked (no meta-model leakage)
- ✅ **3 threshold strategies** (Ablation B: B1, B2, B3)

**Expected Benefits:**
- **Complementary errors:** LR and TCN may excel on different cases
- **Flexible trade-offs:** Choose fusion strategy based on deployment constraints
- **Improved G-mean:** Learned fusion (Stacked) expected to optimize balance

---

### 10.2 Research Contribution

**RQ3 addresses:**
1. **Novelty:** Decision-level fusion for near-future stress prediction (not common in literature)
2. **Practical value:** Multiple deployment modes (safety vs. usability)
3. **Methodological rigor:** Nested LOSO ensures subject-independent validation
4. **Comparative analysis:** Three distinct strategies reveal trade-off space

**Next Steps:**
- Empirical evaluation on all 21 subjects
- Statistical significance testing (Wilcoxon, Cliff's delta)
- Qualitative analysis: Which subjects benefit most from fusion?

---

## 11. Code References

**Fusion Implementation:**
- **Main script:** `experiments/final_models/train_fusion.py`
- **OR fusion:** Lines 42-45
- **Cascade fusion:** Lines 48-55
- **Stacked meta-model:** Lines 58-148
- **Rule-based training:** Lines 176-314
- **Configuration:** `experiments/final_models/config.py`

**Evaluation Utilities:**
- **Threshold optimization:** `experiments/shared/evaluation.py`
- **Metrics computation:** `experiments/shared/evaluation.py`

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Corresponding Code:** `experiments/final_models/train_fusion.py`  
**Related Documentation:**
- `MODEL_ARCHITECTURE.md` - Base model details (LR, TCN)
- `EXPERIMENTAL_DESIGN.md` - Overall methodology
- `PREPROCESSING_PIPELINE.md` - Data preparation
- `LABELING_AND_WINDOWING_STRATEGY.md` - Window creation
