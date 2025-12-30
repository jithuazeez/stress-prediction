# Focal Loss Implementation for TCN

## Date: 2025-12-28

## Summary of Changes

Modified the TCN implementation to use **Focal Loss** instead of weighted cross-entropy for handling class imbalance (1:7 positive-to-negative ratio).

---

## What is Focal Loss?

Focal Loss was introduced by Lin et al. (2017) for object detection with extreme class imbalance. It addresses two key problems:

1. **Class imbalance**: More weight on minority (positive) class
2. **Easy vs hard examples**: Down-weights easy examples, focuses on hard negatives

**Formula:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

**Parameters:**
- **α (alpha) = 0.88**: Weight for positive class
  - Matches 1:7 imbalance ratio (1520 neg / 1733 total ≈ 0.877)
  - Positive class gets 88% weight, negative gets 12%
  
- **γ (gamma) = 2.0**: Focusing parameter (standard value)
  - γ=0 → standard weighted cross-entropy
  - γ=2 → standard focusing (most common)
  - Higher γ → more aggressive focusing on hard examples

**How it works:**
- **Easy examples** (high confidence): Low loss contribution (down-weighted)
- **Hard examples** (low confidence): High loss contribution (focused on)
- **Misclassified**: Maximum loss contribution

---

## Files Modified

### 1. `model.py`
**Added:**
- `FocalLoss` class (lines 9-87)
  - Implements Focal Loss with configurable α and γ
  - Supports binary/multi-class classification
  - Includes input validation and documentation

**Features:**
- Takes raw logits (pre-softmax) as input
- Computes focal weight: `(1 - p_t)^γ`
- Applies alpha weight based on class
- Combines with cross-entropy loss

### 2. `train.py`
**Modified:**

**Import (line 53):**
```python
from model import create_tcn_model, FocalLoss
```

**Function signature (lines 276-296):**
- Added `loss_type` parameter (default: `"focal"`)
- Added `focal_alpha` parameter (default: `0.88`)
- Added `focal_gamma` parameter (default: `2.0`)

**Loss function selection (lines 433-454):**
```python
if loss_type == "focal":
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)
    
# Commented out weighted cross-entropy (kept for comparison):
# elif loss_type == "weighted_ce":
#     weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], ...)
#     criterion = nn.CrossEntropyLoss(weight=weight)
```

**Main training call (lines 699-718):**
```python
results = loso_cross_validation(
    ...,
    loss_type="focal",
    focal_alpha=0.88,
    focal_gamma=2.0
)
```

**Logging updates:**
- Added loss type and parameters to training logs
- Hyperparameters now include focal loss settings

---

## Configuration

### Current Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss Type** | `focal` | Better for imbalanced data |
| **Alpha (α)** | `0.88` | Matches 1:7 class imbalance |
| **Gamma (γ)** | `2.0` | Standard focusing parameter |
| **Threshold Method** | `constrained_gmean` | ✅ **FIXED**: Now properly applies constraints |
| **Min Recall** | `0.75` | Catch at least 75% of stress events |
| **Max FPR** | `0.25` | Allow max 25% false alarm rate |

### To Switch Back to Weighted Cross-Entropy

In `train.py` line 717:
```python
# Change:
loss_type="focal",

# To:
loss_type="weighted_ce",
```

Then uncomment the weighted CE code (lines 443-448).

---

## Expected Impact

### Potential Improvements:
1. ✅ **Better handling of hard examples** - More focus on difficult stress cases
2. ✅ **Reduced easy negative dominance** - Easy non-stress samples contribute less
3. ✅ **Improved AUROC/PR-AUC** - Especially if many easy negatives exist
4. ✅ **Better gradient flow** - Hard examples drive learning

### Realistic Expectations:

**Baseline (Weighted CE + Constrained threshold):**
- Expected: Recall ~75%, AUROC ~75%, PR-AUC ~30-35%

**With Focal Loss:**
- Optimistic: Recall 75-80%, AUROC 77-80%, PR-AUC 33-38%
- Realistic: Recall 75-77%, AUROC 76-78%, PR-AUC 31-36%
- **Gain: 2-5% improvement if lucky**

---

## Hyperparameter Tuning

### Alpha (α) - Class Weight

Try these values if needed:
```python
focal_alpha = 0.75  # Less aggressive
focal_alpha = 0.88  # Current (matches 1:7 ratio)
focal_alpha = 0.90  # More aggressive
```

**When to adjust:**
- If recall too low → increase alpha (more weight on positive class)
- If too many false positives → decrease alpha

### Gamma (γ) - Focusing Parameter

Try these values if needed:
```python
focal_gamma = 1.0   # Mild focusing
focal_gamma = 2.0   # Current (standard)
focal_gamma = 3.0   # Aggressive focusing
focal_gamma = 5.0   # Very aggressive (for extreme imbalance)
```

**When to adjust:**
- If training unstable → decrease gamma
- If still dominated by easy examples → increase gamma

---

## Training Commands

### Run with Focal Loss (Default):
```bash
cd experiments/08_tcn
python train.py
```

### Run with different parameters:
```python
# In train.py main(), modify:
results = loso_cross_validation(
    ...,
    loss_type="focal",
    focal_alpha=0.75,  # Experiment
    focal_gamma=1.0    # Experiment
)
```

---

## Comparison Study

To compare Focal Loss vs Weighted CE:

1. **Run with Focal Loss** (current setup)
   - Save results to `results/focal_alpha88_gamma2/`

2. **Run with Weighted CE**
   - Change `loss_type="weighted_ce"` in line 717
   - Uncomment weighted CE code (lines 443-448)
   - Save results to `results/weighted_ce/`

3. **Compare metrics:**
   - AUROC (threshold-independent)
   - PR-AUC (threshold-independent)
   - Recall @ constrained threshold
   - Training stability (loss curves)

---

## Key Fixes Applied

### ✅ Fixed Threshold Selection
**Before:**
```python
threshold_method="geometric_mean",  # IGNORED constraints!
```

**After:**
```python
threshold_method="constrained_gmean",  # ✅ Applies constraints
```

This ensures:
- Minimum 75% recall (catches most stress events)
- Maximum 25% FPR (controls false alarms)
- Clinical safety while maintaining usability

---

## References

1. **Focal Loss Paper:**
   - Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
   - https://arxiv.org/abs/1708.02002

2. **TCN Papers:**
   - Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
   - https://arxiv.org/pdf/1803.01271.pdf

3. **Implementation:**
   - PyTorch-style implementation with proper gradient flow
   - Compatible with standard PyTorch training loops

---

## Notes

- **Weighted CE code is commented out**, not removed
- Can easily switch between loss functions
- Both loss functions work with same threshold selection
- Focal Loss probabilities may be less calibrated (use AUROC/PR-AUC for comparison)

---

## Next Steps

1. ✅ **Code updated** with Focal Loss
2. ⏳ **Train model** (~30 minutes for LOSO CV)
3. ⏳ **Compare results** with baseline
4. ⏳ **Tune hyperparameters** if needed
5. ⏳ **Report findings** in dissertation

---

## Questions?

If you want to:
- Switch back to weighted CE: Change `loss_type` parameter
- Tune focal parameters: Modify `focal_alpha` and `focal_gamma`
- Compare both: Run experiments with each loss function
- Understand results: Check AUROC, PR-AUC, and recall metrics

