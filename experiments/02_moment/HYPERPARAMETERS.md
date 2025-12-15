# MOMENT Model Hyperparameters Guide

This document lists all tunable hyperparameters for the MOMENT foundation model training pipeline.

---

## Core Training Hyperparameters

### 1. **Learning Rate** (`learning_rate`)
**Current:** `1e-3` (0.001)
**Typical Range:** `1e-5` to `1e-2`

Controls how fast the model learns.

- **Higher (1e-3, 5e-3):** Faster learning, but risk of overshooting optimal weights
- **Lower (1e-5, 5e-5):** Slower, more stable learning, better for fine-tuning
- **Recommended:** Start with `1e-3` for frozen backbone, try `1e-4` if unstable

```python
results = loso_cross_validation(..., learning_rate=1e-3)
```

---

### 2. **Batch Size** (`batch_size`)
**Current:** `16`
**Typical Range:** `8` to `64`

Number of samples processed together before updating weights.

- **Smaller (8, 16):** More frequent weight updates, more stochastic, uses less memory
- **Larger (32, 64):** More stable gradients, faster training, requires more GPU memory
- **Recommended:** `16` for balanced performance, reduce to `8` if GPU memory limited

```python
results = loso_cross_validation(..., batch_size=16)
```

---

### 3. **Number of Epochs** (`n_epochs`)
**Current:** `10`
**Typical Range:** `5` to `50`

How many times the model sees the entire training set per fold.

- **Fewer (5-10):** Faster training, risk of underfitting
- **More (20-50):** Better convergence, risk of overfitting, much longer training
- **Recommended:** `10-15` for frozen backbone, `20-30` if unfreezing layers

```python
results = loso_cross_validation(..., n_epochs=10)
```

---

### 4. **Class Weight** (`weight` in CrossEntropyLoss)
**Current:** `n_neg / n_pos` (dynamic, ~6.8)
**Options:**
- **Dynamic:** `weight = n_neg / n_pos` (current, ~6.8)
- **Square Root:** `weight = sqrt(n_neg / n_pos)` (~2.6)
- **Fixed Moderate:** `weight = 3.0`
- **Balanced:** `weight = 2.0`
- **None:** `weight = 1.0` (no weighting)

Controls how much the model penalizes missing stress cases vs. false alarms.

- **Higher weight (6-8):** Catches more stress, but many false alarms
- **Lower weight (2-3):** Fewer false alarms, might miss some stress cases
- **Recommended:** Try `3.0` to reduce false positives

**To change in `train.py` line ~343:**
```python
# Current (dynamic):
weight = torch.tensor([1.0, n_neg / max(n_pos, 1)]).to(device)

# Recommended (fixed moderate):
weight = torch.tensor([1.0, 3.0]).to(device)

# Or square root:
weight = torch.tensor([1.0, np.sqrt(n_neg / max(n_pos, 1))]).to(device)
```

---

## Model Architecture Hyperparameters

### 5. **Number of Input Channels** (`n_channels`)
**Current:** `3` (acc_magnitude, skin_temp, eda_stress_skin)
**Options:** `3` or `4` (if re-adding PPG)

Defined in `config.py`.

```python
# In experiments/shared/config.py
n_channels: int = 3
```

---

### 6. **Freeze Backbone** (`freeze_backbone`)
**Current:** `True` (only classification head trained)
**Options:**
- `True`: Freeze entire MOMENT encoder, train only classification head (fast, less overfitting)
- `False`: Train entire model (slow, requires more data, can overfit)
- **Partial (custom):** Freeze bottom 70%, train top 30% + head (see below)

```python
model = create_moment_model(..., freeze_backbone=True)
```

**To unfreeze top layers (better adaptation):**

In `experiments/02_moment/model.py`, modify `_freeze_backbone()`:
```python
def _freeze_backbone(self):
    """Freeze only bottom 70% of layers."""
    total_params = list(self.moment.named_parameters())
    freeze_until = int(len(total_params) * 0.7)
    
    for idx, (name, param) in enumerate(total_params):
        if 'head' in name.lower() or 'class' in name.lower():
            param.requires_grad = True  # Always train head
        elif idx < freeze_until:
            param.requires_grad = False  # Freeze bottom 70%
        else:
            param.requires_grad = True   # Train top 30%
```

---

## Optimizer Hyperparameters

### 7. **Optimizer Choice**
**Current:** `Adam`
**Options:**
- **Adam:** Adaptive learning rates, good default
- **AdamW:** Adam with weight decay (better regularization)
- **SGD:** Simple, requires learning rate tuning

```python
# Adam (current):
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# AdamW (recommended for less overfitting):
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# SGD with momentum:
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
```

---

### 8. **Weight Decay** (L2 regularization)
**Current:** `0.0` (not used)
**Typical Range:** `1e-5` to `1e-2`

Penalizes large weights to prevent overfitting.

```python
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
```

---

### 9. **Gradient Clipping**
**Current:** Not used
**Typical Range:** `0.5` to `5.0`

Prevents exploding gradients.

**To add in `train_epoch()` function after `loss.backward()`:**
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this
optimizer.step()
```

---

## Learning Rate Scheduling

### 10. **Learning Rate Scheduler**
**Current:** Not used
**Options:**
- **ReduceLROnPlateau:** Reduce LR when loss plateaus
- **CosineAnnealingLR:** Smooth reduction over training
- **StepLR:** Drop LR at fixed intervals

**To add ReduceLROnPlateau:**
```python
# After optimizer creation:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# In training loop, after each epoch:
scheduler.step(train_loss)
```

---

## Data Preprocessing Hyperparameters

### 11. **Window Size** (`window_size_sec`)
**Current:** `120` seconds
**Options:** `60`, `90`, `120`, `180`

Defined in `config.py`. Longer windows = more context, fewer samples.

```python
# In experiments/shared/config.py
window_size_sec: int = 120
```

---

### 12. **Window Overlap** (`overlap_ratio`)
**Current:** `0.0` (no overlap)
**Options:** `0.0` to `0.75`

Higher overlap = more training samples, but correlated data.

```python
# In experiments/shared/config.py
overlap_ratio: float = 0.0  # Try 0.5 for 50% overlap
```

---

### 13. **Prediction Horizon** (`horizons_minutes`)
**Current:** `[3, 5, 10]` minutes
**Target:** `5` minutes (using `label_5min`)

Time ahead to predict stress.

```python
# In experiments/shared/config.py
horizons_minutes: List[int] = field(default_factory=lambda: [3, 5, 10])
target_label: str = "label_5min"
```

---

## Advanced Hyperparameters

### 14. **Dropout** (if adding to classification head)
**Current:** Not explicitly set (MOMENT internal)
**Typical Range:** `0.1` to `0.5`

Randomly drops neurons during training to prevent overfitting.

**To modify in `model.py`:**
```python
# In MOMENTClassifier, add dropout to head
self.dropout = nn.Dropout(0.3)

def forward(self, x):
    output = self.moment.classify(x_enc=x)
    logits = self.dropout(output.logits)  # Apply dropout
    return logits
```

---

### 15. **Early Stopping**
**Current:** Not used
**Typical:** Stop if validation loss doesn't improve for 5 epochs

```python
best_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(n_epochs):
    train_loss = train_epoch(...)
    
    if train_loss < best_loss:
        best_loss = train_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## Alternative Loss Functions

### 16. **Focal Loss** (better for imbalanced data)
**Current:** `CrossEntropyLoss` with class weights

Focal Loss focuses on hard-to-classify examples.

**To add:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Replace CrossEntropyLoss:
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

---

## Recommended Hyperparameter Combinations

### **Configuration 1: Reduce False Alarms** (Most Important)
```python
results = loso_cross_validation(
    ...,
    n_epochs=10,
    batch_size=16,
    learning_rate=1e-3
)

# In train.py, change class weight to:
weight = torch.tensor([1.0, 3.0]).to(device)  # Fixed moderate weight
```

**Expected:** Precision improves to ~0.35-0.45, Recall drops to ~0.50-0.60

---

### **Configuration 2: Better Adaptation (Unfreeze Layers)**
```python
# Modify model.py _freeze_backbone() as shown above
# Then run with:
results = loso_cross_validation(
    ...,
    n_epochs=20,  # More epochs needed
    batch_size=16,
    learning_rate=5e-5  # Lower LR for unfrozen layers
)
```

**Expected:** Better subject-wise generalization, longer training

---

### **Configuration 3: More Training Data (Overlap Windows)**
```python
# In config.py:
overlap_ratio: float = 0.5  # 50% overlap

# Then run with standard settings
```

**Expected:** ~1600 samples instead of 800, better learning

---

### **Configuration 4: Focal Loss + Moderate Weight**
```python
# Add Focal Loss class to train.py
criterion = FocalLoss(alpha=0.25, gamma=2.0)

results = loso_cross_validation(
    ...,
    n_epochs=15,
    batch_size=16,
    learning_rate=1e-3
)
```

**Expected:** Better handling of hard-to-classify samples

---

## How to Track Hyperparameter Experiments

All hyperparameters are now automatically saved in:
- `experiments/02_moment/results/models/hyperparameters.json`
- `experiments/02_moment/results/models/model_summary.txt`

**Best practice:** Create a tracking spreadsheet:

| Run | LR | Batch | Epochs | Weight | Freeze | AUROC | PR-AUC | Precision | Recall |
|-----|-----|-------|--------|--------|--------|-------|--------|-----------|--------|
| 1 | 1e-3 | 16 | 10 | 6.8 | True | 0.73 | 0.26 | 0.24 | 0.59 |
| 2 | 1e-3 | 16 | 10 | 3.0 | True | ? | ? | ? | ? |

---

## Quick Reference: What to Tune First

**Priority 1:** Class weight (3.0 instead of 6.8) → Reduces false alarms  
**Priority 2:** Unfreeze top 30% layers → Better adaptation  
**Priority 3:** Increase epochs to 20 → Better convergence  
**Priority 4:** Add window overlap 50% → More training data  
**Priority 5:** Try Focal Loss → Better for imbalanced data  

Start with Priority 1, evaluate, then try Priority 2, etc.

