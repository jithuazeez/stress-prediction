# LOSO Model Saving Strategy

## The Problem with "Final Model" in LOSO

### What Was Happening (CRITICAL BUG) ⚠️

```python
# After all 21 folds complete...
# Create a NEW, UNTRAINED model with RANDOM WEIGHTS
final_model = create_moment_model(...)  # ← Fresh model, random initialization

# Save this UNTRAINED model as "final"
save_final_model_and_config(final_model, ...)  # ← Useless for inference!
```

**Problem**: The saved `moment_final_model.pt` had **random weights** and was **completely useless**!

---

## Understanding LOSO Cross-Validation

### What LOSO Actually Does

```
Fold 1: Train on [Subjects 2-21] → Test on [Subject 1]  → Model 1 (20 subjects)
Fold 2: Train on [Subjects 1,3-21] → Test on [Subject 2] → Model 2 (20 subjects)
...
Fold 21: Train on [Subjects 1-20] → Test on [Subject 21] → Model 21 (20 subjects)

Result: 21 DIFFERENT models, each trained on different data!
```

### Key Insight: No Single "Final" Model ✨

- **LOSO is for EVALUATION**, not deployment
- Each fold produces a different model (different training subjects)
- The "aggregate metrics" come from 21 different models
- There is no single "final model" that was evaluated

---

## The Correct Approach

### During Training: Save Per-Fold Checkpoints ✅

```python
# Inside LOSO loop:
for fold_idx, test_subject in enumerate(subjects):
    # Train model on N-1 subjects
    model = train_fold(...)
    
    # Save this fold's model
    save_model_checkpoint(
        model, fold_idx, test_subject, metrics,
        is_best=(fold_gmean > best_gmean)
    )
    
    if is_best:
        # Also save as best_model.pt
        torch.save(checkpoint, "checkpoints/best_model.pt")
```

**Saved Files**:
- `checkpoints/fold_1_subject_ABC.pt` - Model trained on subjects 2-21
- `checkpoints/fold_2_subject_DEF.pt` - Model trained on subjects 1,3-21
- ...
- `checkpoints/fold_21_subject_XYZ.pt` - Model trained on subjects 1-20
- `checkpoints/best_model.pt` - Copy of the fold with highest G-mean

### After Training: Document, Don't Create ✅

```python
# After all folds:
logger.info("Best model saved: checkpoints/best_model.pt")
logger.info(f"  - Fold: {best_fold_idx}")
logger.info(f"  - Test Subject: {fold_metrics[best_fold_idx]['subject']}")
logger.info(f"  - G-mean: {best_gmean:.4f}")
logger.info(f"  - Trained on: 20 subjects (LOSO)")
logger.info(f"\nTo use for inference:")
logger.info(f"  checkpoint = torch.load('checkpoints/best_model.pt')")
logger.info(f"  model.load_state_dict(checkpoint['model_state_dict'])")
```

**No new model created!** Just document what was saved.

---

## Which Model to Use

### For Research/Analysis ✅

Use **`checkpoints/best_model.pt`**:
- The fold that achieved highest G-mean
- Trained on 20 subjects
- Validated on 1 subject
- Represents your best LOSO performance

```python
# Load best LOSO model
checkpoint = torch.load('results/checkpoints/best_model.pt')

model = create_moment_model(
    n_channels=7,
    num_classes=2,
    freeze_backbone=True,
    unfreeze_last_n_blocks=2
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# See what it's trained on
print(f"Test subject: {checkpoint['subject_id']}")
print(f"Trained on: 20 other subjects")
print(f"Metrics: {checkpoint['metrics']}")
```

### For Deployment 🚀

**Option A: Use best LOSO model (Quick)**
- Already trained
- Good generalization (validated via LOSO)
- But only trained on 20/21 subjects

**Option B: Retrain on ALL data (Recommended)**
```python
# After LOSO completes, retrain on everything
logger.info("Retraining on all subjects for deployment...")

all_windows = []
for subject_id, windows in windows_by_subject.items():
    all_windows.extend(windows)

# Train on all 21 subjects
deployment_model = train_on_all_data(all_windows, config, best_hyperparameters)

# Save as deployment model
torch.save({
    'model_state_dict': deployment_model.state_dict(),
    'training_info': 'Trained on all 21 subjects for deployment',
    'loso_metrics': aggregate_metrics,  # Reference to validation results
    'hyperparameters': best_hyperparameters
}, 'models/deployment_model.pt')
```

**Why retrain?**
- Uses all available data (21 subjects, not 20)
- Maximizes performance
- Still validated via LOSO metrics

---

## Updated File Structure

```
results/
├── checkpoints/
│   ├── fold_1_subject_ABC.pt       # Trained on subjects 2-21
│   ├── fold_2_subject_DEF.pt       # Trained on subjects 1,3-21
│   ├── ...
│   ├── fold_21_subject_XYZ.pt      # Trained on subjects 1-20
│   └── best_model.pt               # Best fold (highest G-mean)
│
├── config/
│   ├── training_config.json        # Hyperparameters used
│   └── training_summary.txt        # Full summary
│
├── moment_fold_metrics.csv         # Per-fold performance
├── predictions.npz                 # All predictions
└── plots/                          # Visualization
```

**Key Changes**:
- ❌ Removed `models/moment_final_model.pt` (was useless)
- ✅ Kept `checkpoints/best_model.pt` (actual trained weights)
- ✅ Added `config/` directory for documentation

---

## Summary of Changes

### Before (Broken) ❌

```python
# After LOSO:
final_model = create_moment_model(...)  # New untrained model
save_final_model_and_config(final_model, ...)  # Save random weights

# Result: useless "final" model
```

### After (Fixed) ✅

```python
# After LOSO:
logger.info("Best model: checkpoints/best_model.pt")
logger.info(f"  Fold: {best_fold_idx}, G-mean: {best_gmean:.4f}")

# Save config only (no new model)
save_config_and_summary(hyperparameters, metrics)

# Result: clear documentation, actual trained weights available
```

---

## Key Takeaways

1. **LOSO produces multiple models** (one per fold), not a single "final" model
2. **Best fold model** is the one with highest validation G-mean
3. **Aggregate metrics** come from 21 different models, not one
4. **For deployment**: Either use best fold model OR retrain on all data
5. **Never save untrained models** - confusing and useless!

---

## How to Load and Use

### Load Best LOSO Model

```python
import torch
from experiments.02_moment.model import create_moment_model

# Load checkpoint
checkpoint = torch.load('experiments/02_moment/results/checkpoints/best_model.pt')

# Create model
model = create_moment_model(
    n_channels=checkpoint['hyperparameters']['n_channels'],
    num_classes=2,
    freeze_backbone=True,
    unfreeze_last_n_blocks=checkpoint['hyperparameters']['unfreeze_last_n_blocks']
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
import numpy as np
with torch.no_grad():
    x = torch.randn(1, 7, 512)  # [batch, channels, timesteps]
    logits = model(x)
    proba = torch.softmax(logits, dim=-1)[:, 1]
    
# Use fold-specific threshold
threshold = checkpoint['metrics']['threshold']
pred = (proba >= threshold).int()
```

---

## Date

Fixed: 2025-12-21

## Related

- `AGGREGATION_FIX.md` - How aggregate metrics are computed
- `THRESHOLD_METHOD_UPDATE.md` - Threshold selection strategy
- `THRESHOLD_PARAMETER_FIX.md` - Threshold parameter handling
