# MAML (Model-Agnostic Meta-Learning) for Stress Prediction

## Overview

This experiment implements **MAML** (Finn et al., 2017) for personalized stress prediction using the VitaStress dataset. MAML learns meta-parameters that can quickly adapt to new subjects with only a few examples.

### Key Concept: ONE Model, Not Two

MAML uses **one model architecture** that learns to be easily adaptable:

```python
# Single base model with meta-parameters θ
base_model = create_maml_model(...)

# Meta-training loop
for each subject (task):
    adapted_model = base_model.clone()        # Same architecture, same weights
    adapted_model.adapt(support_set)          # Fast adaptation (5-10 steps)
    loss = adapted_model.evaluate(query_set)  # Evaluate adaptation quality
    base_model.update(loss)                   # Update θ to be more adaptable
```

The magic: **θ is optimized to be a good starting point** for any subject, enabling fast few-shot adaptation.

---

## Model Architectures

### 1. MLP (Default) - Extracted Features

**Input:** 61 statistical features per window
- 41 accelerometer features (time + frequency domain)
- 12 temperature/heat flux features
- 8 EDA features (if available)

**Architecture:**
```
Input: 61 features
   ↓
FC1: 61 → 64 (ReLU)
   ↓
FC2: 64 → 64 (ReLU)
   ↓
FC3: 64 → 2 (Logits)
```

**Parameters:** ~4,290 (very small for fast adaptation)

**Best for:**
- ✅ Few-shot learning (5-10 examples per subject)
- ✅ Fast training and inference
- ✅ Interpretable features

---

### 2. CNN - Raw Signal Input

**Input:** 8 channels × 120 timesteps (2 minutes at 1Hz)

**Channels:** `acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd`
(Same as MOMENT and SSL experiments)

**Architecture:**
```
Input: [batch, 8 channels, 120 timesteps]
   ↓
Conv1: 8 → 32 filters, kernel=3, stride=2, ReLU  → 60 timesteps
   ↓
Conv2: 32 → 32 filters, kernel=3, stride=2, ReLU → 30 timesteps
   ↓
Conv3: 32 → 32 filters, kernel=3, stride=2, ReLU → 15 timesteps
   ↓
Conv4: 32 → 32 filters, kernel=3, stride=2, ReLU → 8 timesteps
   ↓
Flatten: 32 × 8 = 256
   ↓
FC: 256 → 2 (Logits)
```

**Parameters:** ~8,500 (still small for MAML)

**Best for:**
- ✅ Learning temporal patterns directly from raw signals
- ✅ Comparison with other deep learning models (MOMENT, SSL)
- ✅ When you have more data per subject (>30 samples)

---

## Usage

### Basic Commands

```bash
# MLP with extracted features (default)
python train.py --model mlp

# CNN with raw 8-channel signals
python train.py --model cnn

# Adjust meta-training epochs
python train.py --model mlp --epochs 100

# Custom hyperparameters
python train.py --model cnn \
    --epochs 50 \
    --tasks-per-batch 4 \
    --adaptation-steps 5 \
    --meta-lr 0.001 \
    --inner-lr 0.01
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `mlp` | Model architecture: `mlp` or `cnn` |
| `--epochs` | int | `50` | Meta-training epochs per fold |
| `--tasks-per-batch` | int | `4` | Tasks (subjects) per meta-batch |
| `--adaptation-steps` | int | `5` | Inner loop gradient steps |
| `--meta-lr` | float | `0.001` | Meta-learning rate (outer loop) |
| `--inner-lr` | float | `0.01` | Adaptation learning rate (inner loop) |
| `--threshold` | str | `constrained_gmean` | Threshold method (see below) |
| `--min-recall` | float | `0.85` | Min sensitivity for constrained_gmean |
| `--max-fpr` | float | `0.20` | Max false alarm rate for constrained_gmean |

### Threshold Methods

**Default: `constrained_gmean`** (recommended, same as MOMENT/SSL)

Maximizes G-Mean (√(recall × specificity)) subject to constraints:
- **Recall ≥ 0.85:** Catch at least 85% of stress events
- **FPR ≤ 0.20:** Keep false alarm rate below 20%

**Other options:**
- `youden`: Maximizes Youden's J (sensitivity + specificity - 1)
- `geometric_mean`: Unconstrained G-mean maximization
- `f1`: Maximizes F1 score
- `balanced`: Threshold where sensitivity ≈ specificity

---

## How MAML Works

### Two Nested Loops

**Inner Loop (Fast Adaptation):**
```python
for step in range(adaptation_steps):
    loss = compute_loss(model, support_x, support_y)
    model.adapt(loss)  # Update model for this specific subject
```

**Outer Loop (Meta-Learning):**
```python
for epoch in range(meta_epochs):
    for batch_of_subjects:
        # Clone model for each subject
        adapted_models = [model.clone().adapt(subject_data) for subject in batch]
        
        # Evaluate on query sets
        meta_loss = sum(loss(adapted_model, query_set) for adapted_model in adapted_models)
        
        # Update meta-parameters to improve adaptation
        meta_optimizer.step(meta_loss)
```

### Why This Works

1. **Meta-parameters θ** learn to be a good initialization
2. **Few gradient steps** on support set → rapid personalization
3. **Evaluation on query set** → ensures generalization
4. **Gradient through adaptation** → meta-learning signal

---

## LOSO Cross-Validation

For each subject:
1. **Meta-train** on 20 other subjects (learn adaptable parameters)
2. **Find optimal threshold** on training subjects using constrained G-mean
3. **Adapt** to test subject using 20% of their data (support set)
4. **Evaluate** on remaining 80% with fold-specific threshold (query set)

This tests: "Can the model quickly personalize to a completely new subject?"

### Proper Aggregation Strategy

**CRITICAL:** Aggregate uses fold-level predictions (NO re-thresholding)

```python
# ✅ CORRECT (implemented):
for each fold:
    threshold = find_optimal_on_training(fold_train_data)  # Per-fold threshold
    predictions = (proba >= threshold)                      # Apply to test
    store(predictions)                                      # Store actual predictions

aggregate_metrics = evaluate(all_predictions, all_labels)   # Use actual predictions

# ❌ WRONG (avoided):
mean_threshold = mean(all_fold_thresholds)                 # Average thresholds
predictions = (all_proba >= mean_threshold)                # Re-threshold everything
```

**Why this matters:**
- Each fold has different optimal threshold (different subjects, different distributions)
- Averaging thresholds is statistically meaningless
- Re-thresholding violates constraints and creates inconsistent metrics
- Aggregate must reflect what actually happened

---

## Key Design Choices

### Why No Batch Normalization?

❌ BatchNorm interferes with MAML because:
- Statistics computed on small support sets are unreliable
- Parameters change during adaptation but BN doesn't adapt properly
- MAML paper explicitly avoids BN

### Why Small Networks?

✅ Small networks (~4K-8K params) because:
- Faster adaptation with fewer parameters
- Less prone to overfitting on small support sets
- MAML paper recommends simple architectures

### Why Balanced Sampling?

✅ Critical for stress prediction:
- Without balanced sampling, support sets can be all-negative
- No gradient signal for positive class → adaptation fails
- Solution: Sample k/2 positive, k/2 negative examples

---

## Output Files

Results are saved with model type in filename:

```
results/
├── maml_mlp_metrics.json          # MLP results
├── maml_mlp_fold_metrics.csv      # Per-fold MLP performance
├── maml_mlp_predictions.npz       # MLP predictions
├── maml_cnn_metrics.json          # CNN results
├── maml_cnn_fold_metrics.csv      # Per-fold CNN performance
├── maml_cnn_predictions.npz       # CNN predictions
└── training.log                   # Detailed logs
```

---

## Expected Performance

### MLP (61 Features)
- **AUROC:** ~0.75-0.80
- **Sensitivity:** ~0.80-0.85
- **Specificity:** ~0.70-0.75
- **Training time:** ~30-40 min (21 folds × 50 epochs)

### CNN (8 Channels)
- **AUROC:** ~0.78-0.82
- **Sensitivity:** ~0.82-0.87
- **Specificity:** ~0.72-0.77
- **Training time:** ~45-60 min (21 folds × 50 epochs)

---

## Comparison with Other Experiments

| Experiment | Model | Input | Purpose |
|------------|-------|-------|---------|
| Classical ML | XGBoost/RF | 61 features | Baseline performance |
| MOMENT | Transformer | 8ch @ 1Hz | Foundation model fine-tuning |
| SSL | ResNet1D | 8ch @ 8Hz | Self-supervised pre-training |
| Multi-Rate | Mixed encoders | Native rates | Multi-modal fusion |
| **MAML-MLP** | Small MLP | 61 features | **Fast personalization** |
| **MAML-CNN** | Small CNN | 8ch @ 1Hz | **Fast personalization** |

**MAML's unique advantage:** Learns to quickly adapt to new subjects with only 5-10 examples.

---

## Dependencies

```bash
pip install torch
pip install learn2learn  # Critical for proper MAML implementation
pip install numpy pandas scikit-learn tqdm scipy
```

---

## References

1. Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)
2. [learn2learn library](https://github.com/learnables/learn2learn) - Proper second-order MAML implementation

---

## Troubleshooting

### "learn2learn not installed"
```bash
pip install learn2learn
```

### "No module named 'scipy'"
```bash
pip install scipy
```

### Low performance with CNN
- Increase `--adaptation-steps` to 10
- Increase `--epochs` to 100
- Check that you have enough samples per subject (>30)

### CUDA out of memory
- Reduce `--tasks-per-batch` to 2
- Use `--model mlp` (smaller memory footprint)

---

## Future Improvements

1. **ANIL (Almost No Inner Loop):** Only adapt classification head, freeze encoder
2. **Reptile:** First-order approximation (faster, similar performance)
3. **MAML++:** Task-specific learning rates and layer-wise adaptation
4. **Multi-step adaptation:** Use validation set to find optimal adaptation steps per subject

---

Last updated: December 22, 2024

