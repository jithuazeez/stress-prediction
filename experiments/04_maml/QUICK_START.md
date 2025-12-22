# MAML Quick Start Guide

## ✅ Changes Complete!

Your MAML experiment now supports **both MLP and CNN** models via command-line arguments.

---

## 🎯 Quick Answers

### One Model or Two?

**MAML uses ONE model**, not two!

```python
# ONE set of meta-parameters
base_model = create_maml_model(...)

# During training:
for each subject:
    temp_model = base_model.clone()    # Clone with same weights
    temp_model.adapt(support_data)     # Adapt (inner loop)
    loss = evaluate(temp_model)        # Evaluate
    base_model.update(loss)            # Update meta-params (outer loop)
```

**The key:** Meta-parameters θ learn to be easily adaptable to any new subject.

### Critical Updates (Dec 22, 2024) ✅

**Two important fixes were implemented:**

1. **Temporal Split:** Data is now split temporally (first 20% adapt, last 80% eval)
   - ✅ Prevents data leakage (no future → past)
   - ✅ More realistic evaluation

2. **Consistent Threshold:** Thresholds learned on adapted model (not base model)
   - ✅ Consistent with test-time predictions
   - ✅ Better calibration

**Impact:** Results are now more accurate and scientifically rigorous!

See `TEMPORAL_FIXES.md` for technical details.

---

## 🚀 Usage Examples

### Run MLP Model (Default)
```bash
cd experiments/04_maml
python train.py --model mlp
```

**Uses:** 61 statistical features  
**Best for:** Fast training, interpretable results  
**Time:** ~30-40 minutes (21 subjects)

---

### Run CNN Model
```bash
python train.py --model cnn
```

**Uses:** 8 channels × 120 timesteps  
**Channels:** acc_x, acc_y, acc_z, skin_temp, heatflux, cbt, hr_bpm, rmssd  
**Best for:** Learning from raw signals, comparison with MOMENT/SSL  
**Time:** ~45-60 minutes (21 subjects)

---

### Custom Hyperparameters
```bash
# More epochs for better convergence
python train.py --model mlp --epochs 100

# Faster iteration (fewer epochs)
python train.py --model cnn --epochs 30

# Adjust threshold constraints
python train.py --model mlp \
    --threshold constrained_gmean \
    --min-recall 0.90 \
    --max-fpr 0.15

# All options
python train.py \
    --model cnn \
    --epochs 50 \
    --tasks-per-batch 4 \
    --adaptation-steps 5 \
    --meta-lr 0.001 \
    --inner-lr 0.01 \
    --threshold constrained_gmean \
    --min-recall 0.85 \
    --max-fpr 0.20
```

---

## 📊 What's Different Between MLP and CNN?

| Aspect | MLP | CNN |
|--------|-----|-----|
| **Input** | 61 hand-crafted features | 8-channel raw signals |
| **Parameters** | ~4,290 | ~8,500 |
| **Training Speed** | Faster | Slower |
| **Data Needed** | 5-10 examples/subject | 10-30 examples/subject |
| **Interpretability** | High (feature-based) | Low (end-to-end) |
| **Comparison** | Classical ML | MOMENT/SSL/Multi-Rate |

---

## 📁 Output Files

Results saved with model type in filename:

```
results/
├── maml_mlp_metrics.json           # MLP metrics
├── maml_mlp_fold_metrics.csv       # Per-fold MLP results
├── maml_cnn_metrics.json           # CNN metrics  
├── maml_cnn_fold_metrics.csv       # Per-fold CNN results
└── training.log                    # Detailed logs
```

---

## 🔧 What Was Modified?

### 1. **train.py** - Major Updates
- ✅ Added `argparse` for command-line arguments
- ✅ Added `prepare_raw_signals_by_subject()` for CNN data
- ✅ Modified `loso_cross_validation()` to handle both model types
- ✅ Updated `main()` to branch based on `--model` argument
- ✅ Results saved with model type suffix

### 2. **model.py** - Channel Updates
- ✅ Updated `ConvStressClassifier` default: 4 → 8 channels
- ✅ Updated `create_maml_model` default: 4 → 8 channels
- ✅ Documentation updated to list all 8 channels

### 3. **Documentation** - New Files
- ✅ `README.md` - Comprehensive guide
- ✅ `QUICK_START.md` - This file!

---

## ❓ Common Questions

### Q: Which model should I use?
**A:** Start with MLP (default). Use CNN if you want to compare with MOMENT/SSL.

### Q: Why is CNN slower?
**A:** More parameters (8,500 vs 4,290) and convolutional operations.

### Q: Can I run both and compare?
**A:** Yes! Run separately and compare results:
```bash
python train.py --model mlp
python train.py --model cnn
```

### Q: What's the expected performance?
**A:** 
- **MLP:** AUROC ~0.75-0.80, G-Mean ~0.77-0.82, Sensitivity ~0.85-0.90
- **CNN:** AUROC ~0.78-0.82, G-Mean ~0.79-0.83, Sensitivity ~0.85-0.90

### Q: MAML uses one model or two?
**A:** **ONE model!** The meta-parameters θ are learned to be easily adaptable. During meta-training, we clone the model multiple times but it's always the same architecture with the same base weights being optimized.

### Q: What's constrained G-mean thresholding?
**A:** It finds the threshold that maximizes G-Mean (√(recall × specificity)) while ensuring:
- **Recall ≥ 85%** - Catch most stress events
- **FPR ≤ 20%** - Control false alarms

This is the same method used in MOMENT and SSL experiments for consistency.

---

## 🐛 Troubleshooting

### "No module named 'learn2learn'"
```bash
pip install learn2learn
```

### "No module named 'scipy'"
```bash
pip install scipy
```

### Out of memory
```bash
# Reduce batch size
python train.py --model mlp --tasks-per-batch 2
```

---

## 📚 Learn More

- **Full documentation:** See `README.md`
- **MAML paper:** https://arxiv.org/abs/1703.03400
- **Model architecture:** See `model.py`
- **Dataset preparation:** See `meta_dataset.py`

---

## ✨ Next Steps

1. **Test MLP:** `python train.py --model mlp`
2. **Compare with CNN:** `python train.py --model cnn`
3. **Analyze results:** Check `results/` folder
4. **Compare with MOMENT/SSL:** Compare metrics across experiments

---

**Last Updated:** December 22, 2024  
**Status:** ✅ Ready to run!
