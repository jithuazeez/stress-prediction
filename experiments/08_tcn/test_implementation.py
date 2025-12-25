"""
Simple test to verify TCN implementation is working.

Tests:
1. Import all modules
2. Check TCN model can be created
3. Check dataset can be created
4. Verify forward pass works
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("TCN IMPLEMENTATION VERIFICATION")
print("=" * 60)

# Test 1: Import model
print("\n[1/4] Testing model imports...")
try:
    from model import TCNClassifier, create_tcn_model, TemporalConvNet, TemporalBlock
    print("  ✅ Model imports successful")
except Exception as e:
    print(f"  ❌ Model import failed: {e}")
    sys.exit(1)

# Test 2: Create model
print("\n[2/4] Testing model creation...")
try:
    model = create_tcn_model(
        num_inputs=57,
        num_classes=2,
        num_channels=[32, 32, 32],
        kernel_size=3,
        dropout=0.2
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Model created successfully")
    print(f"     - Parameters: {total_params:,}")
    print(f"     - Input features: 57")
    print(f"     - Output classes: 2")
    print(f"     - Receptive field: {model.get_receptive_field(3, 2)}")
except Exception as e:
    print(f"  ❌ Model creation failed: {e}")
    sys.exit(1)

# Test 3: Test forward pass
print("\n[3/4] Testing forward pass...")
try:
    batch_size = 16
    num_features = 57
    seq_len = 1
    
    x = torch.randn(batch_size, num_features, seq_len)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
    
    assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities not in [0, 1]"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size)), "Probabilities don't sum to 1"
    
    print(f"  ✅ Forward pass successful")
    print(f"     - Input shape: {x.shape}")
    print(f"     - Output shape: {logits.shape}")
    print(f"     - Probs shape: {probs.shape}")
    print(f"     - Sample output: {probs[0].numpy()}")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Import dataset (without actually loading data)
print("\n[4/4] Testing dataset imports...")
try:
    from dataset import VitaStressTCNDataset, create_tcn_datasets
    print("  ✅ Dataset imports successful")
    print("     Note: Full dataset test requires actual data")
except Exception as e:
    print(f"  ❌ Dataset import failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✅ All basic tests passed!")
print("\nNext steps:")
print("  1. Run full dataset test: python dataset.py")
print("  2. Run full model test: python model.py")
print("  3. Run training: python train.py")
print("\n" + "=" * 60)

