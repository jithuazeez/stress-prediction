"""
Simple test script for TCN encoders.
"""

import torch
from encoders import ACCEncoder, PhysioEncoder, create_encoders, count_parameters

print("Testing TCN-based Multi-Rate Encoders...")
print("="*60)

# Test 1: Create encoders
print("\n1. Creating encoders...")
try:
    acc_enc, physio_enc = create_encoders(window_size_sec=120)
    print("   ✓ Encoders created successfully")
except Exception as e:
    print(f"   ✗ Error creating encoders: {e}")
    exit(1)

# Test 2: Check encoder types
print("\n2. Checking encoder architecture...")
print(f"   ACC Encoder: {type(acc_enc).__name__}")
print(f"   Physio Encoder: {type(physio_enc).__name__}")
print(f"   ACC has TCN: {hasattr(acc_enc, 'tcn')}")
print(f"   Physio has TCN: {hasattr(physio_enc, 'tcn')}")

# Test 3: Parameter counts
print("\n3. Parameter counts:")
acc_params = count_parameters(acc_enc)
physio_params = count_parameters(physio_enc)
total_params = acc_params + physio_params
print(f"   ACC Encoder:    {acc_params:,} parameters")
print(f"   Physio Encoder: {physio_params:,} parameters")
print(f"   Total:          {total_params:,} parameters")

# Test 4: Forward pass with 120s window
print("\n4. Testing forward pass (120s window)...")
try:
    batch_size = 2
    acc_input = torch.randn(batch_size, 3, 3840)  # 3 channels, 120s @ 32Hz
    physio_input = torch.randn(batch_size, 5, 120)  # 5 channels, 120s @ 1Hz
    
    acc_emb = acc_enc(acc_input)
    physio_emb = physio_enc(physio_input)
    
    print(f"   Input shapes:")
    print(f"     ACC:    {acc_input.shape}")
    print(f"     Physio: {physio_input.shape}")
    print(f"   Output shapes:")
    print(f"     ACC:    {acc_emb.shape}")
    print(f"     Physio: {physio_emb.shape}")
    print("   ✓ Forward pass successful")
except Exception as e:
    print(f"   ✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test with 60s window
print("\n5. Testing with 60s window...")
try:
    acc_enc_60, physio_enc_60 = create_encoders(window_size_sec=60)
    
    acc_input_60 = torch.randn(2, 3, 1920)  # 60s @ 32Hz
    physio_input_60 = torch.randn(2, 5, 60)  # 60s @ 1Hz
    
    acc_emb_60 = acc_enc_60(acc_input_60)
    physio_emb_60 = physio_enc_60(physio_input_60)
    
    print(f"   Output shapes (60s):")
    print(f"     ACC:    {acc_emb_60.shape}")
    print(f"     Physio: {physio_emb_60.shape}")
    print("   ✓ 60s window test successful")
except Exception as e:
    print(f"   ✗ Error with 60s window: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ All tests passed! TCN encoders are working correctly.")
print("="*60)

