"""
Sanity check: verify Online SCNet model dimensions, causality, and forward pass.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn

# Test 1: Model instantiation and forward pass
print("=" * 70)
print("TEST 1: Model instantiation and forward pass")
print("=" * 70)

from scnet.SCNet import SCNet

model = SCNet(
    sources=['drums', 'bass', 'other', 'vocals'],
    audio_channels=2,
    dims=[4, 32, 64, 128],
    nfft=4096,
    hop_size=1024,
    win_size=4096,
    normalized=True,
    band_SR=[0.175, 0.392, 0.433],
    band_stride=[1, 4, 16],
    band_kernel=[3, 4, 16],
    conv_depths=[3, 2, 1],
    compress=4,
    conv_kernel=3,
    num_dplayer=6,
    expand=1,
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# Test with a realistic input: batch=2, channels=2, ~5 seconds at 44100 Hz
B, C, L = 2, 2, 44100 * 5
x = torch.randn(B, C, L)
print(f"Input shape: {x.shape}")

model.eval()
with torch.no_grad():
    y = model(x)

print(f"Output shape: {y.shape}")
assert y.shape == (B, 4, C, L), f"Expected shape {(B, 4, C, L)}, got {y.shape}"
print("✓ Output shape matches expected (B, sources, channels, length)")

# Test 2: Verify causal convolutions
print("\n" + "=" * 70)
print("TEST 2: Verify causal convolution modules")
print("=" * 70)

from scnet.SCNet import CausalConv1d, CausalConv2d

# CausalConv1d
conv1d = CausalConv1d(8, 16, kernel_size=3)
x1d = torch.randn(1, 8, 20)
y1d = conv1d(x1d)
print(f"CausalConv1d: input {x1d.shape} -> output {y1d.shape}")
assert y1d.shape[-1] == x1d.shape[-1], "CausalConv1d should preserve temporal length"
print("✓ CausalConv1d preserves temporal length")

# CausalConv2d
conv2d = CausalConv2d(8, 16, kernel_size=3)
x2d = torch.randn(1, 8, 10, 20)  # (B, C, F, T)
y2d = conv2d(x2d)
print(f"CausalConv2d: input {x2d.shape} -> output {y2d.shape}")
assert y2d.shape[-1] == x2d.shape[-1], "CausalConv2d should preserve temporal length"
assert y2d.shape[-2] == x2d.shape[-2], "CausalConv2d should preserve frequency length"
print("✓ CausalConv2d preserves temporal and frequency length")

# Test 3: Verify cLN
print("\n" + "=" * 70)
print("TEST 3: Verify Cumulative LayerNorm (cLN)")
print("=" * 70)

from scnet.separation import CumulativeLayerNorm

cln3d = CumulativeLayerNorm(8)
x3d = torch.randn(2, 8, 50)
y3d = cln3d(x3d)
print(f"cLN 3D: input {x3d.shape} -> output {y3d.shape}")
assert y3d.shape == x3d.shape, "cLN should preserve shape"

cln4d = CumulativeLayerNorm(16)
x4d = torch.randn(2, 16, 10, 50)
y4d = cln4d(x4d)
print(f"cLN 4D: input {x4d.shape} -> output {y4d.shape}")
assert y4d.shape == x4d.shape, "cLN should preserve shape"
print("✓ cLN preserves shape for both 3D and 4D inputs")

# Test 4: Verify separation net (no FFT)
print("\n" + "=" * 70)
print("TEST 4: Verify SeparationNet (no FFT, unidirectional time LSTM)")
print("=" * 70)

from scnet.separation import SeparationNet, DualPathRNN

sep_net = SeparationNet(channels=128, expand=1, num_layers=6)
x_sep = torch.randn(2, 128, 8, 20)  # (B, C, F, T)
y_sep = sep_net(x_sep)
print(f"SeparationNet: input {x_sep.shape} -> output {y_sep.shape}")
assert y_sep.shape == x_sep.shape, "SeparationNet should preserve shape (no FFT channel changes)"
print("✓ SeparationNet preserves shape (no FFT doubling)")

# Verify LSTM directionality
dprnn = sep_net.dp_modules[0]
print(f"Freq LSTM bidirectional: {dprnn.lstm_freq.bidirectional}")
print(f"Time LSTM bidirectional: {dprnn.lstm_time.bidirectional}")
assert dprnn.lstm_freq.bidirectional == True, "Frequency LSTM should be bidirectional"
assert dprnn.lstm_time.bidirectional == False, "Time LSTM should be unidirectional (causal)"
print("✓ Freq LSTM is bidirectional, Time LSTM is unidirectional")

# Test 5: Verify no accelerate imports
print("\n" + "=" * 70)
print("TEST 5: Verify no accelerate/HuggingFace imports")
print("=" * 70)

import importlib
modules_to_check = [
    'scnet.SCNet', 'scnet.separation', 'scnet.train',
    'scnet.solver', 'scnet.wav', 'scnet.apply', 'scnet.log'
]
for mod_name in modules_to_check:
    mod = importlib.import_module(mod_name)
    source_file = mod.__file__
    with open(source_file, 'r') as f:
        content = f.read()
    has_accelerate = 'accelerate' in content.lower()
    status = "✗ CONTAINS accelerate" if has_accelerate else "✓ Clean"
    print(f"  {mod_name}: {status}")
    assert not has_accelerate, f"{mod_name} still contains accelerate references!"

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
