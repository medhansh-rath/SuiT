#!/usr/bin/env python3
"""Quick test of SLIC algorithm implementation."""

import numpy as np
from algorithms import SLICAlgorithm

# Create a simple synthetic image
h, w = 100, 100
rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
depth = np.random.rand(h, w).astype(np.float32)

# Test SLIC
slic = SLICAlgorithm(n_segments=50, compactness=10.0, use_depth=True)
result = slic.segment(rgb, depth)

print(f"✓ SLIC test successful")
print(f"  - Input image: {rgb.shape}")
print(f"  - Output labels shape: {result.labels.shape}")
print(f"  - Number of superpixels: {result.num_labels}")
print(f"  - Runtime: {result.runtime_seconds:.4f}s")
