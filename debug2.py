import numpy as np
import os

bad_patches = []
for i in range(1, 1387):
    before = np.load(f'data/patches/before/{i:05d}.npy').astype(np.float32)
    after = np.load(f'data/patches/after/{i:05d}.npy').astype(np.float32)
    
    if np.isnan(before).any() or np.isinf(before).any():
        bad_patches.append(f"before_{i}")
    if np.isnan(after).any() or np.isinf(after).any():
        bad_patches.append(f"after_{i}")

print(f"Bad patches: {len(bad_patches)}")
if bad_patches:
    print("First 10:", bad_patches[:10])
else:
    print("No nan/inf in patches")
    
# Also check band 4 specifically
sample = np.load('data/patches/before/00001.npy').astype(np.float32)
for b in range(4):
    print(f"Band {b+1}: min={sample[:,:,b].min():.1f} max={sample[:,:,b].max():.1f} mean={sample[:,:,b].mean():.1f}")