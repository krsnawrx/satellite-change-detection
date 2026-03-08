import numpy as np
import os
for i in range(1, 6):
    before = np.load(f'data/patches/before/{i:05d}.npy').astype(np.float32)
    after = np.load(f'data/patches/after/{i:05d}.npy').astype(np.float32)
    print(f"Patch {i} - Before min:{before.min():.1f} max:{before.max():.1f} | After min:{after.min():.1f} max:{after.max():.1f}")