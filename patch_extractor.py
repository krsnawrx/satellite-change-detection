import rasterio
import numpy as np
import os

# Settings
PATCH_SIZE = 256
STRIDE = 128  # 50% overlap between patches
MIN_FLOOD_PIXELS = 10  # skip patches with almost no flood pixels

# Load images
before = rasterio.open('data/patna_before_flood_2023.tif')
after = rasterio.open('data/patna_after_flood_2023.tif')
mask = rasterio.open('data/patna_flood_mask_2023.tif')

# Read all bands
before_img = np.stack([before.read(i) for i in range(1, 5)], axis=-1)  # 4 bands
after_img = np.stack([after.read(i) for i in range(1, 5)], axis=-1)    # 4 bands
mask_img = mask.read(1)                                                  # 1 band

print(f"Before shape: {before_img.shape}")
print(f"After shape: {after_img.shape}")
print(f"Mask shape: {mask_img.shape}")

# Create output dirs
os.makedirs('data/patches/before', exist_ok=True)
os.makedirs('data/patches/after', exist_ok=True)
os.makedirs('data/patches/mask', exist_ok=True)

h, w = mask_img.shape
patch_count = 0
flood_patch_count = 0

for y in range(0, h - PATCH_SIZE, STRIDE):
    for x in range(0, w - PATCH_SIZE, STRIDE):
        # Extract patches
        before_patch = before_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        after_patch = after_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        mask_patch = mask_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        # Skip patches with nodata (zeros across all bands)
        if np.sum(before_patch) == 0 or np.sum(after_patch) == 0:
            continue

        patch_count += 1
        if np.sum(mask_patch) >= MIN_FLOOD_PIXELS:
            flood_patch_count += 1

        # Save
        np.save(f'data/patches/before/{patch_count:05d}.npy', before_patch)
        np.save(f'data/patches/after/{patch_count:05d}.npy', after_patch)
        np.save(f'data/patches/mask/{patch_count:05d}.npy', mask_patch)

print(f"\nTotal patches: {patch_count}")
print(f"Patches with flood pixels: {flood_patch_count}")
print(f"Patches saved to data/patches/")