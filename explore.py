import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load the images
before = rasterio.open('data/patna_before_flood_2023.tif')
after = rasterio.open('data/patna_after_flood_2023.tif')
mask = rasterio.open('data/patna_flood_mask_2023.tif')

# Read RGB bands (B4=red, B3=green, B2=blue are bands 1,2,3)
before_rgb = np.stack([before.read(1), before.read(2), before.read(3)], axis=-1)
after_rgb = np.stack([after.read(1), after.read(2), after.read(3)], axis=-1)
flood_mask = mask.read(1)

# Normalize for display
def normalize(img):
    img = img.astype(float)
    img = np.clip(img / 3000, 0, 1)
    return img

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(normalize(before_rgb))
axes[0].set_title('Before Flood (May-Jun 2023)')
axes[0].axis('off')

axes[1].imshow(normalize(after_rgb))
axes[1].set_title('After Flood (Aug-Sep 2023)')
axes[1].axis('off')

axes[2].imshow(flood_mask, cmap='Blues')
axes[2].set_title('Flood Mask (Ground Truth)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('data/flood_visualization.png', dpi=150)
plt.show()

print(f"Image shape: {before_rgb.shape}")
print(f"Mask unique values: {np.unique(flood_mask)}")
print(f"Flooded pixels: {np.sum(flood_mask == 1)}")
print(f"Non-flooded pixels: {np.sum(flood_mask == 0)}")