import torch
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '../satellite-change-detection/models/best_model.pth'
PATCH_SIZE = 256

def load_model():
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=8,
        classes=1,
        activation=None
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(path):
    with rasterio.open(path) as src:
        img = np.stack([src.read(i) for i in range(1, 5)], axis=-1).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=3000.0, neginf=0.0)
    img = np.clip(img / 3000.0, 0, 1)
    return img

def predict(model, before_path, after_path):
    before = preprocess_image(before_path)
    after = preprocess_image(after_path)

    h, w = before.shape[:2]
    flood_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - PATCH_SIZE, PATCH_SIZE // 2):
        for x in range(0, w - PATCH_SIZE, PATCH_SIZE // 2):
            before_patch = before[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            after_patch = after[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            image = np.concatenate([before_patch, after_patch], axis=-1)
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(image)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()

            flood_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    count_map = np.maximum(count_map, 1)
    flood_map = flood_map / count_map
    binary_map = (flood_map > 0.5).astype(np.uint8)

    flooded_pixels = binary_map.sum()
    area_km2 = round(flooded_pixels * 100 / 1e6, 2)  # 10m pixel = 100m²

    return flood_map, binary_map, area_km2

def visualize(before_path, after_path, flood_map, binary_map, area_km2):
    with rasterio.open(before_path) as src:
        before_rgb = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
    with rasterio.open(after_path) as src:
        after_rgb = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)

    def norm(x):
        return np.clip(x / 3000.0, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Flood Detection Result — Flooded Area: {area_km2} sq km', fontsize=14)

    axes[0].imshow(norm(before_rgb))
    axes[0].set_title('Before Flood')
    axes[0].axis('off')

    axes[1].imshow(norm(after_rgb))
    axes[1].set_title('After Flood')
    axes[1].axis('off')

    axes[2].imshow(norm(after_rgb))
    axes[2].imshow(binary_map, alpha=0.5, cmap='Blues')
    axes[2].set_title(f'Flood Mask Overlay\n{area_km2} sq km flooded')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('flood_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Flooded area: {area_km2} sq km")
    print("Result saved to flood_result.png")

if __name__ == '__main__':
    model = load_model()
    before_path = '../satellite-change-detection/data/patna_before_flood_2023.tif'
    after_path = '../satellite-change-detection/data/patna_after_flood_2023.tif'
    flood_map, binary_map, area_km2 = predict(model, before_path, after_path)
    visualize(before_path, after_path, flood_map, binary_map, area_km2)