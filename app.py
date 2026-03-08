import streamlit as st
import torch
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH_SIZE = 256

st.set_page_config(page_title="Bihar Flood Mapper", layout="wide")

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    import os
    model_path = hf_hub_download(
        repo_id='krsnawrx/bihar-flood-mapper',
        filename='best_model.pth',
        repo_type='model',
        token=os.environ.get('HF_TOKEN')
    )
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=8,
        classes=1,
        activation=None
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess(path):
    with rasterio.open(path) as src:
        img = np.stack([src.read(i) for i in range(1, 5)], axis=-1).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=3000.0, neginf=0.0)
    return np.clip(img / 3000.0, 0, 1)

def predict(model, before, after):
    h, w = before.shape[:2]
    flood_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - PATCH_SIZE, PATCH_SIZE // 2):
        for x in range(0, w - PATCH_SIZE, PATCH_SIZE // 2):
            bp = before[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            ap = after[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            image = np.concatenate([bp, ap], axis=-1)
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = torch.sigmoid(model(image)).squeeze().cpu().numpy()
            flood_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    count_map = np.maximum(count_map, 1)
    flood_map /= count_map
    binary = (flood_map > 0.5).astype(np.uint8)
    area_km2 = round(binary.sum() * 100 / 1e6, 2)
    return flood_map, binary, area_km2

def norm(x):
    return np.clip(x, 0, 1)

# UI
st.title("Bihar Flood Mapper")
st.markdown("Upload pre and post flood Sentinel-2 GeoTIFF images to detect flooded areas using a U-Net deep learning model.")

col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Before flood image (.tif)", type=['tif', 'tiff'])
with col2:
    after_file = st.file_uploader("After flood image (.tif)", type=['tif', 'tiff'])

st.markdown("---")
use_demo = st.checkbox("Use demo data (Patna 2023 floods)", value=True)

if st.button("Run Detection", type="primary"):
    model = load_model()

    if use_demo:
        from huggingface_hub import hf_hub_download
        import os
        token = os.environ.get('HF_TOKEN')
        before_path = hf_hub_download(
            repo_id='krsnawrx/bihar-flood-mapper',
            filename='patna_before_flood_2023.tif',
            repo_type='model',
            token=token
        )
        after_path = hf_hub_download(
            repo_id='krsnawrx/bihar-flood-mapper',
            filename='patna_after_flood_2023.tif',
            repo_type='model',
            token=token
        )
    elif before_file and after_file:
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(before_file.read())
            before_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(after_file.read())
            after_path = f.name
    else:
        st.error("Please upload both images or use demo data.")
        st.stop()

    with st.spinner("Running flood detection..."):
        before = preprocess(before_path)
        after = preprocess(after_path)
        flood_map, binary, area_km2 = predict(model, before, after)

    st.success("Detection complete")
    st.metric("Flooded Area", f"{area_km2} sq km")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    with rasterio.open(before_path) as src:
        before_rgb = norm(np.stack([src.read(1), src.read(2), src.read(3)], axis=-1) / 3000.0)
    with rasterio.open(after_path) as src:
        after_rgb = norm(np.stack([src.read(1), src.read(2), src.read(3)], axis=-1) / 3000.0)

    axes[0].imshow(before_rgb)
    axes[0].set_title('Before Flood', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(after_rgb)
    axes[1].set_title('After Flood', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(after_rgb)
    axes[2].imshow(binary, alpha=0.5, cmap='Blues')
    axes[2].set_title(f'Flood Detection — {area_km2} sq km flooded', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Technical Details")
    st.markdown("""
    - **Data:** Sentinel-2 multispectral imagery, 10m resolution, accessed via Google Earth Engine
    - **Model:** U-Net with ResNet34 encoder, pretrained on ImageNet, fine-tuned on Bihar flood data
    - **Input:** 8 channels (4 spectral bands x before + after image pair)
    - **Output:** Binary flood mask with flooded area quantification in sq km
    - **Validation IoU:** 0.6477
    """)