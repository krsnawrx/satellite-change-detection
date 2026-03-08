import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

class FloodDataset(Dataset):
    def __init__(self, indices, split='train'):
        self.indices = indices
        self.split = split
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        before = np.load(f'data/patches/before/{i:05d}.npy').astype(np.float32)
        after = np.load(f'data/patches/after/{i:05d}.npy').astype(np.float32)
        mask = np.load(f'data/patches/mask/{i:05d}.npy').astype(np.float32)
        before = np.nan_to_num(before, nan=0.0, posinf=3000.0, neginf=0.0)
        after = np.nan_to_num(after, nan=0.0, posinf=3000.0, neginf=0.0)
        before = np.clip(before / 3000.0, 0, 1)
        after = np.clip(after / 3000.0, 0, 1)
        image = np.concatenate([before, after], axis=-1)
        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)
        return image, mask
def get_dataloaders(batch_size=8):
    total = len(os.listdir('data/patches/before'))
    indices = list(range(1, total + 1))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_ds = FloodDataset(train_idx, 'train')
    val_ds = FloodDataset(val_idx, 'val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")
    return train_loader, val_loader
if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()
    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  # should be (8, 8, 256, 256)
    print(f"Batch mask shape: {masks.shape}")    # should be (8, 1, 256, 256)
    print("Dataset ready!")