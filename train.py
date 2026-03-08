import torch
import segmentation_models_pytorch as smp
from dataset import get_dataloaders
import numpy as np
import os

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on: {DEVICE}")

train_loader, val_loader = get_dataloaders(BATCH_SIZE)

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=8,
    classes=1,
    activation=None
).to(DEVICE)

loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
bce_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(DEVICE))

def combined_loss(pred, target):
    return loss_fn(pred, target) + 0.5 * bce_fn(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

os.makedirs('models', exist_ok=True)
best_iou = 0

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    scheduler.step()

    model.eval()
    val_ious = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            iou = iou_score(outputs, masks)
            val_ious.append(iou.item())

    avg_loss = np.mean(train_losses)
    avg_iou = np.mean(val_ious)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f}")

    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f"  --> Best model saved (IoU: {best_iou:.4f})")

print(f"\nTraining complete. Best IoU: {best_iou:.4f}")