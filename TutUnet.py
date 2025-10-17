import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import PlantDataset
from unet import Unet4
from metric import compute_metrics

# Torchmetrics
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.segmentation import DiceScore

import torchvision.transforms as transforms

# -----------------------------
# Dataset Loading and Split
# -----------------------------
def find_folders(base_path, target_folder):
    results = []
    for root, dirs, files in os.walk(base_path):
        if target_folder in dirs:
            results.append(os.path.join(root, target_folder))
    return results

base_path = os.getcwd()
data_paths = find_folders(base_path, "Data")

if data_paths:
    plant_data_path = data_paths[0]
    natural_path = os.path.join(plant_data_path, "Directory 1")
    mask_path = os.path.join(plant_data_path, "Directory 2")
else:
    raise FileNotFoundError("Data folder not found!")
# Training augmentations (stronger)
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Validation augmentations (mild)
val_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_val_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Test (pure, no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = PlantDataset(natural_path, mask_path, limit=1308, augment_with_opencv=False)

# Split dataset into train/val/test
generator = torch.Generator().manual_seed(25)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * train_size)  # 10% of train for validation
train_size -= val_size
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_dataset = PlantDataset(natural_path, mask_path, limit=len(train_dataset))
train_dataset.transform = train_transforms
train_dataset.mask_transform = mask_train_transforms

val_dataset = PlantDataset(natural_path, mask_path, limit=len(val_dataset))
val_dataset.transform = val_transforms
val_dataset.mask_transform = mask_val_transforms

test_dataset = PlantDataset(natural_path, mask_path, limit=len(test_dataset))
test_dataset.transform = test_transforms
test_dataset.mask_transform = mask_test_transforms

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, generator=generator, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# UNet Model
# -----------------------------
model = Unet4().to(device)

# -----------------------------
# Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            scheduler.step(avg_train_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}")

    return model, train_loss_history, val_loss_history

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_test_set(model, data_loader, device):
    model.eval()
    iou_metric = BinaryJaccardIndex()
    dice_metric = DiceScore(num_classes=2)
    f1_metric = BinaryF1Score()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()

    iou_scores, dice_scores, f1_scores, precision_scores, recall_scores, error_scores, mse_scores = [], [], [], [], [], [], []

    with torch.no_grad():
        for images, masks, *_ in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            iou_scores.append(iou_metric(preds, masks).item())
            dice_scores.append(dice_metric(preds, masks).item())
            f1_scores.append(f1_metric(preds, masks).item())
            precision_scores.append(precision_metric(preds, masks).item())
            recall_scores.append(recall_metric(preds, masks).item())
            _, _, _, error, mse = compute_metrics(preds, masks)
            error_scores.append(error)
            mse_scores.append(mse)

    print(f"\nEvaluation Metrics:")
    print(f"  IoU: {np.mean(iou_scores):.4f}")
    print(f"  Dice: {np.mean(dice_scores):.4f}")
    print(f"  F1: {np.mean(f1_scores):.4f}")
    print(f"  Precision: {np.mean(precision_scores):.4f}")
    print(f"  Recall: {np.mean(recall_scores):.4f}")
    print(f"  Error Rate: {np.mean(error_scores):.4f}")
    print(f"  MSE: {np.mean(mse_scores):.4f}")

# -----------------------------
# Visualization
# -----------------------------
def visualize_predictions(model, dataset, device, num_samples=3, train_loss=None, val_loss=None):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        image, true_mask, filename = dataset[idx]
        with torch.no_grad():
            pred_mask = model(image.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        pred_bin = (pred_mask > 0.5).astype(np.uint8)
        image_np = image.permute(1,2,0).cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()
        intersection = np.logical_and(pred_bin, true_mask_np)
        union = np.logical_or(pred_bin, true_mask_np)
        iou = np.sum(intersection) / (np.sum(union)+1e-8)
        dice = (2*np.sum(intersection))/(np.sum(pred_bin)+np.sum(true_mask_np)+1e-8)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.imshow(true_mask_np, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.imshow(pred_bin, cmap='gray')
        plt.title(f"Predicted\nIoU: {iou:.3f} Dice: {dice:.3f}")
        plt.axis("off")
        plt.show()

    # Plot loss curves
    if train_loss and val_loss:
        plt.figure(figsize=(10,5))
        plt.plot(range(1,len(train_loss)+1), train_loss, label="Train Loss")
        plt.plot(range(1,len(val_loss)+1), val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

# -----------------------------
# Run Pipeline
# -----------------------------
start_time = time.time()
trained_model, train_loss_history, val_loss_history = train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-3)
end_time = time.time()
elapsed = end_time - start_time
print(f"Training time: {int(elapsed//3600)}h {(int(elapsed%3600)//60)}min {int(elapsed%60)}s")

# Evaluate
print("\n--- Train Set Metrics ---")
evaluate_test_set(trained_model, train_loader, device)

print("\n--- Test Set Metrics ---")
evaluate_test_set(trained_model, test_loader, device)

# Visualize predictions
visualize_predictions(trained_model, test_dataset, device, num_samples=5, train_loss=train_loss_history, val_loss=val_loss_history)

# Save model checkpoint
os.makedirs("models", exist_ok=True)
torch.save({
    "model_state_dict": trained_model.state_dict(),
    "train_loss_history": train_loss_history,
    "val_loss_history": val_loss_history
}, "models/pestnet_checkpoint.pth")
print("Model saved at models/pestnet_checkpoint.pth")
