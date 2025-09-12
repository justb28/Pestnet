import torch
from torch.utils.data import DataLoader, random_split
from unet import Unet4,UNet
from dataset import PlantDataset
from train import train_model
import os
from glob import glob
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from visualise import visualize_predictions
from metric import compute_metrics
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.segmentation import DiceScore
import time


def evaluate_test_set(model, data_loader, device):
    model.eval()
    iou_metric = BinaryJaccardIndex()
    dice_metric = DiceScore(num_classes=2)
    f1_metric = BinaryF1Score()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()

    iou_scores = []
    dice_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    error_scores = []
    mse_scores=[]
    with torch.no_grad():
        for images, masks, *_ in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            #outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()
            # torchmetrics expects shape [B, H, W] or [B, 1, H, W]
            iou = iou_metric(preds, masks)
            dice = dice_metric(preds, masks)
            f1 = f1_metric(preds, masks)
            precision = precision_metric(preds, masks)
            recall = recall_metric(preds, masks)
            iou_scores.append(iou.item())
            dice_scores.append(dice.item())
            f1_scores.append(f1.item())
            precision_scores.append(precision.item())
            recall_scores.append(recall.item())
            # Error (from your custom metric)
            _, _, _, error , mse= compute_metrics(preds, masks)
            error_scores.append(error)
            mse_scores.append(mse)
    print(f"Test set evaluation:")
    print(f"  Average IoU = {np.mean(iou_scores):.4f}")
    print(f"  Average Dice = {np.mean(dice_scores):.4f}")
    print(f"  Average F1 = {np.mean(f1_scores):.4f}")
    print(f"  Average Precision = {np.mean(precision_scores):.4f}")
    print(f"  Average Recall = {np.mean(recall_scores):.4f}")
    print(f"  Average Error Rate = {np.mean(error_scores):.4f}")
    print(f"  Average mean square error = {np.mean(mse_scores):.4f}")


def find_folders(base_path, target_folder):
    """Search for a target folder in the directory structure"""
    print(f"Searching for '{target_folder}' from base path: {base_path}")
    results = []
    
    for root, dirs, files in os.walk(base_path):
        if target_folder in dirs:
            found_path = os.path.join(root, target_folder)
            results.append(found_path)
            print(f"Found: {found_path}")
    
    return results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = os.getcwd()
print(f"Current working directory: {base_path}")

# Search for the plant_data folders
data_paths = find_folders(base_path, "Data")

if not data_paths:
    # Try looking for specific folders
    natural_paths = find_folders(base_path, "Directory 1")
    mask_paths = find_folders(base_path, "Directory 2")

    if natural_paths and mask_paths:
        natural_path = natural_paths[0]
        mask_path = mask_paths[0]
    else:
        print("Could not find data folders automatically.")
        # Hard code the paths based on your screenshot
        natural_path = os.path.join(base_path, "UNET_ENV", "Data", "Directory 1")
        mask_path = os.path.join(base_path, "UNET_ENV", "Data", "Directory 2")
else:
    # Construct paths based on found plant_data folder
    plant_data_path = data_paths[0]
    natural_path = os.path.join(plant_data_path, "Directory 1")
    mask_path = os.path.join(plant_data_path, "Directory 2")

print(f"Using image path: {natural_path}")
print(f"Using mask path: {mask_path}")

# Check if directories exist
if not os.path.exists(natural_path): 
    raise FileNotFoundError(f"Error: Image directory not found at {natural_path}")


if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Error: Mask directory not found at {mask_path}")

dataset = PlantDataset(natural_path, mask_path, limit=1308,augment_with_opencv=False)  # Adjust limit as needed



# Set random seed for reproducibility
generator = torch.Generator().manual_seed(25)

# Split dataset into 80% train and 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
val_dataset = train_dataset

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    generator=generator, 
    num_workers=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=False, 
    num_workers=2
)

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = Unet4().to(device)
start_time = time.time()
trained_model, loss_history = train_model(model, train_loader, device, num_epochs=30, learning_rate=0.001)
end_time = time.time()
elapsed = int(end_time - start_time)
hours = elapsed // 3600
minutes = (elapsed % 3600) // 60
seconds = elapsed % 60
print(f"Training time: {hours}h {minutes}min {seconds}sec")

# Evaluate train set
print("Evaluating train set...")
evaluate_test_set(trained_model, train_loader, device)


# Evaluate test set
print("Evaluating test set...")
evaluate_test_set(trained_model, test_loader, device)

# Visualize predictions with overlays and IoU
print("Visualizing predictions...")
visualize_predictions(trained_model, dataset, device, num_samples=3, loss_history=loss_history, show_overlay=True, show_iou=True)

# Ensure models directory exists before saving
os.makedirs("models", exist_ok=True)
# Save model
#
torch.save(trained_model.state_dict(), "models/unet4.pth")