import os
import torch
from torch.utils.data import random_split, DataLoader
from dataset import PlantDataset
from unet import Unet4
from train import train_model
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import torchvision.utils as vutils
from tqdm import tqdm  # progress bar
import random
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


dataset = PlantDataset(natural_path, mask_path, limit=1308, augment_with_opencv=False)
model = Unet4()
model.load_state_dict(torch.load("models/pestnet1.pth", map_location=device))
model.to(device)
model.eval()

# make directories
os.makedirs("predictions", exist_ok=True)
os.makedirs("applied_mask", exist_ok=True)

indices = random.sample(range(len(dataset)), len(dataset))
for idx in tqdm(indices):
    image, true_mask, filename = dataset[idx]

    # prepare input
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(image_input)

    # convert to numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)  # binary mask [H,W]

    # original image back to numpy [H,W,C]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    appl_mask = (image_np * pred_mask_bin[..., np.newaxis] * 255).astype(np.uint8)

    base, _ = os.path.splitext(filename)

    # --- save predicted mask ---
    pred_tensor = torch.from_numpy(pred_mask_bin).unsqueeze(0)  # [1,H,W]
    save_path = os.path.join("predictions", f"{base}_pred.jpg")
    vutils.save_image(pred_tensor.float(), save_path)

    # --- save applied mask (overlayed image) ---
    save_path = os.path.join("applied_mask", f"{base}_applmask.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(appl_mask, cv2.COLOR_RGB2BGR))
