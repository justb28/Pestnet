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

dataset = PlantDataset(natural_path, mask_path, limit=1308)  # Adjust limit as needed

dataset.save_rotated_images(output_natural_dir="AugmentedImages", output_mask_dir="AugmentedMasks")
