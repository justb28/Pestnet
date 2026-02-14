from torch.utils.data import Dataset
from PIL import Image,ImageOps
import numpy as np
import torchvision.transforms as transforms
import os
from torchvision.transforms import functional as F
from PIL import ImageFile
import cv2 as cv
import random

class PlantDataset(Dataset):
    def __init__(self, image_path, mask_path, limit=None, transform=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.limit = limit
        # Get image and mask files directly from the provided paths
        self.images = sorted([os.path.join(image_path, i) for i in os.listdir(image_path)])
        self.masks = sorted([os.path.join(mask_path, i) for i in os.listdir(mask_path)])
        if self.limit is not None:
            self.images = self.images[:self.limit]
            self.masks = self.masks[:self.limit]
        self.transform = transform

    def __getitem__(self, index):
            try:
                img = Image.open(self.images[index]).convert("RGB")
                img = ImageOps.exif_transpose(img)
                
                # --- Mask Loading ---
                # Mask should be loaded as a single channel (L) since it's a binary/grayscale target
                mask = Image.open(self.masks[index]).convert("L")
                mask = ImageOps.exif_transpose(mask)  
                
                if self.transform:
                    img, mask = self.transform(img, mask)
                else:
                    # fallback if no paired transform is passed
                    img = F.to_tensor(F.resize(img, (128, 128))) 
                    mask = F.to_tensor(F.resize(mask, (128, 128)))
                    mask = (mask != 0).float()

                img_filename = os.path.basename(self.images[index])
                return img, mask, img_filename
                
            except Exception as e:
                # Handle errors by skipping to the next item
                return self.__getitem__((index + 1) % len(self))
    def __len__(self):
    
        return min(len(self.images),len(self.masks))


    def save_rotated_images(image_dir, mask_dir, output_image_dir, output_mask_dir):
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        angles = [90, 180, 270]

        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(image_dir, img_name)
            mask_path  = os.path.join(mask_dir, img_name)

            img = Image.open(image_path)
            mask = Image.open(mask_path)

            # Save original
            img.save(os.path.join(output_image_dir, img_name))
            mask.save(os.path.join(output_mask_dir, img_name))

            # Save rotated versions
            for angle in angles:
                rotated_img = img.rotate(angle)
                rotated_mask = mask.rotate(angle)

                base_name = os.path.splitext(img_name)[0]
                ext = os.path.splitext(img_name)[1]

                rotated_img.save(os.path.join(output_image_dir, f"{base_name}_rot{angle}{ext}"))
                rotated_mask.save(os.path.join(output_mask_dir, f"{base_name}_rot{angle}{ext}"))

class PairedTransform:
    def __init__(self, resize=(128,128), augment=False):
        self.resize = resize
        self.augment = augment

    def __call__(self, image, mask):
        # Resize first
        image = F.resize(image, self.resize)
        mask = F.resize(mask, self.resize)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            # Random rotation
            angle = random.choice([0, 90, 180, 270])
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        # Convert to tensor (keeps mask aligned)
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        mask = (mask != 0).float()  # Binarize mask
        return image, mask

