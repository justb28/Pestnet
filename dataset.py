from torch.utils.data import Dataset
from PIL import Image,ImageOps
import numpy as np
import torchvision.transforms as transforms
import os
from torchvision.transforms import functional as F
from PIL import ImageFile

class PlantDataset(Dataset):
    def __init__(self, image_path, mask_path, limit=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.limit = limit
        # Get image and mask files directly from the provided paths
        self.images = sorted([os.path.join(image_path, i) for i in os.listdir(image_path)])
        self.masks = sorted([os.path.join(mask_path, i) for i in os.listdir(mask_path)])
        if self.limit is not None:
            self.images = self.images[:self.limit]
            self.masks = self.masks[:self.limit]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index]).convert("RGB")
            img = ImageOps.exif_transpose(img)

            mask = Image.open(self.masks[index]).convert("L")
            mask = ImageOps.exif_transpose(mask)

            # Apply transforms
            img_tensor = self.transform(img)
            mask_tensor = self.transform(mask)
            mask_tensor = (mask_tensor != 0).float()  # Ensure binary mask

            # Extract filename
            img_filename = os.path.basename(self.images[index])
            return img_tensor, mask_tensor,img_filename
        except Exception as e:
            # Return a placeholder or the next valid image
            return self.__getitem__((index + 1) % len(self))
    
    def __len__(self):
        return len(self.images)
    
    def save_rotated_images(self, output_natural_dir="FlippedNatural", output_mask_dir="FlippedMask"):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        os.makedirs(output_natural_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        rotations = [0, 90, 180, 270]
        for idx in range(len(self.images)):
            try:
                img = Image.open(self.images[idx]).convert("RGB")
                img = ImageOps.exif_transpose(img)
                mask = Image.open(self.masks[idx]).convert("L")
                mask = ImageOps.exif_transpose(mask)
            except Exception as e:
                print(f"Skipping file {self.images[idx]} or {self.masks[idx]} due to error: {e}")
                continue
            base_img_name = os.path.basename(self.images[idx])
            base_mask_name = os.path.basename(self.masks[idx])
            for angle in rotations:
                rotated_img = F.rotate(img, angle)
                rotated_mask = F.rotate(mask, angle)
                img_filename = f"{os.path.splitext(base_img_name)[0]}_rot{angle}.png"
                mask_filename = f"{os.path.splitext(base_mask_name)[0]}_rot{angle}.png"
                rotated_img.save(os.path.join(output_natural_dir, img_filename))
                rotated_mask.save(os.path.join(output_mask_dir, mask_filename))

