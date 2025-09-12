from torch.utils.data import Dataset
from PIL import Image,ImageOps
import numpy as np
import torchvision.transforms as transforms
import os
from torchvision.transforms import functional as F
from PIL import ImageFile
import cv2 as cv

class PlantDataset(Dataset):
    def __init__(self, image_path, mask_path, limit=None, augment_with_opencv=True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.limit = limit
        # Get image and mask files directly from the provided paths
        self.images = sorted([os.path.join(image_path, i) for i in os.listdir(image_path)])
        self.masks = sorted([os.path.join(mask_path, i) for i in os.listdir(mask_path)])
        if self.limit is not None:
            self.images = self.images[:self.limit]
            self.masks = self.masks[:self.limit]
        self.augment_with_opencv = augment_with_opencv
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])


    def apply_red_enhancement(self, pil_image):
        """Enhance red lesions using OpenCV"""
        # Convert to OpenCV
        img_array = np.array(pil_image)
        hsv = cv.cvtColor(img_array, cv.COLOR_RGB2HSV)
        
        # Red detection parameters
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([179, 255, 255])
            
        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv.bitwise_or(mask1, mask2)

        # Convert to PIL Image (mode "L" = grayscale)
        return Image.fromarray(red_mask).convert("L")
        
    def __getitem__(self, index):
            try:
                img = Image.open(self.images[index]).convert("RGB")
                img = ImageOps.exif_transpose(img)
                
                
                
                # Apply OpenCV enhancement
                if self.augment_with_opencv:
                    mask = Image.open(self.masks[index]).convert("RGB")
                    mask = ImageOps.exif_transpose(mask)
                    mask = self.apply_red_enhancement(mask)
                else:
                    mask = Image.open(self.masks[index]).convert("L")
                    mask = ImageOps.exif_transpose(mask)  
                # Apply transforms
                img_tensor = self.transform(img)
                mask_tensor = self.mask_transform(mask)
                mask_tensor = (mask_tensor != 0).float()  # Ensure binary mask

                
                img_filename = os.path.basename(self.images[index])
                return img_tensor, mask_tensor, img_filename
                
            except Exception as e:
                return self.__getitem__((index + 1) % len(self))
    def __len__(self):
    
        return min(len(self.images),len(self.masks))
    
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

