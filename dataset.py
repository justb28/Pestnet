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


    def apply_lesion_detection(self, pil_image, plant_mask_pil):
        """Detects red/brown lesions ONLY on the plant area."""
        img_array = np.array(pil_image)
        plant_mask_array = np.array(plant_mask_pil)
        
        # 1. Convert to HSV
        hsv = cv.cvtColor(img_array, cv.COLOR_RGB2HSV)
        
        # 2. Define Red/Brown Lesion Thresholds (you'll need to fine-tune this)
        # Red (0-10) and Yellow/Brown (10-30) in Hue
        lower_lesion1 = np.array([0, 50, 50])  # Red lower
        upper_lesion1 = np.array([10, 255, 255]) # Red upper
        
        lower_lesion2 = np.array([160, 50, 50]) # Red wrap-around
        upper_lesion2 = np.array([179, 255, 255])
        
        # You might also need a mask for brown/yellow lesions (e.g., H 15-30)
            
        mask1 = cv.inRange(hsv, lower_lesion1, upper_lesion1)
        mask2 = cv.inRange(hsv, lower_lesion2, upper_lesion2)
        lesion_mask_raw = cv.bitwise_or(mask1, mask2)
        
        # 3. Apply the plant segmentation mask to ensure only plant pixels are considered
        # The plant_mask_array must be the same size and a single channel (0/255)
        # We perform a bitwise AND between the lesion detection and the plant mask
        
        # Ensure plant_mask_array is a binary mask (0 or 255)
        plant_mask_binary = (plant_mask_array > 0).astype(np.uint8) * 255
        
        final_lesion_mask = cv.bitwise_and(lesion_mask_raw, lesion_mask_raw, mask=plant_mask_binary)

        # Apply morphological opening to clean up noise
        kernel = np.ones((3,3), np.uint8)
        final_lesion_mask = cv.morphologyEx(final_lesion_mask, cv.MORPH_OPEN, kernel)
        
        # Convert to PIL Image
        return Image.fromarray(final_lesion_mask).convert("L")
        
    def __getitem__(self, index):
            try:
                img = Image.open(self.images[index]).convert("RGB")
                img = ImageOps.exif_transpose(img)
                
                # --- Mask Loading ---
                # Mask should be loaded as a single channel (L) since it's a binary/grayscale target
                mask = Image.open(self.masks[index]).convert("L")
                mask = ImageOps.exif_transpose(mask)  
                    
                # --- Enhancement/Augmentation (Optional) ---
                # If you want to use the red enhancement logic to CREATE a mask, 
                # you must first apply the plant segmentation mask to the image.
                
                # This is complex and usually better done as a separate, pre-generated dataset.
                # For a standard setup, we skip the apply_red_enhancement inside __getitem__.
                
                # Apply transforms
                img_tensor = self.transform(img)
                mask_tensor = self.mask_transform(mask)
                
                # Ensure binary mask: 1.0 for lesion, 0.0 otherwise
                # Note: 0.5 is a common threshold for Image.NEAREST upsampling artifacts
                mask_tensor = (mask_tensor > 0.5).float() 

                img_filename = os.path.basename(self.images[index])
                return img_tensor, mask_tensor, img_filename
                
            except Exception as e:
                # Handle error (original code is fine here)
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

