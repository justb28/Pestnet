import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the mini U-Net model
class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PestNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(PestNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Use fewer filters than the original U-Net to make it "mini"
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Create a dataset class for plant disease segmentation
class PlantDiseaseDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (assuming it's a grayscale image where disease pixels are white)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize to [0, 1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add channel dimension to mask
        mask = mask.unsqueeze(0)

        return image, mask

# Define transforms
def get_transforms(height=256, width=256):
    train_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform

# Dice coefficient for evaluation
def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

# IoU/Jaccard score for evaluation
def iou_score(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

# Binary cross entropy with dice loss
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Binary Cross Entropy loss
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice = 1 - (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)

        return BCE + dice

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_val_score = 0.0
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        print(f'Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0
        val_running_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                # Calculate IoU and Dice
                preds = torch.sigmoid(outputs) > 0.5
                val_running_iou += iou_score(masks, preds).item() * images.size(0)
                val_running_dice += dice_coef(masks, preds).item() * images.size(0)
                val_running_loss += loss.item() * images.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_iou = val_running_iou / len(val_loader.dataset)
        val_epoch_dice = val_running_dice / len(val_loader.dataset)

        val_losses.append(val_epoch_loss)
        val_ious.append(val_epoch_iou)
        val_dices.append(val_epoch_dice)

        print(f'Validation Loss: {val_epoch_loss:.4f}, IoU: {val_epoch_iou:.4f}, Dice: {val_epoch_dice:.4f}')

        # Save best model
        if val_epoch_iou > best_val_score:
            best_val_score = val_epoch_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")

    return train_losses, val_losses, val_ious, val_dices

# Function to visualize model predictions
def visualize_prediction(model, image_path, mask_path=None, transform=None):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transform
    if transform:
        if mask_path:
            # Load mask if provided
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0
            augmented = transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask = augmented['mask']
        else:
            augmented = transform(image=image)
            image_tensor = augmented['image']
    else:
        # Simple transform if none provided
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(image_tensor)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        pred = torch.sigmoid(output) > 0.5
        pred = pred.squeeze().cpu().numpy()

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    if mask_path:
        plt.subplot(1, 3, 3)
        if transform:
            plt.imshow(mask, cmap='gray')
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
            plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    # Parameters
    img_height, img_width = 400, 400  # Your images are 400x400
    batch_size = 8
    num_epochs = 25
    learning_rate = 0.001

    # Setup data loading with your directories
    img_dir = 'Data/Data1/Natural'  # Path to natural images
    mask_dir = 'Data/Data1/Mask'  # Path to infected area masks

    # Get all image filenames from Directory7 (since it has fewer images)
    mask_filenames = [f for f in os.listdir(mask_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Match these with images in Directory1
    img_paths = []
    mask_paths = []

    for mask_filename in mask_filenames:
        # Adjust this matching logic based on your actual filename patterns
        img_filename = mask_filename  # If filenames match exactly
        # img_filename = mask_filename.replace('_mask', '')  # If there's a naming pattern

        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)

        # Only add if the image exists
        if os.path.exists(img_path):
            img_paths.append(img_path)
            mask_paths.append(mask_path)

    print(f"Found {len(img_paths)} matching image-mask pairs")

    # Check if we have data to proceed
    if not img_paths:
        print("No image paths found. Please update the paths in the code or provide sample data.")
        print("Showing how to use the model with sample data...")

        # Initialize model for demonstration
        model = PestNet(n_channels=3, n_classes=1).to(device)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        return

    # Split data
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42)

    # Create transforms
    train_transform, val_transform = get_transforms(img_height, img_width)

    # Create datasets
    train_dataset = PlantDiseaseDataset(train_img_paths, train_mask_paths, transform=train_transform)
    val_dataset = PlantDiseaseDataset(val_img_paths, val_mask_paths, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = PestNet(n_channels=3, n_classes=1).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Define loss function and optimizer
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_losses, val_losses, val_ious, val_dices = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

    # Plot training results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(val_ious, label='Val IoU')
    plt.legend()
    plt.title('IoU')

    plt.subplot(1, 3, 3)
    plt.plot(val_dices, label='Val Dice')
    plt.legend()
    plt.title('Dice Coefficient')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Visualize a prediction
    if val_img_paths:
        sample_img_path = val_img_paths[0]
        sample_mask_path = val_mask_paths[0]
        visualize_prediction(model, sample_img_path, sample_mask_path, val_transform)

if __name__ == "__main__":
    main()