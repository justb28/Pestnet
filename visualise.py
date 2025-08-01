import random
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(model, dataset, device, num_samples=3,loss_history=None, show_overlay=True,show_iou=True):
    model.eval()
    
    indices = random.sample(range(len(dataset)), num_samples)
    # Determine number of columns based on whether overlay is shown
    num_cols = 3  # Base: image, ground truth, predicted
    if show_overlay:
        num_cols += 1
    if show_iou:
        num_cols += 1
    
    plt.figure(figsize=(15, num_samples * 4))
    
    for i, idx in enumerate(indices):
        image, true_mask, filename = dataset[idx]

        image_input = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_mask = model(image_input)
        
        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)  # binarize
        image_np = image.permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()
        appl_mask = image_np * pred_mask_bin[..., np.newaxis]
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_mask_bin, true_mask_np)
        union = np.logical_or(pred_mask_bin, true_mask_np)
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
        dice = (2.0 * np.sum(intersection)) / (np.sum(pred_mask_bin) + np.sum(true_mask_np) + 1e-8)

        print(f"IOU for sample {i}: {iou:.4f}\nDice: {dice:.4f}")

        plt.subplot(num_samples, num_cols, i * num_cols + 1)
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis("off")
        
        plt.subplot(num_samples, num_cols, i * num_cols + 2)
        plt.imshow(true_mask_np, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")
        
        plt.subplot(num_samples, num_cols, i * num_cols + 3)
        plt.imshow(pred_mask_bin, cmap='gray')
        plt.title(f"Predicted Mask\n{filename}")
        plt.axis("off")

        if show_overlay:
            plt.subplot(num_samples, num_cols, i * num_cols + 4)
            overlay = np.zeros((image_np.shape[0], image_np.shape[1], 3))
            overlay[:, :, 1] = true_mask_np * 0.7
            overlay[:, :, 0] = pred_mask * 0.7
            plt.imshow(overlay)
            plt.title("Overlay (Green=True, Red=Pred)")
            plt.axis("off")
        if show_iou:
            col_offset = 5 if show_overlay else 4
            plt.subplot(num_samples, num_cols, i * num_cols + col_offset)
            inter_union_img = np.zeros_like(image_np)
            inter_union_img[:, :, :] = 0
            inter_union_img[union] = [0, 0, 1]
            inter_union_img[intersection] = [1, 1, 1]
            plt.imshow(inter_union_img)
            plt.title(f"IoU: {iou:.3f}\n Dice :{dice:.3f}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    if loss_history is not None and len(loss_history) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
        plt.title('Training Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    plt.figure(figsize=(20, 16))
    plt.suptitle("Applied Masks for First 20 Samples", fontsize=10)
    grid_indices = list(range(min(20, len(dataset))))
    rows, cols = 4, 5





    
    for i, idx in enumerate(grid_indices):
        image, true_mask , filename = dataset[idx]
        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = model(image_input)



        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        appl_mask = image_np * pred_mask_bin[..., np.newaxis]
        true_mask_np = true_mask.squeeze().cpu().numpy()
        intersection = np.logical_and(pred_mask_bin, true_mask_np)
        union = np.logical_or(pred_mask_bin, true_mask_np)
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(appl_mask)
        plt.title(f"Sample {idx}\n{filename}\nIoU: {iou:.3f}", fontsize=15)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
