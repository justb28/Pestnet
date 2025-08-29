import torch
from torch.utils.data import DataLoader, random_split
from unet import Unet4, UNet
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
import itertools
import json
from datetime import datetime


def evaluate_model(model, data_loader, device):
    """Evaluate model and return metrics"""
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
    mse_scores = []
    
    with torch.no_grad():
        for images, masks, *_ in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # Calculate metrics
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
            
            # Custom metrics
            _, _, _, error, mse = compute_metrics(preds, masks)
            error_scores.append(error)
            mse_scores.append(mse)
    
    return {
        'iou': np.mean(iou_scores),
        'dice': np.mean(dice_scores),
        'f1': np.mean(f1_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'error_rate': np.mean(error_scores),
        'mse': np.mean(mse_scores)
    }


def grid_search_hyperparameters(dataset, device, param_grid, results_dir="grid_search_results"):
    """
    Perform grid search over hyperparameters
    
    Args:
        dataset: The dataset to use
        device: torch device
        param_grid: Dictionary with lists of parameters to search
        results_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Starting grid search with {len(param_combinations)} combinations...")
    print(f"Parameter grid: {param_grid}")
    
    # Store results
    all_results = []
    best_score = -1
    best_params = None
    best_model_path = None
    
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(25)
    
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"Combination {i+1}/{len(param_combinations)}")
        
        # Create parameter dictionary
        current_params = dict(zip(param_names, params))
        print(f"Parameters: {current_params}")
        
        try:
            # Extract parameters
            epochs = current_params['epochs']
            batch_size = current_params['batch_size']
            learning_rate = current_params['learning_rate']
            
            # Split dataset (80% train, 20% validation for grid search)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
            
            # Create data loaders with current batch size
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                generator=generator, 
                num_workers=2
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=2
            )
            
            # Initialize fresh model
            model = Unet4().to(device)
            
            # Train model
            print(f"Training with epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            start_time = time.time()
            
            # Note: You'll need to modify your train_model function to accept learning_rate parameter
            # For now, I'll assume it can be passed as an argument
            trained_model, loss_history = train_model(
                model, train_loader, device, 
                num_epochs=epochs, 
                learning_rate=learning_rate  # Add this parameter to train_model function
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on validation set
            val_metrics = evaluate_model(trained_model, val_loader, device)
            train_metrics = evaluate_model(trained_model, train_loader, device)
            
            # Calculate composite score (you can adjust weights as needed)
            composite_score = (val_metrics['iou'] + val_metrics['dice'] + val_metrics['f1']) / 3
            
            # Store results
            result = {
                'combination_id': i + 1,
                'parameters': current_params,
                'training_time': training_time,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'composite_score': composite_score,
                'final_loss': loss_history[-1] if loss_history else None
            }
            
            all_results.append(result)
            
            # Save model if it's the best so far
            if composite_score > best_score:
                best_score = composite_score
                best_params = current_params.copy()
                
                # Save best model
                best_model_path = os.path.join(results_dir, "models", f"best_model_combo_{i+1}.pth")
                torch.save(trained_model.state_dict(), best_model_path)
                
                print(f"🎉 New best model! Score: {composite_score:.4f}")
            
            # Print current results
            print(f"Training time: {training_time:.1f}s")
            print(f"Validation Metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
            print(f"Composite Score: {composite_score:.4f}")
            
            # Save intermediate results
            results_file = os.path.join(results_dir, "grid_search_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'param_grid': param_grid,
                    'completed_combinations': i + 1,
                    'total_combinations': len(param_combinations),
                    'best_score': best_score,
                    'best_params': best_params,
                    'all_results': all_results
                }, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error with combination {i+1}: {str(e)}")
            # Add failed result
            all_results.append({
                'combination_id': i + 1,
                'parameters': current_params,
                'error': str(e),
                'status': 'failed'
            })
            continue
    
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE!")
    print(f"Best parameters: {best_params}")
    print(f"Best composite score: {best_score:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    # Save final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'param_grid': param_grid,
        'total_combinations': len(param_combinations),
        'best_score': best_score,
        'best_params': best_params,
        'best_model_path': best_model_path,
        'all_results': all_results
    }
    
    with open(os.path.join(results_dir, "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Create results summary
    create_results_summary(all_results, results_dir)
    
    return final_results, best_model_path


def create_results_summary(all_results, results_dir):
    """Create a summary table of results"""
    successful_results = [r for r in all_results if 'error' not in r]
    
    if not successful_results:
        print("No successful results to summarize")
        return
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'ID':<4} {'Epochs':<7} {'Batch':<6} {'LR':<8} {'Val IoU':<8} {'Val Dice':<9} {'Val F1':<7} {'Score':<7} {'Time':<6}")
    print("-" * 80)
    
    # Sort by composite score
    successful_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    summary_data = []
    for result in successful_results:
        params = result['parameters']
        metrics = result['val_metrics']
        summary_data.append({
            'id': result['combination_id'],
            'epochs': params['epochs'],
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'val_iou': metrics['iou'],
            'val_dice': metrics['dice'],
            'val_f1': metrics['f1'],
            'composite_score': result['composite_score'],
            'training_time': result['training_time']
        })
        
        print(f"{result['combination_id']:<4} "
              f"{params['epochs']:<7} "
              f"{params['batch_size']:<6} "
              f"{params['learning_rate']:<8.1e} "
              f"{metrics['iou']:<8.4f} "
              f"{metrics['dice']:<9.4f} "
              f"{metrics['f1']:<7.4f} "
              f"{result['composite_score']:<7.4f} "
              f"{result['training_time']:<6.0f}")
    
    # Save summary as CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)
    print(f"\nResults summary saved to: {os.path.join(results_dir, 'results_summary.csv')}")


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


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find data paths (your existing code)
    base_path = os.getcwd()
    print(f"Current working directory: {base_path}")

    data_paths = find_folders(base_path, "Data")

    if not data_paths:
        natural_paths = find_folders(base_path, "Directory 1")
        mask_paths = find_folders(base_path, "Directory 2")
        
        if natural_paths and mask_paths:
            natural_path = natural_paths[0]
            mask_path = mask_paths[0]
        else:
            print("Could not find data folders automatically.")
            natural_path = os.path.join(base_path, "UNET_ENV", "Data", "Directory 1")
            mask_path = os.path.join(base_path, "UNET_ENV", "Data", "Directory 2")
    else:
        plant_data_path = data_paths[0]
        natural_path = os.path.join(plant_data_path, "Directory 1")
        mask_path = os.path.join(plant_data_path, "Directory 2")

    print(f"Using image path: {natural_path}")
    print(f"Using mask path: {mask_path}")

    if not os.path.exists(natural_path): 
        raise FileNotFoundError(f"Error: Image directory not found at {natural_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Error: Mask directory not found at {mask_path}")

    # Create dataset
    dataset = PlantDataset(natural_path, mask_path, limit=1308)
    print(f"Dataset size: {len(dataset)}")
    
    # Define hyperparameter grid
    param_grid = {
        'epochs': [10, 15, 25, 30],  # Different epoch counts
        'batch_size': [2, 4, 8],     # Different batch sizes
        'learning_rate': [1e-4, 5e-4, 1e-3]  # Different learning rates
    }
    
    print(f"Total combinations to test: {len(list(itertools.product(*param_grid.values())))}")
    
    # Run grid search
    results, best_model_path = grid_search_hyperparameters(dataset, device, param_grid)
    
    # Optional: Load and test the best model on a separate test set
    print("\n" + "="*60)
    print("TESTING BEST MODEL ON FINAL TEST SET")
    print("="*60)
    
    # Create final train/test split (90/10 for final evaluation)
    generator = torch.Generator().manual_seed(42)  # Different seed for final split
    final_train_size = int(0.9 * len(dataset))
    final_test_size = len(dataset) - final_train_size
    _, final_test_dataset = random_split(dataset, [final_train_size, final_test_size], generator=generator)
    
    final_test_loader = DataLoader(final_test_dataset, batch_size=results['best_params']['batch_size'], shuffle=False)
    
    # Load best model
    best_model = Unet4().to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on final test set
    final_test_metrics = evaluate_model(best_model, final_test_loader, device)
    
    print("Final test set results with best hyperparameters:")
    print(f"Best parameters: {results['best_params']}")
    for metric, value in final_test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize predictions with best model
    print("Visualizing predictions with best model...")
    visualize_predictions(best_model, dataset, device, num_samples=3, show_overlay=True, show_iou=True)


if __name__ == "__main__":
    main()