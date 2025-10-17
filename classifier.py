import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
from models import LeNet
from torch.utils.data import Subset
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import trange, tqdm

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
num_epochs = 20
learning_rate = 1e-4
k_folds = 5  # Number of folds for cross-validation

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root="Dataset2.1", transform=transform)
targets = [label for _, label in full_dataset.samples]

print("Class names:", full_dataset.classes)
print(f"Total samples: {len(full_dataset)}")

# Training function
def train(model, optimizer, loader, device, loss_fun, loss_logger):
    model.train()
    for i, (x, y) in enumerate(tqdm(loader, leave=False, desc="Training")):
        fx = model(x.to(device))
        loss = loss_fun(fx, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_logger.append(loss.item())
    return model, optimizer, loss_logger

# Evaluation function
def evaluate(model, device, loader):
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, leave=False, desc="Evaluating")):
            fx = model(x.to(device))
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()
    
    return epoch_acc / len(loader.dataset)

# Function to get predictions for confusion matrix
def get_predictions(model, device, loader):
    y_true = []
    y_pred = []
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    return y_true, y_pred

# K-Fold Cross Validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store results for each fold
fold_results = {
    'train_acc': [],
    'val_acc': [],
    'test_acc': [],
    'train_loss': [],
    'val_loss': []
}

print(f"\nStarting {k_folds}-Fold Cross Validation")
print("=" * 60)

for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(len(targets)), targets)):
    print(f"\nFold {fold + 1}/{k_folds}")
    print("-" * 60)
    
    # Further split train_val into train and validation (80-20 split)
    train_val_targets = np.array(targets)[train_val_idx]
    train_idx, val_idx = [], []
    
    # Stratified split for train/val
    for class_idx in range(len(full_dataset.classes)):
        class_indices = train_val_idx[train_val_targets == class_idx]
        n_train = int(0.8 * len(class_indices))
        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train:])
    
    # Create data subsets
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Initialize model for this fold
    model = LeNet(channels_in=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fun = nn.CrossEntropyLoss()
    
    # Training loop for this fold
    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []
    
    for epoch in trange(num_epochs, leave=False, desc=f"Fold {fold+1} Epochs"):
        model, optimizer, training_loss_logger = train(
            model=model, 
            optimizer=optimizer, 
            loader=train_loader, 
            device=device, 
            loss_fun=loss_fun, 
            loss_logger=training_loss_logger
        )
        
        train_acc = evaluate(model=model, device=device, loader=train_loader)
        valid_acc = evaluate(model=model, device=device, loader=val_loader)
        
        validation_acc_logger.append(valid_acc)
        training_acc_logger.append(train_acc)
    
    # Evaluate on test set
    test_acc = evaluate(model=model, device=device, loader=test_loader)
    
    # Store results
    fold_results['train_acc'].append(training_acc_logger[-1])
    fold_results['val_acc'].append(validation_acc_logger[-1])
    fold_results['test_acc'].append(test_acc)
    
    print(f"Fold {fold+1} Results:")
    print(f"  Train Acc: {training_acc_logger[-1]*100:.2f}%")
    print(f"  Val Acc:   {validation_acc_logger[-1]*100:.2f}%")
    print(f"  Test Acc:  {test_acc*100:.2f}%")
    
    # Generate confusion matrix for this fold
    y_true, y_pred = get_predictions(model, device, test_loader)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.classes,
                yticklabels=full_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.tight_layout()
    plt.savefig(f'fold_{fold+1}_confusion_matrix.png')
    plt.show()
    
    # Save model for this fold
    torch.save({
        'fold': fold + 1,
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': training_acc_logger[-1],
        'val_acc': validation_acc_logger[-1],
        'test_acc': test_acc
    }, f'models/lenet_fold_{fold+1}.pth')

print("\n" + "=" * 60)
print("K-Fold Cross Validation Results Summary")
print("=" * 60)
print(f"Train Accuracy:      {np.mean(fold_results['train_acc'])*100:.2f}% ± {np.std(fold_results['train_acc'])*100:.2f}%")
print(f"Validation Accuracy: {np.mean(fold_results['val_acc'])*100:.2f}% ± {np.std(fold_results['val_acc'])*100:.2f}%")
print(f"Test Accuracy:       {np.mean(fold_results['test_acc'])*100:.2f}% ± {np.std(fold_results['test_acc'])*100:.2f}%")

# Plot results across folds
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Accuracy across folds
fold_numbers = list(range(1, k_folds + 1))
axes[0].plot(fold_numbers, np.array(fold_results['train_acc'])*100, 'o-', label='Train Acc')
axes[0].plot(fold_numbers, np.array(fold_results['val_acc'])*100, 's-', label='Val Acc')
axes[0].plot(fold_numbers, np.array(fold_results['test_acc'])*100, '^-', label='Test Acc')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Accuracy Across Folds')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Box plot of test accuracies
axes[1].boxplot([np.array(fold_results['train_acc'])*100, 
                 np.array(fold_results['val_acc'])*100, 
                 np.array(fold_results['test_acc'])*100],
                labels=['Train', 'Validation', 'Test'])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kfold_results_summary.png')
plt.show()

print(f"\nResults saved to 'kfold_results_summary.png'")
print(f"Individual fold models saved to 'models/lenet_fold_X.pth'")