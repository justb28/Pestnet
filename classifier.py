from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
from models import LeNet, AlexNet
from torch.utils.data import Subset
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import trange, tqdm

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Training Configuration
batch_size = 64
num_epochs = 20
learning_rate = 1e-4

# Find dataset folders
def find_folders(base_path, target_folder):
    results = []
    for root, dirs, files in os.walk(base_path):
        if target_folder in dirs:
            results.append(os.path.join(root, target_folder))
    return results

base_path = os.getcwd()
data_paths = find_folders(base_path, "Dataset2.1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
full_dataset = datasets.ImageFolder(root="Dataset2.1", transform=transform)
targets = [label for _, label in full_dataset.samples]

# Split dataset
train_idx, temp_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=np.array(targets)[temp_idx],
    random_state=42
)

train_set = Subset(full_dataset, train_idx)
val_set = Subset(full_dataset, val_idx)
test_set = Subset(full_dataset, test_idx)

print("Class names:", full_dataset.classes)

def class_distribution(dataset, indices):
    labels = [dataset.samples[i][1] for i in indices]
    return Counter(labels)

print("Train:", class_distribution(full_dataset, train_idx))
print("Val:", class_distribution(full_dataset, val_idx))
print("Test:", class_distribution(full_dataset, test_idx))

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

print(f'Number of training examples: {len(train_set)}')
print(f'Number of validation examples: {len(val_set)}')
print(f'Number of testing examples: {len(test_set)}')

# Sample test images
dataiter = iter(test_loader)
test_images, test_labels = next(dataiter)

# Visualize batch
plt.figure(figsize=(20, 10))
out = torchvision.utils.make_grid(test_images, 8, normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))
plt.title("Sample Test Images")
plt.show()

# Create model
model = AlexNet(num_classes=4).to(device)
print(model)

# Test forward pass
out = model(test_images.to(device))
print(f"Output shape: {out.shape}")

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_fun = nn.CrossEntropyLoss()

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

# Training loop
training_loss_logger = []
validation_acc_logger = []
training_acc_logger = []

for epoch in trange(num_epochs, leave=False, desc="Epoch"):
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

print("Training Complete")

# Plot training loss
plt.figure(figsize=(10, 10))
train_x = np.linspace(0, num_epochs, len(training_loss_logger))
plt.plot(train_x, training_loss_logger)
plt.title("AlexNet Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot accuracies
plt.figure(figsize=(10, 10))
train_x = np.linspace(0, num_epochs, len(training_acc_logger))
plt.plot(train_x, training_acc_logger, c="y", label="Training accuracy")
valid_x = np.linspace(0, num_epochs, len(validation_acc_logger))
plt.plot(valid_x, validation_acc_logger, c="k", label="Validation accuracy")
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Test evaluation
test_acc = evaluate(model=model, device=device, loader=test_loader)
print("The total test accuracy is: %.2f%%" % (test_acc * 100))

# Generate predictions for confusion matrix
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Visualize predictions
plt.figure(figsize=(20, 10))
out = torchvision.utils.make_grid(test_images[:8], 8, normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))
plt.title("Test Images with Predictions")
plt.show()

# Get predictions for first 8 images
with torch.no_grad():
    outputs = model(test_images[:8].to(device))
    _, preds = torch.max(outputs, 1)

print("Predicted Values\n", list(preds.cpu().numpy()))
print("True Values\n", list(test_labels[:8].numpy()))

# Visualize feature maps from layer1 (first conv layer)
with torch.no_grad():
    model_out = model.layer1[0](test_images.to(device))  # Get conv output before BatchNorm
    
plt.figure(figsize=(20, 10))
out = torchvision.utils.make_grid(model_out[:16, 5].unsqueeze(1).detach().cpu(), 8, normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))
plt.title("Layer 1 Feature Maps (Channel 5)")
plt.show()

# Visualize feature maps from layer2
with torch.no_grad():
    out1 = model.layer1(test_images.to(device))
    model_out2 = model.layer2[0](out1)  # Get conv output before BatchNorm
    
plt.figure(figsize=(20, 10))
out = torchvision.utils.make_grid(model_out2[:16, 10].unsqueeze(1).detach().cpu(), 8, normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))
plt.title("Layer 2 Feature Maps (Channel 10)")
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'M1', 'M2', 'M3'],
            yticklabels=['Healthy', 'M1', 'M2', 'M3'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=['Healthy', 'M1', 'M2', 'M3']))

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model checkpoint
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_acc': training_acc_logger[-1],
    'val_acc': validation_acc_logger[-1],
    'test_acc': test_acc
}, 'models/alexnet_model.pth')
print(f"Model saved to 'models/alexnet_model.pth'")