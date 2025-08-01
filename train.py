from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from metric import compute_metrics

def train_model(model, train_loader, device, num_epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_history = []

    total_batches = num_epochs * len(train_loader)
    global_step = 0

    progress_bar = tqdm(total=total_batches, desc='Training', unit='batch')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            # Apply sigmoid to model output for BCELoss
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Epoch': f'{epoch+1}/{num_epochs}',
                'Loss': f'{loss.item():.4f}'
            })

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

    progress_bar.close()
    print(f"\n✅ Training completed. Final Average Loss: {loss_history[-1]:.4f}")
    # Optional: run validation metrics
    # validate_model(model, val_loader, device)

    return model, loss_history


def validate_model(model, val_loader, device):
    model.eval()
    precision_total, recall_total, f1_total, error_total = 0, 0, 0, 0
    count = 0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            precision, recall, f1, error = compute_metrics(outputs, masks)

            precision_total += precision
            recall_total += recall
            f1_total += f1
            error_total += error
            count += 1

    print(f"\n🔍 Evaluation Metrics:")
    print(f"Precision: {precision_total/count:.4f}")
    print(f"Recall: {recall_total/count:.4f}")
    print(f"F1 Score: {f1_total/count:.4f}")
    print(f"❌ Error Rate: {error_total/count:.4f}")
