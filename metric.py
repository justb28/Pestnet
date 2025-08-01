import torch

def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute precision, recall, F1-score, and overall error.
    Args:
        preds: predicted masks, shape [B, 1, H, W]
        targets: ground truth masks, shape [B, 1, H, W]
    Returns:
        precision, recall, f1, error_rate (all floats)
    """
    preds_bin = (preds > threshold).float()
    targets = targets.float()

    tp = (preds_bin * targets).sum(dim=(1, 2, 3))
    fp = (preds_bin * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds_bin) * targets).sum(dim=(1, 2, 3))
    total = torch.numel(targets[0])

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    error = (fp + fn) / total  # per sample

    return (
        precision.mean().item(),
        recall.mean().item(),
        f1.mean().item(),
        error.mean().item()
    )
