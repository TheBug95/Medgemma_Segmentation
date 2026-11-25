"""
Utility functions and helper classes.
"""
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import json


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda") -> torch.device:
    """Get torch device"""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, *] predicted logits
            targets: [N, *] ground truth (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, *] predicted probabilities (after sigmoid)
            targets: [N, *] ground truth (0 or 1)
        """
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss for segmentation.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, *] predicted logits
            targets: [N, *] ground truth (0 or 1)
        """
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou


class CombinedLoss(nn.Module):
    """
    Combined loss function for SAM training.
    Combines Focal Loss, Dice Loss, and IoU Loss.
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            focal_weight: Weight for focal loss
            dice_weight: Weight for dice loss
            iou_weight: Weight for IoU loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

    def forward(self, pred_masks: torch.Tensor, true_masks: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_masks: [B, N, H, W] predicted masks (logits)
            true_masks: [B, N, H, W] ground truth masks

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        focal = self.focal_loss(pred_masks, true_masks)
        dice = self.dice_loss(pred_masks, true_masks)
        iou = self.iou_loss(pred_masks, true_masks)

        total_loss = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.iou_weight * iou
        )

        loss_dict = {
            'focal_loss': focal.item(),
            'dice_loss': dice.item(),
            'iou_loss': iou.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) metric.

    Args:
        pred_mask: Predicted mask (logits or probabilities)
        true_mask: Ground truth mask (0 or 1)
        threshold: Threshold for binarizing predictions

    Returns:
        IoU score
    """
    if pred_mask.max() > 1.0:
        pred_mask = torch.sigmoid(pred_mask)

    pred_mask = (pred_mask > threshold).float()

    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = (intersection / union).item()
    return iou


def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice coefficient.

    Args:
        pred_mask: Predicted mask (logits or probabilities)
        true_mask: Ground truth mask (0 or 1)
        threshold: Threshold for binarizing predictions

    Returns:
        Dice score
    """
    if pred_mask.max() > 1.0:
        pred_mask = torch.sigmoid(pred_mask)

    pred_mask = (pred_mask > threshold).float()

    # Caso especial: si ambas máscaras son vacías, el modelo acertó (predijo "nada" y era "nada")
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()
    if pred_sum == 0 and true_sum == 0:
        return 1.0

    intersection = (pred_mask * true_mask).sum()
    dice = (2. * intersection) / (pred_sum + true_sum + 1e-8)

    return dice.item()


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' (whether lower or higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Current validation score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_predictions(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize image with predicted and ground truth masks.

    Args:
        image: [H, W, 3] RGB image
        pred_mask: [H, W] predicted mask
        true_mask: [H, W] ground truth mask
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth mask
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Predicted mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Overlay
    axes[3].imshow(image)
    axes[3].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[3].set_title('Prediction Overlay')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_metrics(metrics: dict | list, save_path: Path) -> None:
    """Save metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4, default=str)


def load_metrics(load_path: str) -> dict:
    """Load metrics from JSON file"""
    with open(load_path, 'r') as f:
        return json.load(f)


class CheckpointManager:
    """
    Manages model checkpoints, keeping only the best N checkpoints.
    """

    def __init__(self, save_dir: str, keep_last_n: int = 3, mode: str = 'min'):
        """
        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Number of checkpoints to keep
            mode: 'min' or 'max' for determining best checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.mode = mode
        self.checkpoints = []  # List of (score, path) tuples

    def save(self, model, epoch: int, score: float, optimizer=None, metrics=None):
        """
        Save checkpoint and manage existing checkpoints.

        Args:
            model: Model to save
            epoch: Current epoch
            score: Score to rank checkpoint by
            optimizer: Optimizer state (optional)
            metrics: Additional metrics (optional)
        """
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}_score_{score:.4f}.pt"

        model.save_checkpoint(
            str(checkpoint_path),
            epoch=epoch,
            optimizer_state=optimizer.state_dict() if optimizer else None,
            metrics=metrics
        )

        self.checkpoints.append((score, checkpoint_path))

        # Sort checkpoints by score
        if self.mode == 'min':
            self.checkpoints.sort(key=lambda x: x[0])
        else:
            self.checkpoints.sort(key=lambda x: -x[0])

        # Remove excess checkpoints
        while len(self.checkpoints) > self.keep_last_n:
            _, old_path = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                logger.info(f"Removed old checkpoint: {old_path.name}")

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        if len(self.checkpoints) > 0:
            return self.checkpoints[0][1]
        return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint"""
        if len(self.checkpoints) > 0:
            # Find checkpoint with highest epoch number
            latest = max(self.checkpoints, key=lambda x: int(x[1].stem.split('_')[2]))
            return latest[1]
        return None
