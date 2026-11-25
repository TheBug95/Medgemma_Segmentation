"""
Trainer module for SAM LoRA fine-tuning.
"""
import logging
import os
import time
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from .config import Config
from .model import SAMLoRA
from .utils import (
    CombinedLoss, EarlyStopping, AverageMeter,
    calculate_iou, calculate_dice, CheckpointManager
)


logger = logging.getLogger(__name__)


class SAMTrainer:
    """
    Trainer class for SAM LoRA fine-tuning.
    """

    def __init__(
        self,
        model: SAMLoRA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: str = "cuda",
        use_wandb: bool = False
    ):
        """
        Args:
            model: SAM LoRA model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to train on
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Loss function
        self.criterion = CombinedLoss(
            focal_weight=config.training.focal_loss_weight,
            dice_weight=config.training.dice_loss_weight,
            iou_weight=config.training.iou_loss_weight
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_amp = config.training.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='min'
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=os.path.join(config.experiment_dir, 'checkpoints'),
            keep_last_n=config.training.keep_last_n,
            mode='min'
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': []
        }

        # Initialize wandb if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=vars(config),
                    name=config.experiment_name
                )
            except ImportError:
                logger.info("Warning: wandb not installed. Logging disabled.")
                self.use_wandb = False

    def _create_optimizer(self):
        """Create optimizer"""
        trainable_params = self.model.get_trainable_parameters()

        if self.config.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        elif self.config.training.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler.lower() == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.training.num_epochs
            )
        elif self.config.training.scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            masks_list = batch['masks']  # List of tensors
            boxes_list = batch['boxes']
            point_coords_list = batch['point_coords']
            point_labels_list = batch['point_labels']

            batch_loss = 0.0
            batch_iou = 0.0
            batch_dice = 0.0
            num_samples = len(images)

            self.optimizer.zero_grad()

            # Process each image in batch
            for i in range(num_samples):
                image = images[i:i+1]  # [1, 3, H, W]
                true_masks = masks_list[i].to(self.device)  # [N, H, W]

                # Get prompts
                boxes = boxes_list[i].to(self.device) if boxes_list[i] is not None else None
                point_coords = point_coords_list[i].to(self.device) if point_coords_list[i] is not None else None
                point_labels = point_labels_list[i].to(self.device) if point_labels_list[i] is not None else None

                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        # Process each mask/object separately
                        sample_loss = 0.0
                        sample_iou = 0.0
                        sample_dice = 0.0

                        for obj_idx in range(len(true_masks)):
                            # Get prompts for this object
                            obj_boxes = boxes[obj_idx:obj_idx+1] if boxes is not None else None
                            obj_points = point_coords[obj_idx:obj_idx+1] if point_coords is not None else None
                            obj_labels = point_labels[obj_idx:obj_idx+1] if point_labels is not None else None

                            # Predict mask
                            pred_masks, iou_pred = self.model(
                                image,
                                point_coords=obj_points,
                                point_labels=obj_labels,
                                boxes=obj_boxes,
                                multimask_output=False
                            )

                            # Get true mask
                            true_mask = true_masks[obj_idx:obj_idx+1].unsqueeze(0)  # [1, 1, H, W]

                            # Calculate loss
                            loss, _ = self.criterion(pred_masks, true_mask)
                            sample_loss += loss

                            # Calculate metrics
                            with torch.no_grad():
                                sample_iou += calculate_iou(pred_masks, true_mask)
                                sample_dice += calculate_dice(pred_masks, true_mask)

                        # Average over objects
                        num_objects = len(true_masks)
                        sample_loss = sample_loss / num_objects
                        sample_iou = sample_iou / num_objects
                        sample_dice = sample_dice / num_objects

                    # Backward pass
                    self.scaler.scale(sample_loss).backward()
                else:
                    # Same as above but without autocast
                    sample_loss = 0.0
                    sample_iou = 0.0
                    sample_dice = 0.0

                    for obj_idx in range(len(true_masks)):
                        obj_boxes = boxes[obj_idx:obj_idx+1] if boxes is not None else None
                        obj_points = point_coords[obj_idx:obj_idx+1] if point_coords is not None else None
                        obj_labels = point_labels[obj_idx:obj_idx+1] if point_labels is not None else None

                        pred_masks, iou_pred = self.model(
                            image,
                            point_coords=obj_points,
                            point_labels=obj_labels,
                            boxes=obj_boxes,
                            multimask_output=False
                        )

                        true_mask = true_masks[obj_idx:obj_idx+1].unsqueeze(0)
                        loss, _ = self.criterion(pred_masks, true_mask)
                        sample_loss += loss

                        with torch.no_grad():
                            sample_iou += calculate_iou(pred_masks, true_mask)
                            sample_dice += calculate_dice(pred_masks, true_mask)

                    num_objects = len(true_masks)
                    sample_loss = sample_loss / num_objects
                    sample_iou = sample_iou / num_objects
                    sample_dice = sample_dice / num_objects

                    sample_loss.backward()

                batch_loss += sample_loss.item()
                batch_iou += sample_iou
                batch_dice += sample_dice

            # Average over batch
            batch_loss /= num_samples
            batch_iou /= num_samples
            batch_dice /= num_samples

            # Gradient clipping and optimizer step
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                    self.optimizer.step()

            # Update meters
            loss_meter.update(batch_loss, num_samples)
            iou_meter.update(batch_iou, num_samples)
            dice_meter.update(batch_dice, num_samples)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'iou': f'{iou_meter.avg:.4f}',
                'dice': f'{dice_meter.avg:.4f}'
            })

        metrics = {
            'train_loss': loss_meter.avg,
            'train_iou': iou_meter.avg,
            'train_dice': dice_meter.avg
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        with torch.no_grad():
            for batch in pbar:
                images = batch['images'].to(self.device)
                masks_list = batch['masks']
                boxes_list = batch['boxes']
                point_coords_list = batch['point_coords']
                point_labels_list = batch['point_labels']

                batch_loss = 0.0
                batch_iou = 0.0
                batch_dice = 0.0
                num_samples = len(images)

                # Process each image in batch
                for i in range(num_samples):
                    image = images[i:i+1]
                    true_masks = masks_list[i].to(self.device)

                    boxes = boxes_list[i].to(self.device) if boxes_list[i] is not None else None
                    point_coords = point_coords_list[i].to(self.device) if point_coords_list[i] is not None else None
                    point_labels = point_labels_list[i].to(self.device) if point_labels_list[i] is not None else None

                    sample_loss = 0.0
                    sample_iou = 0.0
                    sample_dice = 0.0

                    for obj_idx in range(len(true_masks)):
                        obj_boxes = boxes[obj_idx:obj_idx+1] if boxes is not None else None
                        obj_points = point_coords[obj_idx:obj_idx+1] if point_coords is not None else None
                        obj_labels = point_labels[obj_idx:obj_idx+1] if point_labels is not None else None

                        pred_masks, iou_pred = self.model(
                            image,
                            point_coords=obj_points,
                            point_labels=obj_labels,
                            boxes=obj_boxes,
                            multimask_output=False
                        )

                        true_mask = true_masks[obj_idx:obj_idx+1].unsqueeze(0)
                        loss, _ = self.criterion(pred_masks, true_mask)
                        sample_loss += loss.item()

                        sample_iou += calculate_iou(pred_masks, true_mask)
                        sample_dice += calculate_dice(pred_masks, true_mask)

                    num_objects = len(true_masks)
                    sample_loss /= num_objects
                    sample_iou /= num_objects
                    sample_dice /= num_objects

                    batch_loss += sample_loss
                    batch_iou += sample_iou
                    batch_dice += sample_dice

                batch_loss /= num_samples
                batch_iou /= num_samples
                batch_dice /= num_samples

                loss_meter.update(batch_loss, num_samples)
                iou_meter.update(batch_iou, num_samples)
                dice_meter.update(batch_dice, num_samples)

                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'iou': f'{iou_meter.avg:.4f}',
                    'dice': f'{dice_meter.avg:.4f}'
                })

        metrics = {
            'val_loss': loss_meter.avg,
            'val_iou': iou_meter.avg,
            'val_dice': dice_meter.avg
        }

        return metrics

    def train(self) -> dict[str, list]:
        """Main training loop"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        logger.info(f"{'='*50}\n")

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']

            # Update history
            for key, value in epoch_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)

            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log(epoch_metrics)

            # Print epoch summary
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Train IoU: {train_metrics['train_iou']:.4f} | "
                  f"Val IoU: {val_metrics['val_iou']:.4f}")
            logger.info(f"Train Dice: {train_metrics['train_dice']:.4f} | "
                  f"Val Dice: {val_metrics['val_dice']:.4f}")
            logger.info(f"LR: {epoch_metrics['lr']:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.checkpoint_manager.save(
                    self.model,
                    epoch=epoch + 1,
                    score=val_metrics['val_loss'],
                    optimizer=self.optimizer,
                    metrics=epoch_metrics
                )

            # Check for best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                best_path = os.path.join(
                    self.config.experiment_dir,
                    'checkpoints',
                    'best_model.pt'
                )
                self.model.save_checkpoint(
                    best_path,
                    epoch=epoch + 1,
                    optimizer_state=self.optimizer.state_dict(),
                    metrics=epoch_metrics
                )
                logger.info(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_metrics['val_loss']):
                logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            logger.info("")

        logger.info(f"\n{'='*50}")
        logger.info(f"Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"{'='*50}\n")

        return self.training_history
