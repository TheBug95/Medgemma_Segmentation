"""Inferencia de máscaras binarias con SAM y cálculo de IoU y Dice."""
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SAMLoRA
from .utils import AverageMeter, calculate_dice, calculate_iou, visualize_predictions


class SAMInference:
    """
    Inference class for trained SAM LoRA model.
    """

    def __init__(
        self,
        model: SAMLoRA,
        device: str | torch.device = "cuda",
        threshold: float = 0.5
    ) -> None:
        """
        Args:
            model: Trained SAM LoRA model
            device: Device to run inference on
            threshold: Threshold for binary mask prediction
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()

    def predict_single_image(
        self,
        image: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict mask for a single image.

        Args:
            image: [1, 3, H, W] or [3, H, W] input image
            point_coords: Point prompts
            point_labels: Point labels
            boxes: Box prompts
            multimask_output: Whether to output multiple masks

        Returns:
            Dictionary with 'masks' and 'iou_predictions'
        """
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)

            image = image.to(self.device)

            if point_coords is not None:
                point_coords = point_coords.to(self.device)
            if point_labels is not None:
                point_labels = point_labels.to(self.device)
            if boxes is not None:
                boxes = boxes.to(self.device)

            masks, iou_predictions = self.model(
                image,
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=boxes,
                multimask_output=multimask_output
            )

            # Apply threshold
            masks = (torch.sigmoid(masks) > self.threshold).float()

        return {
            'masks': masks.cpu(),
            'iou_predictions': iou_predictions.cpu()
        }

    def predict_batch(
        self,
        dataloader: DataLoader,
        save_dir: Optional[Path] = None,
        masks_dir: Optional[Path] = None
    ) -> tuple[dict[str, float], list[dict]]:
        """
        Run inference on a batch of images.

        Args:
            dataloader: DataLoader with test images
            save_dir: Directory to save visualizations (optional)
            masks_dir: Directory to save predicted SAM masks (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if masks_dir is not None:
            masks_dir = Path(masks_dir)
            masks_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            images = batch['images'].to(self.device)
            masks_list = batch['masks']
            boxes_list = batch['boxes']
            point_coords_list = batch['point_coords']
            point_labels_list = batch['point_labels']
            image_ids = batch['image_ids']
            image_names = batch.get('image_names', [None] * len(images))

            # Process each image
            for i in range(len(images)):
                image = images[i:i+1]
                true_masks = masks_list[i]
                image_name = image_names[i]

                boxes = boxes_list[i] if boxes_list[i] is not None else None
                point_coords = point_coords_list[i] if point_coords_list[i] is not None else None
                point_labels = point_labels_list[i] if point_labels_list[i] is not None else None

                image_results = {
                    'image_id': image_ids[i].item(),
                    'objects': []
                }

                # Process each object/mask
                for obj_idx in range(len(true_masks)):
                    true_mask = true_masks[obj_idx:obj_idx+1]

                    obj_boxes = boxes[obj_idx:obj_idx+1] if boxes is not None else None
                    obj_points = point_coords[obj_idx:obj_idx+1] if point_coords is not None else None
                    obj_labels = point_labels[obj_idx:obj_idx+1] if point_labels is not None else None

                    # Predict
                    pred = self.predict_single_image(
                        image,
                        point_coords=obj_points,
                        point_labels=obj_labels,
                        boxes=obj_boxes,
                        multimask_output=False
                    )

                    pred_mask = pred['masks'][0, 0]  # [H, W]
                    true_mask_2d = true_mask[0]  # [H, W]

                    # Calculate metrics
                    iou = calculate_iou(pred_mask, true_mask_2d, threshold=self.threshold)
                    dice = calculate_dice(pred_mask, true_mask_2d, threshold=self.threshold)

                    iou_meter.update(iou)
                    dice_meter.update(dice)

                    image_results['objects'].append({
                        'object_id': obj_idx,
                        'iou': iou,
                        'dice': dice
                    })

                    # Save predicted mask using the original image file name
                    if masks_dir is not None and image_name is not None:
                        base_name = Path(image_name).stem
                        mask_filename = f"{base_name}_obj_{obj_idx}.png"
                        mask_path = masks_dir / mask_filename

                        mask_np = (pred_mask.numpy() * 255).astype(np.uint8)
                        cv2.imwrite(str(mask_path), mask_np)

                    # Save visualization for first object of first few images
                    if save_dir is not None and batch_idx < 10 and obj_idx == 0:
                        # Convert image to numpy
                        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                        # Denormalize
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = (img_np * std + mean) * 255
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                        pred_mask_np = pred_mask.numpy()
                        true_mask_np = true_mask_2d.numpy()

                        save_path = save_dir / f"image_{image_ids[i].item()}_obj_{obj_idx}.png"
                        visualize_predictions(
                            img_np,
                            pred_mask_np,
                            true_mask_np,
                            save_path=str(save_path)
                        )

                results.append(image_results)

        # Aggregate metrics
        metrics = {
            'mean_iou': iou_meter.avg,
            'mean_dice': dice_meter.avg,
            'num_images': len(results),
            'num_objects': iou_meter.count
        }

        return metrics, results
