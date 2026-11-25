"""
SAM model with LoRA adaptation.
"""
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict
from segment_anything import sam_model_registry, SamPredictor
from peft import LoraConfig, get_peft_model
import warnings

from .config import Config


logger = logging.getLogger(__name__)


class SAMLoRA(nn.Module):
    """
    SAM model with LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[Path] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
        freeze_image_encoder: bool = False,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        device: str | torch.device = "cuda"
    ) -> None:
        """
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Path to pretrained SAM checkpoint
            lora_rank: Rank of LoRA matrices
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout rate for LoRA layers
            lora_target_modules: List of module names to apply LoRA to
            freeze_image_encoder: Whether to freeze the image encoder
            freeze_prompt_encoder: Whether to freeze the prompt encoder
            freeze_mask_decoder: Whether to freeze the mask decoder
            device: Device to load model on
        """
        super().__init__()

        self.model_type = model_type
        self.device = device

        # Default LoRA target modules for SAM's ViT
        if lora_target_modules is None:
            lora_target_modules = ["qkv", "proj"]

        # Load base SAM model
        if checkpoint_path is None:
            logger.info("Warning: No checkpoint path provided. Loading model without pretrained weights.")
            logger.info("Please download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            self.sam = sam_model_registry[model_type]()
        else:
            logger.info(f"Loading SAM model from {checkpoint_path}")
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        self.sam.to(device)

        # Freeze components as specified
        if freeze_image_encoder:
            logger.info("Freezing image encoder")
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            logger.info("Freezing prompt encoder")
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            logger.info("Freezing mask decoder")
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = False

        # Apply LoRA to image encoder
        if not freeze_image_encoder:
            logger.info(f"Applying LoRA to image encoder with rank={lora_rank}, alpha={lora_alpha}")
            self.sam.image_encoder = self._apply_lora_to_module(
                self.sam.image_encoder,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )

        # Count trainable parameters
        self._print_trainable_parameters()

    def _apply_lora_to_module(
        self,
        module: nn.Module,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list
    ) -> nn.Module:
        """
        Apply LoRA to specified modules within a parent module.
        """
        try:
            # Configure LoRA
            # Note: No task_type needed for custom models like SAM
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                # task_type is not needed for SAM - it's for HuggingFace models
            )

            # Apply LoRA using PEFT
            module = get_peft_model(module, lora_config)
            return module
        except Exception as e:
            warnings.warn(f"Failed to apply LoRA with PEFT: {e}. Continuing without LoRA.")
            return module

    def _print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        for name, param in self.sam.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable%: {100 * trainable_params / all_params:.2f}%")

    def forward(
        self,
        images: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_inputs: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAM.

        Args:
            images: [B, 3, H, W] input images
            point_coords: [B, N, 2] point prompts (x, y)
            point_labels: [B, N] labels for points (1=foreground, 0=background)
            boxes: [B, 4] box prompts in xyxy format
            mask_inputs: [B, 1, 256, 256] mask inputs (optional)
            multimask_output: Whether to output multiple masks

        Returns:
            masks: [B, N_masks, H, W] predicted masks
            iou_predictions: [B, N_masks] predicted IoU scores
        """
        # Encode image
        image_embeddings = self.sam.image_encoder(images)

        # Process prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels) if point_coords is not None else None,
            boxes=boxes,
            masks=mask_inputs
        )

        # Decode masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )

        # Upscale masks to original size
        masks = nn.functional.interpolate(
            low_res_masks,
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False
        )

        return masks, iou_predictions

    def predict(
        self,
        image: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode prediction.

        Returns:
            Dictionary with 'masks' and 'iou_predictions'
        """
        self.eval()
        with torch.no_grad():
            masks, iou_predictions = self.forward(
                image.unsqueeze(0) if image.dim() == 3 else image,
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=boxes,
                multimask_output=multimask_output
            )

        return {
            'masks': masks,
            'iou_predictions': iou_predictions
        }

    def save_checkpoint(
        self, path: Path, epoch: int, optimizer_state: dict | None = None, metrics: dict | None = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_type': self.model_type,
            'model_state_dict': self.sam.state_dict(),
            'metrics': metrics or {}
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path, load_optimizer: bool = False) -> Tuple[int, Optional[dict]]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
            load_optimizer: Whether to return optimizer state

        Returns:
            Tuple of (epoch, optimizer_state_dict or None)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.sam.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        optimizer_state = checkpoint.get('optimizer_state_dict') if load_optimizer else None

        logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch, optimizer_state

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer"""
        return [p for p in self.sam.parameters() if p.requires_grad]


def create_sam_lora_model(config: Config, device: str | torch.device = "cuda") -> SAMLoRA:
    """
    Factory function to create SAM LoRA model from config.

    Args:
        config: Configuration object
        device: Device to load model on

    Returns:
        SAMLoRA model
    """
    model = SAMLoRA(
        model_type=config.model.model_type,
        checkpoint_path=config.model.checkpoint_path,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        lora_target_modules=config.model.lora_target_modules,
        freeze_image_encoder=config.model.freeze_image_encoder,
        freeze_prompt_encoder=config.model.freeze_prompt_encoder,
        freeze_mask_decoder=config.model.freeze_mask_decoder,
        device=device
    )

    return model
