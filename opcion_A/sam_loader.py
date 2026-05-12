"""
Módulo de carga de SAM 2 Tiny.

Fase 1 — Tarea 1.2
- Carga SAM 2 Tiny (38.9M params) via sam2 API
- Genera máscaras candidatas automáticas
- NOTA: El score de SAM NO es confiable en imágenes médicas (per IGPL)

Instalación SAM 2:
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2 && pip install -e .
"""

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAMLoader:
    """Carga y gestiona SAM 2 para generación de máscaras candidatas."""

    DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
    DEFAULT_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"

    def __init__(self, config: str = DEFAULT_CONFIG,
                 checkpoint: str = DEFAULT_CHECKPOINT,
                 device: str = "cuda"):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.sam = None
        self.mask_generator = None

    def load(self):
        """
        Carga el modelo SAM 2 y crea el generador automático de máscaras.

        Returns:
            tuple: (sam_model, mask_generator)
        """
        self.sam = build_sam2(
            config_file=self.config,
            ckpt_path=self.checkpoint,
            device=self.device,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            crop_n_layers=1,
        )
        return self.sam, self.mask_generator

    def generate_masks(self, image: np.ndarray) -> list:
        """
        Genera máscaras candidatas para una imagen.

        Args:
            image: np.ndarray RGB (H, W, 3), dtype uint8

        Returns:
            List[dict] con keys: 'segmentation' (H,W bool),
                'predicted_iou', 'stability_score', 'bbox', 'area'
        """
        if self.mask_generator is None:
            raise RuntimeError("SAM 2 no cargado. Llama a load() primero.")

        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        masks = self.mask_generator.generate(image)
        return masks

    def unload(self):
        """Libera memoria del modelo."""
        del self.mask_generator
        del self.sam
        self.sam = None
        self.mask_generator = None
        torch.cuda.empty_cache()
