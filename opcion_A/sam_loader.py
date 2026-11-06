"""
Módulo de carga de SAM 2 Tiny.

Fase 1 — Tarea 1.2
- Carga SAM 2 Tiny (38.9M params) via sam2 API
- Genera máscaras candidatas automáticas
- Auto-descarga checkpoint si no existe localmente
- NOTA: El score de SAM NO es confiable en imágenes médicas (per IGPL)

Instalación SAM 2:
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2 && pip install -e .
"""

import logging
import os
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

logger = logging.getLogger(__name__)

# URL oficial de Meta para SAM 2.1 Tiny
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)


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

    @staticmethod
    def download_checkpoint(dest_path: str = None,
                            url: str = None) -> str:
        """
        Descarga el checkpoint de SAM 2.1 Tiny si no existe.

        Args:
            dest_path: ruta destino (default: DEFAULT_CHECKPOINT)
            url: URL de descarga (default: URL oficial de Meta)

        Returns:
            str — ruta al checkpoint descargado
        """
        dest_path = dest_path or SAMLoader.DEFAULT_CHECKPOINT
        url = url or SAM2_CHECKPOINT_URL

        if os.path.exists(dest_path):
            logger.info("Checkpoint ya existe en %s", dest_path)
            return dest_path

        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        logger.info("Descargando SAM 2.1 Tiny checkpoint a %s ...", dest_path)
        torch.hub.download_url_to_file(url, dest_path)
        logger.info("Checkpoint descargado exitosamente.")
        return dest_path

    def load(self):
        """
        Carga el modelo SAM 2 y crea el generador automático de máscaras.
        Auto-descarga el checkpoint si no existe.

        Returns:
            tuple: (sam_model, mask_generator)
        """
        # Auto-descargar checkpoint si no existe
        if not os.path.exists(self.checkpoint):
            logger.warning("Checkpoint no encontrado en %s. Descargando...",
                           self.checkpoint)
            self.download_checkpoint(self.checkpoint)

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
