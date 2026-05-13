"""
MedSigLIP Standalone como extractor de features de máscaras.

Fase 1 — Tarea 1.3
- Usa google/medsiglip-448 como modelo independiente (~1.5 GB VRAM)
- NO extrae el vision encoder de MedGemma (estrategia dual)
- Input: imagen original + máscara binaria (H×W)
- Output: embedding 768-dim por máscara
- Preprocesamiento: 448×448, misma normalización que MedSigLIP
"""

import logging
import torch
import numpy as np
from transformers import AutoProcessor, SiglipVisionModel

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 768


class MaskFeatureExtractor:
    """Extrae embeddings de máscaras usando MedSigLIP standalone."""

    DEFAULT_MODEL_ID = "google/medsiglip-448"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.embedding_dim = None

    def load(self):
        """
        Carga MedSigLIP standalone desde HuggingFace.

        Verifica que la dimensión de embedding sea la esperada (768).
        Si no lo es, registra advertencia y ajusta embedding_dim al valor real.

        Returns:
            tuple: (model, processor)
        """
        self.model = SiglipVisionModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Verificar dimensión de embedding
        self.embedding_dim = self.model.config.hidden_size
        if self.embedding_dim != EXPECTED_EMBEDDING_DIM:
            logger.warning(
                "MedSigLIP embedding dim = %d, esperado = %d. "
                "Se usará la dimensión real. Ajustar KDE y text_encoder.",
                self.embedding_dim, EXPECTED_EMBEDDING_DIM,
            )

        return self.model, self.processor

    def extract(self, image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """
        Extrae embedding de una máscara aplicada a una imagen.

        Args:
            image: np.ndarray (H, W, 3) RGB — imagen original
            mask: np.ndarray (H, W) — máscara binaria (bool o 0/1)

        Returns:
            torch.Tensor (embedding_dim,) — embedding de la máscara
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        # Aplicar máscara: fondo a negro
        if mask.ndim == 2:
            mask_3d = mask[..., None]
        else:
            mask_3d = mask
        masked_image = image * mask_3d

        # Preprocesar con el processor de SigLIP
        inputs = self.processor(images=masked_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # pooler_output: (1, dim) → squeeze a (dim,)
        emb = outputs.pooler_output.squeeze(0).float()

        # Verificar dimensión en runtime
        if emb.shape[-1] != self.embedding_dim:
            logger.warning(
                "pooler_output dim=%d != config.hidden_size=%d. "
                "Usando last_hidden_state[:, 0, :].",
                emb.shape[-1], self.embedding_dim,
            )
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).float()

        return emb

    def extract_batch(self, image: np.ndarray, masks: list) -> torch.Tensor:
        """
        Extrae embeddings de múltiples máscaras.

        Args:
            image: np.ndarray (H, W, 3) RGB
            masks: List[np.ndarray o dict] — máscaras binarias

        Returns:
            torch.Tensor (N, embedding_dim) — embeddings de todas las máscaras
        """
        embeddings = []
        for mask in masks:
            if isinstance(mask, dict):
                mask = mask["segmentation"]
            emb = self.extract(image, mask)
            embeddings.append(emb)
        return torch.stack(embeddings)

    def unload(self):
        """Libera memoria del modelo."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.embedding_dim = None
        torch.cuda.empty_cache()
