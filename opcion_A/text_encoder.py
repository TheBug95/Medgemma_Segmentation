"""
SigLIP Text Encoder standalone.

Fase 1 — Tarea 1.4
- Codifica texto diagnóstico de MedGemma a embedding 768-dim
- Usa SiglipModel.get_text_features() (mismo modelo que MedSigLIP vision)
- Se usa para comparar con embeddings de máscaras vía coseno
"""

import logging
import torch
from transformers import AutoProcessor, SiglipModel

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 768


class TextEncoder:
    """Codifica texto a embeddings usando SigLIP text encoder."""

    DEFAULT_MODEL_ID = "google/medsiglip-448"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.embedding_dim = None

    def load(self):
        """
        Carga SiglipModel completo (text + vision encoders).

        SiglipModel.get_text_features() extrae solo el embedding del texto.
        El modelo se carga en bf16 para eficiencia de VRAM.

        Returns:
            tuple: (model, processor)
        """
        self.model = SiglipModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Verificar dimensión
        self.embedding_dim = self.model.config.text_config.hidden_size
        if self.embedding_dim != EXPECTED_EMBEDDING_DIM:
            logger.warning(
                "SigLIP text embedding dim = %d, esperado = %d.",
                self.embedding_dim, EXPECTED_EMBEDDING_DIM,
            )

        return self.model, self.processor

    def encode(self, text: str) -> torch.Tensor:
        """
        Codifica texto a embedding 768-dim.

        Args:
            text: texto diagnóstico de MedGemma

        Returns:
            torch.Tensor (1, embedding_dim) — embedding del texto
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)

        return text_emb.float()

    def encode_batch(self, texts: list) -> torch.Tensor:
        """
        Codifica múltiples textos.

        Args:
            texts: List[str]

        Returns:
            torch.Tensor (N, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embs = self.model.get_text_features(**inputs)

        return text_embs.float()

    def unload(self):
        """Libera memoria del modelo."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.embedding_dim = None
        torch.cuda.empty_cache()
