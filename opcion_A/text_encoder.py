"""
SigLIP Text Encoder.

Fase 1 — Tarea 1.4
- Codifica texto diagnóstico de MedGemma a embedding 768-dim
- Se usa para comparar con embeddings de máscaras vía coseno
"""

import torch


class TextEncoder:
    """Codifica texto a embeddings 768-dim usando SigLIP."""

    def __init__(self, text_model, tokenizer):
        self.model = text_model
        self.tokenizer = tokenizer

    def encode(self, text: str) -> torch.Tensor:
        """
        Codifica texto a embedding 768-dim.

        Args:
            text: texto diagnóstico de MedGemma

        Returns:
            torch.Tensor (1, 768)
        """
        raise NotImplementedError("Tarea 1.4 pendiente de implementación")

    def encode_batch(self, texts: list) -> torch.Tensor:
        """
        Codifica múltiples textos.

        Args:
            texts: List[str]

        Returns:
            torch.Tensor (N, 768)
        """
        raise NotImplementedError("Tarea 1.4 pendiente de implementación")
