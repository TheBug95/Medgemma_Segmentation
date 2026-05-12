"""
KDE Gaussiano por dimensión — Paper IGPL.

Fase 2 — Tarea 2.1
- KDE por cada dimensión del embedding (768 dimensiones)
- Bandwidth: h = k^(-1/5) (Silverman)
- Fórmula: p̂ⱼ(u) = (1/kh) × Σᵢ exp[-½ × ((u - z_ij) / h)²]
- Log-densidad: ℓ* = Σⱼ log p̂ⱼ(z*ⱼ)
"""

import torch
import numpy as np


class PerDimensionKDE:
    """
    KDE Gaussiano por dimensión exactamente como en el paper IGPL.

    Para cada dimensión j del embedding (768 dimensiones):
        p̂ⱼ(u) = (1 / kh) × Σᵢ₌₁ᵏ exp[-½ × ((u - z_ij) / h)²]

    donde:
        k = tamaño del support set
        h = k^(-1/5) → bandwidth óptimo de Silverman para Gaussian KDE
    """

    def __init__(self, support_embeddings: torch.Tensor):
        """
        Args:
            support_embeddings: (k, 768) — embeddings de las máscaras del support set
        """
        self.k = support_embeddings.shape[0]
        self.dim = support_embeddings.shape[1]  # 768
        self.h = self.k ** (-1.0 / 5.0)  # bandwidth
        self.support = support_embeddings  # (k, 768)

    def log_density(self, query_embedding: torch.Tensor) -> tuple:
        """
        Calcula ℓ* = Σⱼ log p̂ⱼ(z*ⱼ).

        Args:
            query_embedding: (1, 768) o (768,)

        Returns:
            total_log_density: ℓ* (scalar)
            per_dim: (768,) — contribución individual por dimensión
        """
        query = query_embedding.reshape(1, self.dim)  # (1, 768)
        support = self.support  # (k, 768)

        # (1, 768) - (k, 768) → (k, 768) diferencias
        diffs = query - support  # (k, 768)

        # log p̂ⱼ(z*ⱼ) para cada dimensión j
        scaled_diffs = diffs / self.h  # (k, 768)
        exponents = -0.5 * scaled_diffs ** 2  # (k, 768)
        log_per_dim = torch.logsumexp(exponents, dim=0)  # (768,)
        log_per_dim = log_per_dim - np.log(self.k * self.h)  # (768,)

        # ℓ* = Σⱼ log p̂ⱼ(z*ⱼ)
        total_log_density = log_per_dim.sum()

        return total_log_density, log_per_dim
