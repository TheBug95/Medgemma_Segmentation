"""
Protocolo de validación FD-UQ vs MC Dropout.

Fase 2 — Tarea 2.5
- Compara FD-Uncertainty con MC Dropout
- Métricas: Pearson r > 0.7, ECE < 0.10
- Curva de calibración monótona creciente
"""

import numpy as np


class UncertaintyValidator:
    """
    Compara FD-UQ con MC Dropout y valida calibración.
    """

    def __init__(self, medsiglip_encoder, support_set, val_set):
        """
        Args:
            medsiglip_encoder: encoder para extraer embeddings
            support_set: dataset de soporte para calibrar FD
            val_set: dataset de validación con (image, mask, is_correct)
        """
        self.encoder = medsiglip_encoder
        self.support_set = support_set
        self.val_set = val_set

    def validate(self, fd_estimator, n_mcd_passes: int = 10) -> dict:
        """
        Evalúa sobre val_set con ground truth.

        Returns:
            dict con pearson_r, pearson_p, ece, deciles
        """
        raise NotImplementedError("Tarea 2.5 pendiente de implementación")

    def _mc_dropout_uncertainty(self, image, mask, n: int = 10) -> float:
        """MC Dropout: N forward passes con dropout activo."""
        raise NotImplementedError("Tarea 2.5 pendiente de implementación")

    @staticmethod
    def _compute_ece(deciles_data: list) -> float:
        """Expected Calibration Error."""
        ece = 0.0
        for d in deciles_data:
            ece += abs(d['error_rate'] - d['mean_uncertainty'])
        return ece / len(deciles_data) if deciles_data else 0.0
