"""
Visualización final — Overlay, barras de confianza, scores.

Fase 5 — Tarea 5.4
- Overlay máscara + imagen
- Barra de confianza
- Score components
- Incertidumbre y top features
"""

import numpy as np


def create_overlay(image: np.ndarray, mask: np.ndarray, decision: dict,
                   xai_info: dict = None, output_path: str = None) -> np.ndarray:
    """
    Genera visualización final con overlay máscara + imagen + metadatos.

    Args:
        image: np.ndarray (H, W, 3)
        mask: np.ndarray (H, W) — máscara binaria seleccionada
        decision: dict con scores, uncertainty, etc.
        xai_info: dict opcional con info XAI
        output_path: str opcional para guardar imagen

    Returns:
        np.ndarray — imagen con overlay
    """
    raise NotImplementedError("Tarea 5.4 pendiente de implementación")


def create_confidence_bar(score: float, width: int = 20) -> str:
    """
    Genera barra de confianza en texto.

    Args:
        score: float [0, 1]
        width: ancho de la barra

    Returns:
        str tipo "████████░░ 80%"
    """
    filled = int(score * width)
    empty = width - filled
    return '█' * filled + '░' * empty + f' {score:.0%}'
