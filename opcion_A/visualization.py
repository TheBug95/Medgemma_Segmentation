"""
Visualización final — Overlay, barras de confianza, scores.

Fase 5 — Tarea 5.4
- Overlay máscara + imagen con contorno
- Barra de confianza
- Score components
- Incertidumbre y top features
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def create_overlay(image: np.ndarray, mask, decision: dict,
                   xai_info: dict = None, output_path: str = None) -> np.ndarray:
    """
    Genera visualización final con overlay máscara + imagen + metadatos.

    Args:
        image: np.ndarray (H, W, 3) RGB
        mask: np.ndarray (H, W) o dict con key 'segmentation'
        decision: dict con scores, uncertainty, etc.
        xai_info: dict opcional con info XAI
        output_path: str opcional para guardar imagen

    Returns:
        np.ndarray — imagen con overlay (H, W, 3) uint8
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Normalizar máscara
    if isinstance(mask, dict):
        mask = mask['segmentation']
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')

    # Panel 1: Imagen original
    axes[0].imshow(image)
    axes[0].set_title("Imagen Original", fontsize=13, color='white', pad=10)
    axes[0].axis("off")

    # Panel 2: Máscara como heatmap
    axes[1].imshow(image, alpha=0.3)
    axes[1].imshow(mask_bool.astype(float), cmap='Reds', alpha=0.7)
    axes[1].set_title("Máscara Seleccionada", fontsize=13, color='white', pad=10)
    axes[1].axis("off")

    # Panel 3: Overlay con contorno
    overlay = _blend_mask(image, mask_bool)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=13, color='white', pad=10)
    axes[2].axis("off")

    # Info de scores
    score = decision.get('best_score', decision.get('score', 0))
    uncertainty = decision.get('uncertainty', None)
    accepted = decision.get('accepted', None)

    bar = create_confidence_bar(float(score))
    info = f"Confianza: {bar}"
    if uncertainty is not None:
        info += f"  |  Incertidumbre: {uncertainty:.4f}"
    if accepted is not None:
        status = "Aceptada" if accepted else "Rechazada"
        info += f"  |  {status}"

    fig.text(0.5, 0.02, info, ha='center', fontsize=11, color='#00ff88',
             fontfamily='monospace')

    # XAI info si disponible
    if xai_info and 'top_contributing' in xai_info:
        top_dims = xai_info['top_contributing']['dims'][:5]
        fig.text(0.5, 0.96, f"Top dims: {top_dims}", ha='center',
                 fontsize=9, color='#aaaaaa', fontfamily='monospace')

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        logger.info("Visualización guardada en %s", output_path)

    plt.close(fig)
    return overlay


def _blend_mask(image: np.ndarray, mask_bool: np.ndarray,
                color=(0, 255, 128), alpha: float = 0.4) -> np.ndarray:
    """
    Aplica overlay de color sobre la región de la máscara con contorno.

    Args:
        image: (H, W, 3) uint8
        mask_bool: (H, W) bool
        color: tupla RGB del color del overlay
        alpha: transparencia del overlay

    Returns:
        np.ndarray (H, W, 3) uint8
    """
    overlay = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color_arr * alpha
    )

    # Contorno por dilatación simple
    contour = _compute_contour(mask_bool)
    overlay[contour] = [255, 255, 0]  # Contorno amarillo

    return overlay.clip(0, 255).astype(np.uint8)


def _compute_contour(mask_bool: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Calcula contorno de una máscara binaria por dilatación."""
    try:
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(mask_bool, iterations=thickness)
        return dilated ^ mask_bool
    except ImportError:
        # Fallback sin scipy: dilatación manual simple
        padded = np.pad(mask_bool, 1, mode='constant')
        dilated = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:]
        )
        return dilated ^ mask_bool


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
