"""
Métricas de evaluación — IoU, Dice, tiempo, VRAM.

Fase 6 — Tarea 6.2
"""

import numpy as np


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                    pred_time_ms: float = 0.0, vram_mb: float = 0.0) -> dict:
    """
    Calcula métricas de segmentación.

    Args:
        pred_mask: np.ndarray binaria (H, W) — máscara predicha
        gt_mask: np.ndarray binaria (H, W) — máscara ground truth
        pred_time_ms: tiempo de inferencia en ms
        vram_mb: uso de VRAM en MB

    Returns:
        dict con iou, dice, time_ms, vram_mb
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0

    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0

    return {
        'iou': float(iou),
        'dice': float(dice),
        'time_ms': pred_time_ms,
        'vram_mb': vram_mb,
    }


def compute_metrics_batch(predictions: list, ground_truths: list) -> dict:
    """
    Calcula métricas promedio sobre un batch de predicciones.

    Args:
        predictions: List[dict] con pred_mask, pred_time_ms, vram_mb
        ground_truths: List[np.ndarray] — máscaras ground truth

    Returns:
        dict con mean_iou, mean_dice, mean_time_ms, mean_vram_mb
    """
    all_metrics = []
    for pred, gt in zip(predictions, ground_truths):
        m = compute_metrics(
            pred['pred_mask'], gt,
            pred.get('pred_time_ms', 0.0),
            pred.get('vram_mb', 0.0),
        )
        all_metrics.append(m)

    return {
        'mean_iou': np.mean([m['iou'] for m in all_metrics]),
        'mean_dice': np.mean([m['dice'] for m in all_metrics]),
        'mean_time_ms': np.mean([m['time_ms'] for m in all_metrics]),
        'mean_vram_mb': np.mean([m['vram_mb'] for m in all_metrics]),
        'n_samples': len(all_metrics),
    }
