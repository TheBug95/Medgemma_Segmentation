"""
Explicabilidad (XAI) — Por qué el sistema seleccionó o rechazó una máscara.

Fase 5 — Tareas 5.1, 5.2, 5.3
- Heatmap de contribución por dimensión (768-dim)
- Visualización de borde OOD
- Desglose de incertidumbre por fuente
"""

import torch


def dimension_contribution_heatmap(per_dim_contrib: torch.Tensor, top_k: int = 10) -> dict:
    """
    Identifica las dimensiones que más contribuyen al log-density total.

    Args:
        per_dim_contrib: (768,) — contribución individual log p̂ⱼ(z*ⱼ) por dimensión
        top_k: número de dimensiones top a retornar

    Returns:
        dict con top_contributing, bottom_contributing, total_log_density
    """
    top_dims = torch.topk(per_dim_contrib, k=top_k)
    bottom_dims = torch.topk(per_dim_contrib, k=top_k, largest=False)

    return {
        'top_contributing': {
            'dims': top_dims.indices.tolist(),
            'values': top_dims.values.tolist(),
        },
        'bottom_contributing': {
            'dims': bottom_dims.indices.tolist(),
            'values': bottom_dims.values.tolist(),
        },
        'total_log_density': per_dim_contrib.sum().item(),
    }


def ood_boundary_visualization(log_density: float, theta_min: float,
                                theta_max: float) -> dict:
    """
    Analiza qué tan cerca está el query del borde OOD.

    Returns:
        dict con distances, centered_score, interpretation
    """
    d_below = log_density - theta_min
    d_above = theta_max - log_density
    centered = (log_density - theta_min) / (theta_max - theta_min)

    return {
        'log_density': log_density,
        'theta_min': theta_min,
        'theta_max': theta_max,
        'distance_below': d_below,
        'distance_above': d_above,
        'centered_score': centered,
        'interpretation': (
            'well_centered' if 0.25 <= centered <= 0.75
            else 'near_boundary'
        ),
    }


def uncertainty_breakdown(u_components: dict) -> dict:
    """
    Desglose de incertidumbre por fuente.

    Args:
        u_components: dict con las 7 señales individuales

    Returns:
        dict con total, distributional_pct, pipeline_pct, components
    """
    distributional = (u_components['U_LD'] + u_components['U_BORDE'] +
                      u_components['U_DIMVAR'] + u_components['U_PROTO'])
    pipeline = (u_components['U_acuerdo'] + u_components['U_magnitud'] +
                u_components['U_margen'])
    total = distributional + pipeline

    return {
        'total': total,
        'distributional_pct': (distributional / total * 100) if total > 0 else 0,
        'pipeline_pct': (pipeline / total * 100) if total > 0 else 0,
        'components': u_components,
    }
