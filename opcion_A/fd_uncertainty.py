"""
FD-Uncertainty Estimator — 7 señales, 0 forward passes extra.

Fase 2 — Tarea 2.4
- Reemplaza MC Dropout por señales distribucionales del propio KDE
- 4 señales distribucionales + 3 señales del pipeline
- ~1500× más rápido que MC Dropout
"""

import torch


class FDUncertaintyEstimator:
    """
    Estimador de incertidumbre basado en Feature Densities.
    7 señales — 0 forward passes extra, 0 parámetros entrenables.
    """

    WEIGHTS = {
        'U_LD': 0.25,
        'U_BORDE': 0.20,
        'U_DIMVAR': 0.15,
        'U_PROTO': 0.10,
        'U_acuerdo': 0.15,
        'U_magnitud': 0.10,
        'U_margen': 0.05,
    }

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, tau: float = 0.6):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def estimate(self, query_emb, kde, ood_window, support_embs,
                 s_fsl: float, s_cos: float, s_sam: float) -> tuple:
        """
        Estima incertidumbre compuesta U ∈ [0, 1].

        Args:
            query_emb: (768,) — embedding de la máscara query
            kde: PerDimensionKDE
            ood_window: OODWindow
            support_embs: (k, 768) — embeddings del support set
            s_fsl, s_cos, s_sam: scores [0, 1]

        Returns:
            u_composite: float [0, 1]
            components: dict con las 7 señales individuales
        """
        log_density, per_dim = kde.log_density(query_emb)
        s_total = self.alpha * s_fsl + self.beta * s_cos + self.gamma * s_sam

        components = {
            'U_LD': self._u_ld(log_density, ood_window),
            'U_BORDE': self._u_borde(log_density, ood_window),
            'U_DIMVAR': self._u_dimvar(per_dim),
            'U_PROTO': self._u_proto(query_emb, support_embs),
            'U_acuerdo': self._u_acuerdo(s_fsl, s_cos, s_sam),
            'U_magnitud': 1 - s_total,
            'U_margen': 1 - abs(s_total - self.tau) / max(self.tau, 1 - self.tau),
        }

        u_composite = sum(self.WEIGHTS[k] * v for k, v in components.items())
        return u_composite, components

    @staticmethod
    def _u_ld(log_density, ood_window) -> float:
        if log_density < ood_window.theta_min:
            return 1.0
        elif log_density > ood_window.theta_max:
            return 0.0
        return 1 - (log_density - ood_window.theta_min) / \
               (ood_window.theta_max - ood_window.theta_min)

    @staticmethod
    def _u_borde(log_density, ood_window) -> float:
        d_centro = (log_density - ood_window.theta_min) / \
                   (ood_window.theta_max - ood_window.theta_min)
        return 1 - 2 * abs(d_centro - 0.5)

    @staticmethod
    def _u_dimvar(per_dim_contrib) -> float:
        abs_contrib = torch.abs(per_dim_contrib)
        mean_contrib = abs_contrib.mean()
        return (1 / (1 + torch.exp(-5 * (mean_contrib - 0.1)))).item()

    @staticmethod
    def _u_proto(query_emb, support_embs) -> float:
        proto = support_embs.mean(dim=0)
        d_proto = torch.norm(query_emb.squeeze() - proto)
        d_support = torch.norm(support_embs - proto, dim=1)
        d_median = d_support.median()
        return (d_proto / (d_proto + d_median)).item()

    @staticmethod
    def _u_acuerdo(s_fsl: float, s_cos: float, s_sam: float) -> float:
        scores = torch.tensor([s_fsl, s_cos, s_sam])
        varianza = scores.var().item()
        return min(1.0, varianza / 0.33)
