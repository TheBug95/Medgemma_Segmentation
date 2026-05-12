"""
Módulo FSL/FD principal — Paper IGPL.

Fase 2
- Integra KDE por dimensión + ventana OOD + FD-Uncertainty
- Score s_FSL ∈ [0, 1]
- Validación ℓ* ∈ [Θ_min, Θ_max]
"""

import torch
from opcion_A.fd_kde import PerDimensionKDE
from opcion_A.ood_window import OODWindow
from opcion_A.fd_uncertainty import FDUncertaintyEstimator


class FSLFDModule:
    """
    Módulo Feature Similarity Learning / Feature Densities (IGPL).

    Combina:
    - PerDimensionKDE: KDE por dimensión con h = k^(-1/5)
    - OODWindow: ventana de aceptación [Θ_min, Θ_max]
    - FDUncertaintyEstimator: 7 señales, 0 forward passes extra
    """

    def __init__(self, support_embeddings, method: str = 'iqr'):
        """
        Args:
            support_embeddings: (k, 768) — embeddings del support set
            method: 'iqr' o 'minmax' para la ventana OOD
        """
        self.kde = PerDimensionKDE(support_embeddings)
        self.ood_window = OODWindow(method=method)
        self.fd_uq = FDUncertaintyEstimator()
        self.support_embs = support_embeddings

        # Calibrar ventana OOD
        self._calibrate()

    def _calibrate(self):
        """Calibra la ventana OOD sobre el support set."""
        support_log_densities = []
        for i in range(len(self.support_embs)):
            ld, _ = self.kde.log_density(self.support_embs[i])
            support_log_densities.append(ld)
        self.ood_window.calibrate(torch.tensor(support_log_densities))

    def evaluate(self, query_emb, s_cos: float = 0.0, s_sam: float = 0.0) -> dict:
        """
        Evalúa una máscara candidata contra la distribución FD.

        Args:
            query_emb: (768,) — embedding de la máscara query
            s_cos: score de coseno (para FD-UQ)
            s_sam: score de SAM (para FD-UQ)

        Returns:
            dict con: log_density, s_fsl, in_ood, per_dim, uncertainty, uq_components
        """
        log_density, per_dim = self.kde.log_density(query_emb)
        s_fsl = self.ood_window.normalize(log_density)
        in_ood = self.ood_window.is_in_window(log_density)

        u_composite, u_components = self.fd_uq.estimate(
            query_emb, self.kde, self.ood_window, self.support_embs,
            s_fsl, s_cos, s_sam,
        )

        return {
            'log_density': log_density,
            's_fsl': s_fsl,
            'in_ood': in_ood,
            'per_dim': per_dim,
            'uncertainty': u_composite,
            'uq_components': u_components,
        }
