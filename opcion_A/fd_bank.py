"""
Banco de distribuciones FD — una por patología.

Fase 4 — Tarea 4.1
- Almacena KDE + ventana OOD por patología
- Registro de nuevas patologías con solo k=6-12 máscaras
- 0 parámetros nuevos, 0 reentrenamiento
"""

import torch
import numpy as np
from opcion_A.fd_kde import PerDimensionKDE
from opcion_A.ood_window import OODWindow


class FDBank:
    """
    Banco de distribuciones FD — una entrada por patología.

    Cada entrada contiene:
    - KDE por dimensión
    - Ventana OOD [Θ_min, Θ_max]
    - Support set embeddings
    """

    def __init__(self):
        self.entries = {}  # {"cataract": {...}, "glaucoma": {...}, ...}

    def register(self, name: str, support_image: np.ndarray,
                 support_masks: list, medsiglip_encoder):
        """
        Registra una nueva patología con k máscaras de ejemplo.

        Args:
            name: nombre de la patología
            support_image: np.ndarray (H, W, 3) — imagen de referencia
            support_masks: List[np.ndarray] — k máscaras binarias (k=6-12)
            medsiglip_encoder: MaskFeatureExtractor para extraer embeddings
        """
        embeddings = []
        for mask in support_masks:
            emb = medsiglip_encoder.extract(support_image, mask)
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)  # (k, 768)
        kde = PerDimensionKDE(embeddings)
        ood = OODWindow(method='iqr')

        # Calibrar ventana OOD sobre el support set
        support_log_densities = []
        for i in range(len(embeddings)):
            ld, _ = kde.log_density(embeddings[i])
            support_log_densities.append(ld)
        ood.calibrate(torch.tensor(support_log_densities))

        self.entries[name] = {
            'kde': kde,
            'ood': ood,
            'support_embs': embeddings,
            'n_support': len(support_masks),
        }

    def evaluate(self, name: str, query_emb: torch.Tensor) -> dict:
        """
        Evalúa un query contra la distribución de la patología name.

        Returns:
            dict con log_density, s_fsl, in_ood, per_dim_contrib
        """
        entry = self.entries[name]
        log_density, per_dim = entry['kde'].log_density(query_emb)
        s_fsl = entry['ood'].normalize(log_density)
        in_ood = entry['ood'].is_in_window(log_density)
        return {
            'log_density': log_density,
            's_fsl': s_fsl,
            'in_ood': in_ood,
            'per_dim_contrib': per_dim,
        }

    def get_pathologies(self) -> list:
        """Retorna lista de patologías registradas."""
        return list(self.entries.keys())

    def has_pathology(self, name: str) -> bool:
        """Verifica si una patología está registrada."""
        return name in self.entries
