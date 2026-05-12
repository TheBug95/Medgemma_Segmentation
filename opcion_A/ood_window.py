"""
Ventana OOD [Θ_min, Θ_max] — Paper IGPL.

Fase 2 — Tarea 2.2
- Calibra la ventana de aceptación sobre el support set
- Método IQR: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Método MinMax: [min(ℓ*), max(ℓ*)]
"""

import torch


class OODWindow:
    """
    Ventana OOD para aceptación/rechazo de máscaras candidatas.

    Calibra [Θ_min, Θ_max] sobre las log-densidades del support set.
    """

    def __init__(self, method: str = 'iqr'):
        """
        Args:
            method: 'iqr' o 'minmax'
        """
        self.method = method
        self.theta_min = None
        self.theta_max = None

    def calibrate(self, support_log_densities: torch.Tensor):
        """
        Calibra la ventana OOD usando las log-densidades del support set.

        Args:
            support_log_densities: (k,) — ℓ* de cada elemento del support set
        """
        if self.method == 'iqr':
            q1 = torch.quantile(support_log_densities, 0.25)
            q3 = torch.quantile(support_log_densities, 0.75)
            iqr = q3 - q1
            self.theta_min = q1 - 1.5 * iqr
            self.theta_max = q3 + 1.5 * iqr
        elif self.method == 'minmax':
            self.theta_min = support_log_densities.min()
            self.theta_max = support_log_densities.max()
        else:
            raise ValueError(f"Método desconocido: {self.method}")

    def is_in_window(self, log_density: float) -> bool:
        """Verifica si una log-densidad está dentro de la ventana OOD."""
        return self.theta_min <= log_density <= self.theta_max

    def normalize(self, log_density: float) -> float:
        """Normaliza ℓ* a [0, 1] usando la ventana."""
        if log_density < self.theta_min:
            return 0.0
        elif log_density > self.theta_max:
            return 1.0
        return (log_density - self.theta_min) / (self.theta_max - self.theta_min)
