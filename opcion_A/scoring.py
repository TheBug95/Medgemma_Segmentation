"""
Score Compuesto y Decisión Final.

Fase 3 — Tareas 3.1, 3.2, 3.3
- S = 0.5·s_FSL + 0.3·s_cos + 0.2·s_SAM
- 3 condiciones: S ≥ τ AND ℓ* ∈ [Θ_min, Θ_max] AND U ≤ 0.5
- Rechazo iterativo de candidatas
"""

from typing import Optional


class CompositeScorer:
    """
    Score compuesto que integra 3 fuentes de señal + incertidumbre.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, tau: float = 0.6):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def compute(self, s_fsl: float, s_cos: float, s_sam: float) -> dict:
        """
        Calcula el score compuesto S.

        Returns:
            dict con s_total, s_fsl, s_cos, s_sam, weights
        """
        s_total = self.alpha * s_fsl + self.beta * s_cos + self.gamma * s_sam
        return {
            's_total': s_total,
            's_fsl': s_fsl,
            's_cos': s_cos,
            's_sam': s_sam,
            'weights': {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma},
        }


def make_decision(s_total: float, log_density: float, uncertainty: float,
                  theta_min: float, theta_max: float, tau: float = 0.6) -> dict:
    """
    Decisión final con 3 condiciones.

    1. s_total >= tau → score compuesto supera el umbral
    2. ℓ* en [Θ_min, Θ_max] → log-density dentro de la ventana OOD
    3. U <= 0.5 → incertidumbre aceptable

    Returns:
        dict con accepted, conditions, s_total, uncertainty, rejection_reason
    """
    in_ood = theta_min <= log_density <= theta_max
    low_uncertainty = uncertainty <= 0.5
    score_ok = s_total >= tau
    accepted = score_ok and in_ood and low_uncertainty

    reasons = []
    if not score_ok:
        reasons.append(f"score {s_total:.3f} < tau {tau}")
    if not in_ood:
        reasons.append(f"log_density {log_density:.1f} fuera de [{theta_min:.1f}, {theta_max:.1f}]")
    if not low_uncertainty:
        reasons.append(f"uncertainty {uncertainty:.3f} > 0.5")

    return {
        'accepted': accepted,
        'conditions': {'score_ok': score_ok, 'ood_ok': in_ood, 'uncertainty_ok': low_uncertainty},
        's_total': s_total,
        'uncertainty': uncertainty,
        'rejection_reason': '; '.join(reasons) if reasons else None,
    }


def select_best_mask(image, masks: list, scorer: CompositeScorer,
                     fd_module, text_emb, feature_extractor,
                     cosine_rank_fn) -> dict:
    """
    Evalúa máscaras en orden (por coseno) hasta encontrar una que pase las 3 condiciones.
    Si ninguna pasa, devuelve la mejor por score compuesto con advertencia.
    """
    raise NotImplementedError("Tarea 3.3 pendiente de implementación")
