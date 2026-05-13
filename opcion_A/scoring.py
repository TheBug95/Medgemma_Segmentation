"""
Score Compuesto y Decisión Final.

Fase 3 — Tareas 3.1, 3.2, 3.3
- S = 0.5·s_FSL + 0.3·s_cos + 0.2·s_SAM
- 3 condiciones: S ≥ τ AND ℓ* ∈ [Θ_min, Θ_max] AND U ≤ 0.5
- Rechazo iterativo de candidatas
"""

from opcion_A.cosine_fusion import rank_masks_by_cosine


class CompositeScorer:
    """Score compuesto que integra 3 fuentes de señal + incertidumbre."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.3,
                 gamma: float = 0.2, tau: float = 0.6):
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
                     fd_module, text_emb, feature_extractor) -> dict:
    """
    Evalúa máscaras en orden (por coseno) hasta encontrar una que pase las 3 condiciones.
    Si ninguna pasa, devuelve la mejor por score compuesto con advertencia.

    Args:
        image: np.ndarray (H, W, 3) — imagen original
        masks: List[dict] — máscaras candidatas de SAM 2
        scorer: CompositeScorer — pesos y umbral
        fd_module: FSLFDModule — módulo FSL/FD calibrado
        text_emb: torch.Tensor (1, dim) — embedding del texto diagnóstico
        feature_extractor: MaskFeatureExtractor — extractor de features

    Returns:
        dict con: mask, accepted, score, uncertainty, log_density, rank,
                  rejection_reason (si no accepted)
    """
    # Extraer embeddings de todas las máscaras
    mask_embs = feature_extractor.extract_batch(image, masks)

    # Ranking inicial por coseno (Variante 1)
    cos_result = rank_masks_by_cosine(mask_embs, text_emb)
    ranking = cos_result['ranking']
    cos_scores = cos_result['scores']

    best_fallback = None
    best_fallback_score = float('-inf')
    best_fallback_meta = {}

    for rank_idx, idx in enumerate(ranking):
        idx = idx.item()
        mask_dict = masks[idx]
        emb = mask_embs[idx]
        s_cos = cos_scores[idx].item()
        s_sam = mask_dict.get('predicted_iou', 0.5)

        # FSL/FD (Variante 2)
        fd_result = fd_module.evaluate(emb, s_cos=s_cos, s_sam=s_sam)
        s_fsl = fd_result['s_fsl']
        in_ood = fd_result['in_ood']
        u_composite = fd_result['uncertainty']
        log_density = fd_result['log_density']

        # Score compuesto
        s_total = (scorer.alpha * s_fsl +
                   scorer.beta * s_cos +
                   scorer.gamma * s_sam)

        # Guardar mejor fallback
        if s_total > best_fallback_score:
            best_fallback = mask_dict
            best_fallback_score = s_total
            best_fallback_meta = {
                's_total': s_total,
                'uncertainty': u_composite,
                'log_density': log_density,
                's_fsl': s_fsl,
                's_cos': s_cos,
                's_sam': s_sam,
            }

        # Decisión: 3 condiciones
        if s_total >= scorer.tau and in_ood and u_composite <= 0.5:
            return {
                'mask': mask_dict,
                'mask_idx': idx,
                'accepted': True,
                'score': s_total,
                'uncertainty': u_composite,
                'log_density': log_density,
                's_fsl': s_fsl,
                's_cos': s_cos,
                's_sam': s_sam,
                'rank': rank_idx,
            }

    # Fallback: ninguna máscara pasó las 3 condiciones
    return {
        'mask': best_fallback,
        'mask_idx': None,
        'accepted': False,
        'warning': 'No mask passed all 3 conditions. Returning best fallback.',
        'score': best_fallback_meta.get('s_total', 0.0),
        'uncertainty': best_fallback_meta.get('uncertainty', 1.0),
        'log_density': best_fallback_meta.get('log_density', 0.0),
        's_fsl': best_fallback_meta.get('s_fsl', 0.0),
        's_cos': best_fallback_meta.get('s_cos', 0.0),
        's_sam': best_fallback_meta.get('s_sam', 0.0),
        'rank': len(ranking) - 1,
        'rejection_reason': 'No mask passed all 3 conditions',
    }
