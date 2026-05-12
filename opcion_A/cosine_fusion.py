"""
Módulo de fusión por Distancia Coseno.

Fase 1 — Tarea 1.5
- Variante 1 del pipeline: ranking rápido texto↔máscara
- Compara embedding del texto diagnóstico con embeddings de máscaras
- Score s_cos ∈ [0, 1]
"""

import torch
import torch.nn.functional as F


def rank_masks_by_cosine(mask_embeddings: torch.Tensor, text_embedding: torch.Tensor) -> dict:
    """
    Ranking de máscaras por similitud coseno con el texto diagnóstico.

    Args:
        mask_embeddings: (N, 768) — embeddings de N máscaras candidatas
        text_embedding:  (1, 768) — embedding del texto diagnóstico

    Returns:
        dict con:
            - ranking: índices ordenados por coseno descendente
            - scores: tensor (N,) con scores de coseno
            - best_idx: índice de la mejor máscara
            - best_score: score de la mejor máscara
    """
    mask_emb = F.normalize(mask_embeddings, p=2, dim=-1)
    text_emb = F.normalize(text_embedding, p=2, dim=-1)

    similarities = (mask_emb @ text_emb.T).squeeze()  # (N,)
    ranking = torch.argsort(similarities, descending=True)

    return {
        'ranking': ranking,
        'scores': similarities,
        'best_idx': ranking[0].item(),
        'best_score': similarities[ranking[0]].item(),
    }
