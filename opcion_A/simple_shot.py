"""
Simple-Shot — Complementario a FSL/FD para k ≥ 15.

Fase 2 — Tarea 2.3
- Clasifica por similitud coseno al centroide de cada clase
- Paper IGPL muestra que domina para k ≥ 15
- Alternativa a FD para support sets grandes
"""

import torch
import torch.nn.functional as F


def simple_shot_score(
    query_emb: torch.Tensor,
    support_embs: torch.Tensor,
    support_labels: torch.Tensor,
) -> float:
    """
    Simple-Shot: clasifica por similitud coseno al centroide de cada clase.

    Args:
        query_emb: (768,) — embedding de la máscara query
        support_embs: (k, 768) — embeddings del support set
        support_labels: (k,) — 1 = objeto de interés, 0 = fondo

    Returns:
        score ∈ [0, 1]: confianza de que la máscara es 'objeto de interés'
    """
    # Centroide de la clase positiva
    positive_mask = support_labels == 1
    proto_positive = support_embs[positive_mask].mean(dim=0)

    # Centroide de la clase negativa (fondo)
    proto_negative = support_embs[~positive_mask].mean(dim=0)

    # Similitud coseno a cada prototipo
    cos_pos = F.cosine_similarity(query_emb, proto_positive, dim=0)
    cos_neg = F.cosine_similarity(query_emb, proto_negative, dim=0)

    # Softmax sobre las dos clases
    scores = torch.softmax(torch.stack([cos_pos, cos_neg]), dim=0)
    return scores[0].item()  # Score de la clase positiva
