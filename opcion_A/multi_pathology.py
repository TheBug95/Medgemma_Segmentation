"""
Multi-Patología — Banco FD + Text-Guided Routing + Iterador.

Fase 4 — Tareas 4.1, 4.2, 4.3
- Itera cada patología detectada en el texto
- Encuentra la mejor máscara para cada patología
- Routing automático desde texto MedGemma
"""

from opcion_A.pathology_parser import parse_pathology
from opcion_A.cosine_fusion import rank_masks_by_cosine
from opcion_A.fd_uncertainty import FDUncertaintyEstimator


class MultiPathologyPipeline:
    """
    Para imágenes con múltiples hallazgos, itera cada patología detectada
    y encuentra la mejor máscara para cada una.
    """

    def __init__(self, fd_bank, scorer, feature_extractor,
                 fd_uq: FDUncertaintyEstimator = None):
        self.fd_bank = fd_bank
        self.scorer = scorer
        self.feature_extractor = feature_extractor
        self.fd_uq = fd_uq or FDUncertaintyEstimator()

    def process(self, image, medgemma_text: str, candidate_masks: list,
                text_emb) -> list:
        """
        Procesa una imagen con múltiples hallazgos.

        Args:
            image: np.ndarray — imagen original
            medgemma_text: str — texto diagnóstico de MedGemma
            candidate_masks: List[dict] — máscaras de SAM 2
            text_emb: torch.Tensor (1, dim) — embedding del texto

        Returns:
            List[dict] — una entrada por patología detectada con:
                pathology, mask, score, uncertainty, s_fsl, s_cos, s_sam
        """
        pathologies = parse_pathology(medgemma_text)
        mask_embs = [self.feature_extractor.extract(image, m['segmentation'])
                     for m in candidate_masks]

        results = []
        for pathology in pathologies:
            if not self.fd_bank.has_pathology(pathology):
                continue

            entry = self.fd_bank.entries[pathology]
            best_result = None
            best_score = float('-inf')

            for i, (mask_dict, emb) in enumerate(zip(candidate_masks, mask_embs)):
                fd_result = self.fd_bank.evaluate(pathology, emb)
                s_fsl = fd_result['s_fsl']

                cos_result = rank_masks_by_cosine(
                    emb.unsqueeze(0), text_emb
                )
                s_cos = cos_result['best_score']
                s_sam = mask_dict.get('predicted_iou', 0.5)

                s_total = (self.scorer.alpha * s_fsl +
                           self.scorer.beta * s_cos +
                           self.scorer.gamma * s_sam)

                # FD-Uncertainty (señales distribucionales, 0 forward passes)
                u_composite, u_components = self.fd_uq.estimate(
                    emb, entry['kde'], entry['ood'], entry['support_embs'],
                    s_fsl, s_cos, s_sam,
                )

                if (s_total >= self.scorer.tau and
                        fd_result['in_ood'] and
                        u_composite <= 0.5):
                    if s_total > best_score:
                        best_score = s_total
                        best_result = {
                            'pathology': pathology,
                            'mask': mask_dict,
                            'mask_idx': i,
                            'score': s_total,
                            'uncertainty': u_composite,
                            'uq_components': u_components,
                            's_fsl': s_fsl,
                            's_cos': s_cos,
                            's_sam': s_sam,
                        }

            if best_result:
                results.append(best_result)

        return results
