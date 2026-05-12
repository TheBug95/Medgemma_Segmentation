"""
Orquestador principal del pipeline MedGemma-Seg.

Fase 1 — Tarea 1.6
- Pipeline secuencial: MedGemma → SAM 2 → Selección (coseno / FSL-FD)
- NO modifica MedGemma (caja negra)
- Integración end-to-end con Variante 1 funcional
"""


class MedGemmaSegPipeline:
    """
    Pipeline principal de segmentación multimodal.

    Etapa 1: MedGemma genera texto diagnóstico
    Etapa 2: SAM 2 genera máscaras candidatas
    Etapa 3: Selección guiada (coseno baseline → FSL/FD validación)
    """

    def __init__(self):
        self.medgemma = None
        self.sam = None
        self.feature_extractor = None
        self.text_encoder = None

    def load(self):
        """Carga todos los modelos del pipeline."""
        raise NotImplementedError("Tarea 1.6 pendiente de implementación")

    def __call__(self, image, prompt: str = "Describe y segmenta la patología") -> dict:
        """
        Ejecución completa del pipeline.

        Args:
            image: np.ndarray (H, W, 3) — imagen médica
            prompt: str — prompt para MedGemma

        Returns:
            dict con:
                - text: texto diagnóstico
                - masks: máscaras candidatas
                - selected_mask: máscara seleccionada
                - cosine_scores: scores de coseno
                - best_score: mejor score
        """
        raise NotImplementedError("Tarea 1.6 pendiente de implementación")
