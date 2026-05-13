"""
Orquestador principal del pipeline MedGemma-Seg.

Fase 1 — Tarea 1.6
- Pipeline secuencial: MedGemma → SAM 2 → Selección (coseno / FSL-FD)
- NO modifica MedGemma (caja negra)
- Integración end-to-end con Variante 1 funcional
- CLI con argparse para ejecución desde terminal
"""

import argparse
import logging
import numpy as np
from PIL import Image

from opcion_A.medgemma_loader import MedGemmaLoader
from opcion_A.sam_loader import SAMLoader
from opcion_A.feature_extractor import MaskFeatureExtractor
from opcion_A.text_encoder import TextEncoder
from opcion_A.cosine_fusion import rank_masks_by_cosine
from opcion_A.scoring import CompositeScorer, select_best_mask
from opcion_A.fsl_fd import FSLFDModule
from opcion_A.fd_uncertainty import FDUncertaintyEstimator

logger = logging.getLogger(__name__)


class MedGemmaSegPipeline:
    """
    Pipeline principal de segmentación multimodal.

    Etapa 1: MedGemma genera texto diagnóstico (~8 GB)
    Etapa 2: SAM 2 genera máscaras candidatas (~0.3 GB)
    Etapa 3a: MedSigLIP extrae features de máscaras (~1.5 GB)
    Etapa 3b: SigLIP codifica texto → comparación coseno + FSL/FD

    Total VRAM estimado: ~10 GB (cabe en T4 16GB)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.medgemma = MedGemmaLoader()
        self.sam = SAMLoader(device=device)
        self.feature_extractor = MaskFeatureExtractor(device=device)
        self.text_encoder = TextEncoder(device=device)
        self.scorer = CompositeScorer()

    def load(self):
        """Carga todos los modelos del pipeline."""
        logger.info("Cargando MedGemma 4B-IT...")
        self.medgemma.load()

        logger.info("Cargando SAM 2 Tiny...")
        self.sam.load()

        logger.info("Cargando MedSigLIP standalone (features)...")
        self.feature_extractor.load()

        logger.info("Cargando SigLIP text encoder...")
        self.text_encoder.load()

        logger.info("Todos los modelos cargados.")

    def __call__(self, image, prompt: str = "Describe y segmenta la patología") -> dict:
        """
        Ejecución completa del pipeline (coseno baseline).

        Args:
            image: np.ndarray (H, W, 3) RGB — imagen médica
            prompt: str — prompt para MedGemma

        Returns:
            dict con:
                - text: texto diagnóstico
                - masks: máscaras candidatas de SAM 2
                - selected_mask: mejor máscara según coseno
                - cosine_scores: scores de coseno por máscara
                - best_score: mejor score de coseno
        """
        # Etapa 1: Texto diagnóstico
        logger.info("Etapa 1: Generando texto diagnóstico...")
        text = self.medgemma.generate(image, prompt)
        text_emb = self.text_encoder.encode(text)

        # Etapa 2: Máscaras candidatas
        logger.info("Etapa 2: Generando máscaras candidatas (SAM 2)...")
        candidate_masks = self.sam.generate_masks(image)
        logger.info("  %d máscaras candidatas generadas.", len(candidate_masks))

        # Etapa 3: Features + Ranking por coseno
        logger.info("Etapa 3: Extrayendo features y rankeando...")
        mask_embs = self.feature_extractor.extract_batch(image, candidate_masks)
        cos_result = rank_masks_by_cosine(mask_embs, text_emb)

        best_idx = cos_result['best_idx']
        return {
            'text': text,
            'masks': candidate_masks,
            'selected_mask': candidate_masks[best_idx],
            'cosine_scores': cos_result['scores'].tolist(),
            'best_score': cos_result['best_score'],
            'best_idx': best_idx,
            'ranking': cos_result['ranking'].tolist(),
        }

    def run_with_fsl(self, image, support_embeddings=None,
                     prompt: str = "Describe y segmenta la patología") -> dict:
        """
        Ejecución completa con FSL/FD + FD-Uncertainty (Variante 2).

        Args:
            image: np.ndarray (H, W, 3) RGB
            support_embeddings: torch.Tensor (k, dim) — support set para FSL/FD
            prompt: str — prompt para MedGemma

        Returns:
            dict extendido con scores FSL, decisión, incertidumbre
        """
        # Etapa 1: Texto
        text = self.medgemma.generate(image, prompt)
        text_emb = self.text_encoder.encode(text)

        # Etapa 2: Máscaras
        candidate_masks = self.sam.generate_masks(image)

        # Etapa 3: Features
        mask_embs = self.feature_extractor.extract_batch(image, candidate_masks)

        if support_embeddings is not None:
            # Con FSL/FD: selección rigurosa
            fd_module = FSLFDModule(support_embeddings)
            result = select_best_mask(
                image, candidate_masks, self.scorer,
                fd_module, text_emb, self.feature_extractor,
            )
            result['text'] = text
            return result
        else:
            # Sin support set: solo coseno
            logger.warning("Sin support_embeddings. Usando solo coseno.")
            basic = self(image, prompt)
            basic['warning'] = 'No support set provided. Cosine-only mode.'
            return basic

    def unload(self):
        """Libera memoria de todos los modelos."""
        self.medgemma.unload()
        self.sam.unload()
        self.feature_extractor.unload()
        self.text_encoder.unload()


def main():
    """CLI para ejecutar el pipeline desde terminal."""
    parser = argparse.ArgumentParser(
        description="MedGemma-Seg: Segmentación multimodal de imágenes médicas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python -m opcion_A.pipeline --image retinal.jpg
  python -m opcion_A.pipeline --image retinal.jpg --prompt "Detecta glaucoma"
  python -m opcion_A.pipeline --image retinal.jpg --mode fsl --support-set support.pt
        """,
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen médica")
    parser.add_argument("--prompt", default="Describe y segmenta la patología",
                        help="Prompt para MedGemma")
    parser.add_argument("--mode", choices=["cosine", "fsl"], default="cosine",
                        help="Modo: cosine (baseline) o fsl (FSL/FD riguroso)")
    parser.add_argument("--support-set", default=None,
                        help="Ruta a support set embeddings (.pt) para modo fsl")
    parser.add_argument("--device", default="cuda", help="Dispositivo (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    args = parser.parse_args()

    # Configurar logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Cargar imagen
    image = np.array(Image.open(args.image).convert("RGB"))

    # Crear y cargar pipeline
    pipeline = MedGemmaSegPipeline(device=args.device)
    pipeline.load()

    # Ejecutar
    if args.mode == "fsl":
        support_embs = None
        if args.support_set:
            import torch
            support_embs = torch.load(args.support_set, map_location=args.device)
            logger.info("Support set cargado: %s", support_embs.shape)
        result = pipeline.run_with_fsl(image, support_embs, args.prompt)
    else:
        result = pipeline(image, args.prompt)

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS MedGemma-Seg")
    print("=" * 60)
    print(f"\nTexto diagnóstico:\n  {result['text']}")
    print(f"\nMáscaras candidatas: {len(result.get('masks', []))}")
    print(f"Mejor score: {result.get('best_score', result.get('score', 'N/A')):.4f}")
    print(f"Aceptada: {result.get('accepted', 'N/A')}")

    if result.get('rejection_reason'):
        print(f"Razón de rechazo: {result['rejection_reason']}")

    if result.get('uncertainty') is not None:
        print(f"Incertidumbre: {result['uncertainty']:.4f}")

    pipeline.unload()
    print("\n✅ Pipeline completado.")


if __name__ == "__main__":
    main()
