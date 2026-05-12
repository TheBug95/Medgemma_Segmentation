# AGENTS.md — MedGemma Segmentation

## What this project does

Extend MedGemma (LLM médico multimodal) with image segmentation: given a medical image,
generate both a diagnostic text description **and** a segmentation mask showing exactly
where the pathology is.

## Architecture (CRITICAL)

- **Option A only** — pipeline secuencial. MedGemma is a black box. Do NOT modify it
  internally (no LoRA, no `[SEG]` tokens). Option B is documented but deferred.
- **3 stages**: MedGemma → text → SAM 2 → candidate masks → FSL/FD (IGPL) → best mask
- **Backbone**: Reuse MedSigLIP (the vision encoder inside MedGemma) as the ViT for FSL.
  Do NOT load a separate ViT. It's already in memory, pre-trained on medical images,
  448×448 resolution, 768-dim embeddings.
- **Two complementary selection variants** used sequentially:
  1. Cosine distance: text embedding ↔ mask embedding (fast ranking)
  2. FSL/FD (IGPL paper): per-dimension KDE + OOD window (rigorous validation)

## Key decisions already made

| Decision | Detail |
|---|---|
| MC Dropout replaced by | FD-Uncertainty Estimator (7 signals, 0 extra forward passes) |
| Multi-pathology from start | FD bank per pathology + text-guided routing |
| SAM version | SAM 2 Tiny (38.9M params) |
| Minimal GPU | 16 GB VRAM (MedGemma ~8 GB + SAM 2 ~2 GB) |
| Stack | Python 3.10+, PyTorch 2.x, HuggingFace `transformers`, `segment-anything` |

## What to implement (40 tasks, ~5 weeks)

Full spec in `implementation_guide_detallado.md`. That is the canonical reference.

### Phase 1 – Pipeline Base (weeks 1-2)
`opcion_A/pipeline.py`, `medgemma_loader.py`, `sam_loader.py`, `feature_extractor.py`,
`text_encoder.py`, `cosine_fusion.py`
→ Working end-to-end pipeline using cosine-only selection (Variant 1).

### Phase 2 – FSL/FD + FD-UQ (weeks 2-3)
`opcion_A/fsl_fd.py`, `fd_kde.py`, `ood_window.py`, `simple_shot.py`,
`fd_uncertainty.py`, plus `validation/fd_vs_mcd.py`
→ KDE per dimension with `h = k^(-1/5)` (IGPL exact), OOD window `[Θ_min, Θ_max]`,
FD-Uncertainty with 7 signals. Validate against MC Dropout (Pearson r > 0.7, ECE < 0.10).

### Phase 3 – Score + Decision (week 3)
`opcion_A/scoring.py`
→ `S = 0.5·s_FSL + 0.3·s_cos + 0.2·s_SAM`. Accept if: `S ≥ 0.6` AND `ℓ* ∈ [Θ_min, Θ_max]` AND `U ≤ 0.5`.

### Phase 4 – Multi-Pathology (weeks 3-4)
`opcion_A/multi_pathology.py`, `pathology_parser.py`, `fd_bank.py`
→ One FD module per pathology. Parser extracts pathology from MedGemma text via keyword matching.

### Phase 5 – XAI (weeks 4-5)
`opcion_A/xai.py`, `visualization.py`
→ Dimension contribution heatmap, OOD boundary visualization, uncertainty breakdown.

### Phase 6 – Benchmark BIP (week 5)
`opcion_A/benchmark.py`, `metrics.py`
→ Grid search over k, α, method. Compare cosine-only vs FSL-only vs combined.

## Hard rules from IGPL paper

- KDE bandwidth: `h = k^(-1/5)` — do not change this formula
- OOD window: `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` computed on support set log-densities
- FD works best for `k ≤ 12`; Simple-Shot takes over for `k ≥ 15`
- Support set: 6-12 correctly annotated masks per pathology, no more needed

## FD-Uncertainty: the 7 signals

Weights: `U = 0.25·U_LD + 0.20·U_borde + 0.15·U_dimvar + 0.10·U_proto + 0.15·U_acuerdo + 0.10·U_mag + 0.05·U_margen`

All computed post-hoc from the single MedSigLIP forward pass. No extra ViT calls.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run pipeline (single image)
python -m opcion_A.pipeline --image path/to/image.jpg

# Run FSL/FD validation
python -m opcion_A.validation.fd_vs_mcd

# Run benchmark grid search
python -m opcion_A.benchmark --k 6,9,12 --alpha 0.3,0.5,0.7
```

## Key files

| File | Purpose |
|---|---|
| `implementation_guide_detallado.md` | **Canonical reference.** All 40 tasks, flow diagrams, code templates |
| `resumen_paper_igpl_vit.md` | IGPL paper summary with MedSigLIP ViT adaptation |
| `propuesta_multi_patologia.md` | Multi-pathology FD bank + routing proposals |
| `implementation_plan.md` | Original option A/B proposal (Option B is deferred) |
| `analysis_lisa_glamm.md` | LISA and GLaMM architecture analysis |
| `comparativa_lisa_vs_opcionB.md` | Why Option B beats LISA on efficiency and medical domain |
| `requirements.txt` | Python dependencies (needs to be created) |

## Gotchas

- MedGemma is loaded via HuggingFace `transformers` in bf16. Do NOT use 4-bit quantization
  unless VRAM is tight — it degrades embedding quality.
- SAM confidence scores are NOT reliable on medical images (per IGPL). Always validate
  through FSL/FD.
- When adding a new pathology, you need ONLY k=6-12 mask examples. Zero new parameters,
  zero retraining.
- The `[SEG]` token and projection layer are Option B only. Do NOT implement these in
  Phase 1-6.
- MedSigLIP generates 256 visual tokens (not 255 like CLIP in LISA). If Option B is ever
  implemented, the seg token mask offset must be 256.
