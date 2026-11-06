# Multimodal MedGemma — Reference Implementation

Reference implementation accompanying:

> **MedGemma for Ophthalmology: Integrating SAM-Segmented Retinal Images and Conditioned Descriptions for Multimodal Medical Report Generation**
> M. Abreu-Cárdenas, D. Xie-Li, D. Alexander-Vargas, J.P. Pérez-Delgado, S. Cabrera-Tabash, S. Calderon-Ramirez, M. Masis-Solano, A. Garcia-Pedrero, K. Dueñas Ángeles, G.A. Osorio Navarro.
> Instituto Tecnológico de Costa Rica · Cornell Tech · Instituto Conde de Valencia · Universidad Politécnica de Madrid.

The paper proposes a three-stage pipeline that conditions MedGemma on automatically generated optic-disc segmentations and predicted disease labels for structured glaucoma report generation. **This repository holds the code for Stage I (classification and routing) in [`Stage_I/`](Stage_I) and Stage III (conditioned MedGemma report generation) in [`Stage_III/`](Stage_III).** Stage II, the SAM 2.1 + LoRA segmentation, is not versioned here — its outputs enter Stage III as pre-rendered overlays.

Dataset: [MM-ODIR-129](https://huggingface.co/datasets/TheBug95/MM-ODIR-129) on Hugging Face.

---

## 1. The pipeline, and where this code sits

| Stage | What the paper does | Code here |
|---|---|---|
| **I — Disease classification & routing** | A lightweight CNN predicts `glaucoma` / `normal`. Glaucoma cases are routed to segmentation; normal cases bypass it. The predicted label is also passed to MedGemma as prior clinical context. | ✅ [`Stage_I/`](Stage_I) |
| **II — Conditional image preparation** | SAM 2.1 adapted with LoRA predicts a binary optic-disc mask, composited as a semi-transparent overlay `x_overlay`. | ❌ Not versioned here |
| **III — Report generation** | MedGemma 1.5 4B receives the (optionally overlaid) image plus a prompt carrying the predicted label; six conditioning configurations are compared against a raw-image baseline. | ✅ [`Stage_III/`](Stage_III) |

Stage II's results still reach this repository as data: the segmentor masks and pre-rendered overlays it produced for the 26 test images ship in [`Stage_III/dataset/test/`](Stage_III/dataset), which is what makes the end-to-end *pipeline* evaluation source reproducible without re-running segmentation.

---

## 2. Paper → code map

### Stage I (paper §3.2 Stage I, §4.1)

| Paper element | Implemented by |
|---|---|
| Prototypical Networks, `p_c = (1/\|S_c\|) Σ f_θ(x_i)`, `ŷ = arg min_c ‖f_θ(x_q) − p_c‖²` | [`prototype_classifier.py`](Stage_I/Experiments/Classifier_Selection/modules/prototype_classifier.py) (EasyFSL `PrototypicalNetworks`) |
| *frozen encoder* setting (ImageNet backbone fixed, nearest-prototype inference) | `PrototypeClassifier` with method `euclidean` — the code's name for the paper's `frozen` |
| *episodic fine-tuning* setting (100 2-way episodes, N support + 10 query per class, cross-entropy, Adam lr 1e-3, augmentation) | `PrototypeClassifier.meta_train()`, parameterized in [`config.yaml`](Stage_I/Experiments/Classifier_Selection/config.yaml) under `meta_training` |
| Three ImageNet-pretrained backbones (ResNet-18, DenseNet-121, EfficientNet-B0) | [`backbones.py`](Stage_I/Experiments/Classifier_Selection/modules/backbones.py) |
| Class-balanced support set sampled from REFUGE train, N ∈ {1..5} | `build_support_indices()` in [`few_shot.py`](Stage_I/Experiments/Classifier_Selection/scripts/few_shot.py) |
| Five fixed seeds (42, 123, 456, 789, 1024) with **identical support sets shared across all backbones within each seed** | seed loop in [`run_experiment.py`](Stage_I/Experiments/Classifier_Selection/run_experiment.py) |
| macro-F1 on the REFUGE validation split | [`evaluate.py`](Stage_I/Experiments/Classifier_Selection/scripts/evaluate.py) |
| Grad-CAM localization (IoU with the optic-disc mask) | `evaluate_gradcam_quality()` in `run_experiment.py`, [`extract_gradcam.py`](Stage_I/Experiments/Classifier_Selection/scripts/extract_gradcam.py), [`gradcam_module.py`](Stage_I/modules/gradcam_module.py) |
| Peak inference memory at batch size 1 | [`benchmark_inference.py`](Stage_I/Experiments/Classifier_Selection/scripts/benchmark_inference.py) |
| REFUGE, 1200 annotated fundus images | [`convert_refuge_format.py`](Stage_I/Experiments/Classifier_Selection/scripts/convert_refuge_format.py) + [`data/`](Stage_I/Experiments/Classifier_Selection/data) |
| **Table 2** — backbone selection on REFUGE | [`selection_summary_few_shot(224x224).json`](Stage_I/Experiments/Classifier_Selection/results/selection_summary_few_shot%28224x224%29.json) |
| MM-ODIR-129 loading (patient-level splits, disc/cup masks, `Pathological`/`Normal` labels) | [`private_data_module.py`](Stage_I/Classifier/private_data_module.py) |
| Support-size sweep N ∈ {3, 6, …, 21} on MM-ODIR-129 | [`meta_nshot_experiment.ipynb`](Stage_I/Classifier/meta_nshot_experiment.ipynb) |
| **Table 3** — support-size selection, N = 6 chosen | produced by that notebook (results written to Drive, not versioned) |
| Deployed classifier on the 26-image test split | [`inference_on_test.py`](Stage_I/Classifier/Inference/inference_on_test.py) |
| Bootstrap and Clopper–Pearson 95 % intervals | [`metrics_generation.py`](Stage_I/Classifier/Inference/metrics_generation.py) |
| **Table 4** — deployed classifier performance | [`test_metrics.json`](Stage_I/Classifier/Inference/test_metrics.json) |
| Per-image label and glaucoma probability consumed by Stage III | `per_image` in `test_metrics.json` → [`pipeline_classifications.json`](Stage_III/dataset) |

### Stage III (paper §3.2 Stage III, §3.3, §4.3)

| Paper element | Implemented by |
|---|---|
| Six conditions = 2 overlay states × 3 label states, baselined on `without_overlay` | [`conditioning.py`](Stage_III/medgemma_utils/conditioning.py) + static prompts in [`prompts.json`](Stage_III/dataset) |
| Red disc overlay composited onto the fundus image | `OVERLAY_RED_ALPHA` in `conditioning.py` |
| Structured describe-before-diagnose prompt (Fig. 3) | [`prompts.json`](Stage_III/dataset) |
| MedGemma as a frozen black box, deterministic greedy decoding | [`medgemma_conditioner.py`](Stage_III/medgemma_utils/medgemma_conditioner.py) (`do_sample=False`) |
| *pipeline* vs *dataset* evaluation sources (real classifier + segmentor vs ground-truth control) | [`pipeline_inputs.py`](Stage_III/medgemma_utils/pipeline_inputs.py) and [`oracle_inputs.py`](Stage_III/medgemma_utils/oracle_inputs.py) |
| BERTScore F1, Sentence-BERT, and their calibrated variants | [`evaluation.py`](Stage_III/medgemma_utils/evaluation.py) |
| Paired Wilcoxon vs baseline, effect size, Holm correction | `Evaluator` in `evaluation.py` |
| Recall@1 and the negation-aware diagnostic probe | [`analyze_results.py`](Stage_III/scripts/analyze_results.py) |
| LoRA adaptation (rank 8, α 16, all linear projections, lr 1e-4, 3 epochs, 5 seeds) | [`train_experimental_medgemma_lora.py`](Stage_III/scripts/train_experimental_medgemma_lora.py) |
| Expert Likert review export | [`export_likert_review.py`](Stage_III/scripts/export_likert_review.py) |
| **Tables 6–7** — similarity floor and pretrained vs LoRA | [`Stage_III/results/`](Stage_III/results) |

---

## 3. Repository layout

```
.
├── Stage_I/                              Classification & routing (Tables 2–4)
│   ├── modules/
│   │   └── gradcam_module.py             Dataset-agnostic Grad-CAM extractor
│   ├── Experiments/
│   │   ├── Classifier_Selection/         Backbone selection on REFUGE (Table 2)
│   │   │   ├── modules/                  backbones · cnn_classifier · prototype_classifier · data_module
│   │   │   ├── scripts/                  convert_refuge_format · few_shot · evaluate · extract_gradcam · benchmark_inference
│   │   │   ├── data/                     annotations.json + splits.json (REFUGE, 1200 images)
│   │   │   ├── results/                  selection_summary_*.json
│   │   │   ├── config.yaml               Stage I configuration
│   │   │   ├── run_experiment.py         CLI orchestrator
│   │   │   └── notebook/                 Colab orchestrator
│   │   └── gradcam_resnet18_refuge/      Grad-CAM localization study (IoU / SSIM / pointing)
│   └── Classifier/                       Transfer to MM-ODIR-129
│       ├── private_data_module.py        DataModule for the MM-ODIR-129 format
│       ├── meta_nshot_experiment.ipynb   Support-size sweep (Table 3)
│       └── Inference/                    Deployed-classifier inference + CIs (Table 4)
│
├── Stage_III/                            Conditioned MedGemma report generation (Tables 6–7)
│   ├── medgemma_utils/                   conditioning · dataset · evaluation · inputs · conditioner
│   ├── scripts/                          run_conditioned_medgemma · train_experimental_medgemma_lora · analyze_results · export_likert_review
│   ├── dataset/                          MM-ODIR-129 images, masks, prompts, pipeline classifications
│   ├── results/                          Generations, per-condition summaries, post-hoc analysis
│   ├── slurm/                            Cluster job scripts
│   ├── config.yaml                       Stage III configuration
│   └── README.md                         Full Stage III documentation
│
├── Glaucoma_labels.pdf                   Annotation schema for the seven glaucoma signs
└── AGENTS.md                             Code conventions
```

---

## 4. Stage I implementation

### 4.1 Data

**REFUGE** — [`data_module.py`](Stage_I/Experiments/Classifier_Selection/modules/data_module.py). `convert_refuge_format.py` turns the raw dataset into the project schema (`annotations.json` + `splits.json`, 1200 images, 400 per split, 40 glaucoma + 360 normal each). Two details matter for correctness: REFUGE's `index.json` is a dict keyed `"0".."399"`, not a list, and the class label must be read from the `Label` field rather than the filename prefix, because only the train split encodes the class in the name (`g####`/`n####`; val uses `V####`, test `T####` and carries no label). Three-class masks (0 = background, 1 = rim, 2 = cup) are flattened to a binary optic disc with `mask > 0`.

**MM-ODIR-129** — [`private_data_module.py`](Stage_I/Classifier/private_data_module.py). Exposes exactly the same public interface as the REFUGE module, so `train_few_shot`, `PrototypeClassifier`, `evaluate_classification` and the Grad-CAM evaluation run unchanged on either dataset. It absorbs the format differences: a `split.json` holding full records per split (its `validation` key is normalized to `val`), `Pathological`/`Normal` labels mapped to `glaucoma`/`normal`, flat image directories, and masks stored as `.npy` pairs (`<stem>_disc.npy`, `<stem>_cup.npy`) whose union `disc | cup` is the ground-truth optic disc. Images without a disc mask are excluded from the IoU computation — the MM-ODIR-129 masks exist only for the 69 glaucomatous eyes.

Both modules load splits from disk and never regenerate them, which is what keeps the paper's patient-level stratification intact.

### 4.2 Backbones

[`backbones.py`](Stage_I/Experiments/Classifier_Selection/modules/backbones.py) is the single source of truth for what a backbone is. It exposes two levels: the raw torchvision model with its parts (Grad-CAM target layer, last block to unfreeze, feature dimension, head attribute), and a frozen feature extractor with an `Identity` head returning `(B, D)` embeddings for the prototype path.

| Backbone | Embedding dim | Grad-CAM target | Last block (fine-tuning) |
|---|---|---|---|
| `resnet18` | 512 | `model.layer4` | `model.layer4` |
| `efficientnet_b0` | 1280 | `model.features` | `model.features[-1]` |
| `densenet121` | 1024 | `model.features` | `model.features.denseblock4` |

All three come from torchvision rather than `timm`, so the documented target layers hold and the dependency surface stays small.

### 4.3 The two configurations compared in Table 2

[`prototype_classifier.py`](Stage_I/Experiments/Classifier_Selection/modules/prototype_classifier.py) implements both, on top of EasyFSL's `PrototypicalNetworks`:

- **`euclidean`** (the paper's *frozen*) — the backbone never trains. `fit(support_loader)` only computes the two class prototypes as mean support embeddings; a query is assigned to the nearest prototype under Euclidean distance. No backprop anywhere.
- **`meta`** (the paper's *episodic fine-tuning*) — `meta_train(data_module, n_way=2, n_shot=N, n_query=10, n_episodes=100, lr=1e-3, augment=True)` optimizes the backbone over episodes that simulate the deployment task, then the deployment prototypes are built from a balanced N-shot support set.

Two implementation points worth knowing when reading the code:

*Score convention.* `PrototypicalNetworks(use_softmax=False)` returns the **negative** Euclidean distance to each prototype, so `argmax` still selects the nearest prototype and the shared evaluation code needs no special case. `predict()` applies its own softmax to produce the probability distribution — this is the `P(glaucoma)` that `inference_on_test.py` reports and that Stage III consumes.

*Class ordering.* EasyFSL orders prototypes by ascending label, so prototype 0 = normal, prototype 1 = glaucoma, matching `CLASS_TO_IDX` in both DataModules. Getting this wrong silently inverts every metric, so it is asserted by construction rather than by convention.

*Grad-CAM with a frozen backbone.* When no parameter requires grad, activations carry no gradient and Grad-CAM yields nothing. The module therefore routes the gradient through the **image** (`batch.requires_grad_(True)`) using a forward-hook + `tensor.register_hook` pattern, which works for all three backbones — including DenseNet, whose in-place ReLU breaks naive backward-hook implementations.

`PrototypeClassifier` deliberately mirrors `CNNClassifier`'s interface (`.model`, `.class_names`, `.device`, `.target_layer`, `predict`, `get_gradcam`, `save`, `load`), with `fit()` replacing `train()`. That drop-in property is what lets the same orchestrator, evaluator and Grad-CAM code serve both methods.

### 4.4 Fine-tuning baseline (not in the paper)

[`cnn_classifier.py`](Stage_I/Experiments/Classifier_Selection/modules/cnn_classifier.py) implements a conventional supervised baseline: trainable head, everything frozen except the last conv block, cross-entropy, early stopping, `ReduceLROnPlateau`. It is exercised by the `finetune` method, whose sweeps are recorded in `results/selection_summary_finetune(*).json`. The paper's Table 2 covers only the two prototype configurations.

### 4.5 Grad-CAM

The Grad-CAM/optic-disc overlap in Table 2 is computed by `evaluate_gradcam_quality()` in [`run_experiment.py`](Stage_I/Experiments/Classifier_Selection/run_experiment.py): heatmap for the glaucoma class → threshold at the 95th percentile → IoU against the ground-truth disc mask, plus pointing accuracy (does the hottest pixel fall inside the disc?). Note that this path uses each classifier's **own** `get_gradcam`, not the shared module, for backbone safety.

[`Stage_I/modules/gradcam_module.py`](Stage_I/modules/gradcam_module.py) is the standalone, dataset-agnostic extractor: model + image in, heatmap out. It does not train, load data, or compute segmentation metrics.

```python
from modules import GradCAMConfig, GradCAMExtractorModule

extractor = GradCAMExtractorModule(model, GradCAMConfig(target_layer="layer4[-1]", percentile=95))
heatmap = extractor.extract(image)          # ndarray (H, W) in [0, 1]
binary  = extractor.extract_binary(image)   # ndarray (H, W) in {0, 1}
```

It follows a Strategy pattern (`BaseCAMExtractor` → `GradCAMExtractorModule`) so Grad-CAM++, Score-CAM or Eigen-CAM can be added without touching existing code; `LayerResolver` resolves layer specs such as `"layer4[-1]"` or `"features.denseblock4"` and can suggest a target layer automatically; and either a manual hook implementation or `pytorch-grad-cam` can serve as backend.

[`Stage_I/Experiments/gradcam_resnet18_refuge/`](Stage_I/Experiments/gradcam_resnet18_refuge) is the standalone localization study behind that metric: ResNet-18 trained in few-shot regime on REFUGE, evaluated against ground-truth disc masks with IoU, SSIM (manual implementation, Gaussian window 11×11) and pointing accuracy, with 2×2 panels per image (original · heatmap · GT mask · GT-vs-contour overlay). Its extractor also implements fundus auto-crop for datasets that are not pre-cropped.

### 4.6 Deployed-classifier evaluation

[`Stage_I/Classifier/Inference/`](Stage_I/Classifier/Inference) produces Table 4. `inference_on_test.py` loads the selected checkpoint, runs `predict` over the test split and returns `y_true`, `P(glaucoma)`, `y_pred` (threshold 0.5) and per-image ids. `metrics_generation.py` computes point estimates plus two interval types appropriate for n = 26: **stratified bootstrap** (5000 resamples drawn within each class, preserving 12 normal / 14 glaucoma) and **Clopper–Pearson** exact intervals for the proportion-based metrics. The bootstrap is seeded; Clopper–Pearson is deterministic.

---

## 5. Stage III implementation

[`Stage_III/README.md`](Stage_III/README.md) documents this stage in full; what follows is the shape of it.

MedGemma (`google/medgemma-1.5-4b-it`) is used as a frozen black box with deterministic generation (`do_sample=False`, up to 512 new tokens), so every difference between runs comes from the conditioning rather than from sampling noise.

### 5.1 The six conditions

Each image is described under six conditions, built from two axes — whether the optic-disc mask is composited into the image in red, and whether the prompt states a class:

| Condition | Overlay | Prompt |
|---|---|---|
| `with_overlay_glaucoma` | yes | glaucoma |
| `without_overlay_glaucoma` | no | glaucoma |
| `with_overlay_normal` | yes | normal |
| `without_overlay_normal` | no | normal |
| `with_overlay` | yes | unspecified |
| `without_overlay` | no | unspecified |

`without_overlay` — raw image, no class hint — is the fixed baseline for every paired comparison. The prompts themselves are static and live in `dataset/prompts.json`.

### 5.2 Two evaluation sources

The same six conditions run over two input sources, which is what separates a generator failure from an upstream one:

- **dataset** — the oracle control: labels are ground truth and overlays are rendered from the expert disc masks.
- **pipeline** — the deployed setting: the label comes from the Stage I classifier and the overlay from the SAM-based segmentor. Both the prediction and the ground truth it is judged against come from `dataset/pipeline_classifications.json`, so provenance stays traceable.

Crossing model (base vs LoRA) with source gives the paper's 2 × 2 study. The base model is deterministic, so one run per source suffices; LoRA is repeated over the same five seeds and reported as mean ± std.

### 5.3 Evaluation

[`evaluation.py`](Stage_III/medgemma_utils/evaluation.py) scores every generation against its own expert reference with BERTScore F1 (BiomedBERT) and Sentence-BERT cosine (Bioclinical ModernBERT), plus **calibrated** variants: the score against the true reference minus the mean score against shuffled references. That subtraction is the point — glaucoma reports share so much boilerplate that raw similarity stays high even when the diagnosis is wrong, and the calibrated score is what reveals it. Each condition is compared with the baseline by paired Wilcoxon over shared images, with effect size and Holm correction.

[`analyze_results.py`](Stage_III/scripts/analyze_results.py) adds a CPU-only post-hoc pass — Recall@1 and a negation-aware class-level diagnostic probe — which surfaces the class collapse that the similarity metrics hide.

### 5.4 LoRA variant

A standard LoRA (`target_modules="all-linear"`, rank 8, α 16, lr 1e-4, 3 epochs) trained **only on the train split against raw image + generic prompt → expert transcription**. It never sees masks, classes or conditioning prompts, so any gain it shows is report style rather than diagnosis — which is exactly the paper's finding: similarity metrics rise while Recall@1 stays at chance.

---

## 6. Reproducing the paper's tables

### Table 2 — backbone selection on REFUGE

```bash
python Stage_I/Experiments/Classifier_Selection/run_experiment.py --data-dir /path/to/REFUGE
```

Expects `<data-dir>/{train,val,test}/{Images,Masks,index.json}`. The first run converts REFUGE into `annotations.json` + `splits.json`; pass `--skip-convert` to reuse them. The run sweeps the methods listed in `config.yaml` (`methods: [euclidean, meta]`) × 3 backbones × N ∈ {1..5} × 5 seeds, benchmarks VRAM once per architecture, and writes `results/selection_summary.json`.

To match the paper exactly, keep `data.image_size: [224, 224]` as shipped in [`config.yaml`](Stage_I/Experiments/Classifier_Selection/config.yaml).

### Table 3 — support-size selection on MM-ODIR-129

Run [`meta_nshot_experiment.ipynb`](Stage_I/Classifier/meta_nshot_experiment.ipynb), setting `DATASET_DIR`, `SPLIT_JSON` and `RESULTS_DIR`. It sweeps `N_SHOT ∈ {3,6,9,12,15,21}` for EfficientNet-B0 + meta over five seeds, evaluating macro-F1 on the full validation split (non-episodic evaluation, matching the REFUGE protocol) and Grad-CAM IoU against `disc ∪ cup`.

### Table 4 — deployed classifier on the test split

```bash
python Stage_I/Classifier/Inference/inference_on_test.py --pth /path/to/model_N6.pth --dataset-dir /path/to/images --split-json /path/to/split.json
```

Or run [`inference_metrics_orchestrator.ipynb`](Stage_I/Classifier/Inference/inference_metrics_orchestrator.ipynb), which additionally writes `test_metrics.json` with the per-image predictions.

### Tables 6–7 — conditioned report generation

From inside `Stage_III/`:

```bash
python -m scripts.run_conditioned_medgemma --config config.yaml --split-file dataset/split.json --split-name test --output results/medgemma_base_dataset.json
```

```bash
python -m scripts.run_conditioned_medgemma --config config.yaml --pipeline-dir dataset/test --pipeline-classifications-json dataset/pipeline_classifications.json --output results/medgemma_base_pipeline.json
```

```bash
python -m scripts.train_experimental_medgemma_lora --split-file dataset/split.json --lora-rank 8 --lora-alpha 16 --learning-rate 1e-4 --num-epochs 3 --seed 42 --output-dir checkpoints/lora/seed42
```

```bash
python -m scripts.analyze_results --base results/medgemma_base_dataset.json --lora-glob "results/medgemma_lora_dataset_seed*.json" --output results/analysis_dataset.json
```

SLURM job scripts for the five-seed sweep are in [`Stage_III/slurm/`](Stage_III/slurm); they hardcode cluster paths and are not portable as-is.

### Grad-CAM localization study

```bash
python Stage_I/Experiments/gradcam_resnet18_refuge/scripts/train_fewshot.py --config config.yaml --data-dir /path/to/REFUGE
```

```bash
python Stage_I/Experiments/gradcam_resnet18_refuge/scripts/evaluate_fewshot.py --config config.yaml --data-dir /path/to/REFUGE --n-viz 10
```

### Module tests

```bash
python Stage_I/Experiments/gradcam_resnet18_refuge/scripts/test_gradcam_module.py
```

---

## 7. Results as recorded in this repository

### Table 2 — REFUGE backbone selection (224×224)

F1 is the macro-F1 averaged over support sizes N ∈ {1..5}; IoU is Grad-CAM/optic-disc overlap; VRAM is peak inference memory at batch size 1. Mean ± std over the five seeds.

| Backbone | Config | macro-F1 | IoU | VRAM (MB) |
|---|---|---|---|---|
| ResNet-18 | frozen (`euclidean`) | 0.448 ± 0.127 | 0.013 ± 0.005 | 72.9 |
| ResNet-18 | meta | 0.536 ± 0.066 | 0.037 ± 0.017 | 72.9 |
| DenseNet-121 | frozen (`euclidean`) | 0.398 ± 0.172 | 0.016 ± 0.001 | 77.3 |
| DenseNet-121 | meta | 0.572 ± 0.062 | 0.056 ± 0.059 | 77.3 |
| EfficientNet-B0 | frozen (`euclidean`) | 0.537 ± 0.054 | 0.003 ± 0.001 | 65.8 |
| **EfficientNet-B0** | **meta** | **0.602 ± 0.043** | **0.036 ± 0.024** | **65.8** |

Episodic meta-training beats the frozen encoder for every backbone. EfficientNet-B0 + meta has the highest macro-F1, the third-best IoU and the lowest memory footprint, and is the configuration carried forward.

The code additionally reduces these three criteria to a single scalar, `score = 0.40·F1 + 0.40·IoU + 0.20·(1 − VRAM_norm)` (`aggregate_and_save()` in `run_experiment.py`), which picks the same winner. The paper reports the three criteria directly instead.

### Table 3 — support-size selection on MM-ODIR-129

N = 6 gives the best validation macro-F1 (0.846 ± 0.024) and is deployed. Produced by the notebook; the summary JSON is not versioned here.

### Table 4 — deployed classifier on the test split

EfficientNet-B0, meta-trained, N = 6; test split n = 26 (14 glaucoma, 12 normal); threshold 0.5. From [`test_metrics.json`](Stage_I/Classifier/Inference/test_metrics.json).

| Metric | Estimate | Bootstrap 95 % CI | Clopper–Pearson 95 % CI |
|---|---|---|---|
| AUC | 0.833 | [0.637, 0.994] | — |
| Accuracy | 0.808 | [0.654, 0.923] | [0.606, 0.934] |
| Sensitivity | 0.929 | [0.786, 1.000] | [0.661, 0.998] |
| Specificity | 0.667 | [0.417, 0.917] | [0.349, 0.901] |
| F1 | 0.839 | [0.727, 0.933] | — |
| PPV | 0.765 | [0.632, 0.933] | [0.501, 0.932] |
| NPV | 0.889 | [0.700, 1.000] | [0.518, 0.997] |

Confusion matrix: 8 TN, 4 FP, 1 FN, 13 TP. The classifier is markedly more sensitive than specific, which is the behaviour the paper's Stage-I routing analysis discusses: a false negative sends a glaucomatous eye down the normal path, bypassing segmentation and handing MedGemma an incorrect label.

The checkpoint recorded in that file is `efficientnet_b0/meta/seed_1024/model_N6.pth`.

### Table 7 — pretrained vs LoRA MedGemma on the baseline condition

`without_overlay` is the only setting where the class is withheld, so it is the only valid diagnostic probe. Recall@1 chance is 0.038.

| Model | sBERT | sBERT_cal | ROUGE-L | BLEU | Recall@1 | Bal. Acc. | Sens. (g / n) |
|---|---|---|---|---|---|---|---|
| Pretrained | 0.497 | +0.00 | 0.077 | 0.006 | ≈0.08 | 0.506 | 0.93 / 0.08 |
| LoRA | 0.779 | +0.09 | 0.417 | 0.231 | ≈0.06 | 0.735 | 0.99 / 0.48 |

LoRA lifts every surface metric and makes reports gradeable, but Recall@1 stays at chance and the calibrated gain remains small: the adapter learned the distribution of expert reports, not descriptions tied to the specific eye. The cross-validation floor in Table 6 is why those raw numbers cannot be read at face value — expert reports of *opposite* diagnosis already score BERTScore F1 = 0.935 and Sentence-BERT = 0.670 against each other.

Raw generations, per-condition summaries, paired comparisons and the post-hoc analysis are in [`Stage_III/results/`](Stage_III/results).

---

## 8. Reproducibility

- Global seed `SEED = 42` in [`Stage_I/Experiments/Classifier_Selection/config.yaml`](Stage_I/Experiments/Classifier_Selection/config.yaml); every module receives `seed` in its config and calls `set_global_seed()` on construction.
- Five fixed seeds `{42, 123, 456, 789, 1024}` for all multi-seed results in both stages, reported as mean ± std.
- **Within iteration `i`, every backbone and method uses `seeds[i]`**, so all configurations see exactly the same support set — this is what makes the Table 2 comparison fair, and it is enforced by the loop order in `run_experiment.py`, not left to chance.
- Splits are always loaded from `splits.json` / `split.json` and never regenerated.
- `torch.backends.cudnn.deterministic = True`; DataLoader workers are seeded through `worker_init_fn`.
- The bootstrap in `metrics_generation.py` is seeded; Clopper–Pearson is exact and deterministic.
- MedGemma generation is greedy (`do_sample=False`), so the base model needs only one run per source.

---

## 9. Installation

The two stages have independent environments; install whichever you need.

**Stage I**

```bash
git clone https://github.com/TheBug95/Medgemma_Segmentation.git
cd Medgemma_Segmentation
pip install torch torchvision easyfsl numpy scipy scikit-learn pyyaml pillow opencv-python matplotlib tqdm
```

Python 3.10+, PyTorch 2.x, and a GPU with ≥16 GB VRAM for the full sweeps (experiments were run on Colab Pro / GCP with L4 or T4). `grad-cam` is optional — `gradcam_module.py` uses it when `use_external: true` and falls back to its manual hook implementation otherwise. `easyfsl` is required for the prototype classifier.

**Stage III**

```bash
pip install -r Stage_III/requirements.txt
```

Requires a CUDA 12.8 GPU (`torch==2.11.0+cu128`); both generation and LoRA training raise at startup without a CUDA device. `google/medgemma-1.5-4b-it` is gated on Hugging Face, so a valid `HF_TOKEN` or a cached `huggingface-cli login` is needed.

### Notebooks

| Notebook | Purpose |
|---|---|
| [`experiment_orchestrator.ipynb`](Stage_I/Experiments/Classifier_Selection/notebook/experiment_orchestrator.ipynb) | Backbone selection (Table 2) on Colab |
| [`CAM_generation.ipynb`](Stage_I/Experiments/gradcam_resnet18_refuge/CAM_generation.ipynb) | Grad-CAM generation and visualization |
| [`meta_nshot_experiment.ipynb`](Stage_I/Classifier/meta_nshot_experiment.ipynb) | Support-size sweep (Table 3) |
| [`inference_metrics_orchestrator.ipynb`](Stage_I/Classifier/Inference/inference_metrics_orchestrator.ipynb) | Deployed-classifier inference and CIs (Table 4) |

---

## 10. Datasets

**REFUGE** — public. 1200 fundus images, 400 per split, 40 glaucoma + 360 normal in each; the test split is unlabelled. Masks are PNGs with 0 = background, 1 = rim, 2 = cup. Used only for backbone selection, and not versioned here.

**MM-ODIR-129** — [public on Hugging Face](https://huggingface.co/datasets/TheBug95/MM-ODIR-129). 129 fundus images from 76 patients curated and re-annotated from ODIR-5K: 60 normal, 69 glaucomatous. Five board-certified ophthalmologists annotated binary diagnosis, ordinal scoring of seven glaucoma signs (cup-to-disc ratio, neuroretinal rim, disc hemorrhage, peripapillary atrophy, RNFL defect, disc pallor, vessel changes) and free-text clinical transcriptions; optic disc and cup masks were traced in ImageJ for the 69 glaucomatous images. Splits are patient-level — both eyes of a patient stay in one partition.

| Split | Total | Glaucoma | Normal |
|---|---:|---:|---:|
| Train | 77 | 41 | 36 |
| Validation | 26 | 14 | 12 |
| Test | 26 | 14 | 12 |

A flat copy of MM-ODIR-129 ships in [`Stage_III/dataset/`](Stage_III/dataset) so the report-generation experiments run without external downloads: images, `*_disc` / `*_cup` masks, `annotations.json`, `split.json`, `prompts.json`, the Stage I predictions in `pipeline_classifications.json`, and the pipeline-style test set with its segmentor masks and pre-rendered overlays. The grading scheme behind the ordinal annotations is documented in [`Glaucoma_labels.pdf`](Glaucoma_labels.pdf).

---

## 11. Known gaps

- Stage II (SAM 2.1 + LoRA segmentation, Table 5) is not versioned here; only its outputs for the test split, shipped as masks and overlays under `Stage_III/dataset/test/`.
- Table 3's per-seed sweep results are written to Drive by the notebook and are not versioned.
- The `Stage_III/slurm/` scripts hardcode cluster-specific paths and GPU constraints; they need editing before running elsewhere.
- In the paper's Table 2, the IoU standard deviations for DenseNet-121/meta and EfficientNet-B0/meta appear transposed relative to the recorded results (`0.056 ± 0.024` and `0.036 ± 0.058` in print, against `0.056 ± 0.059` and `0.036 ± 0.024` in `selection_summary_few_shot(224x224).json`). Means and all other entries match exactly.
