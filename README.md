# Multimodal MedGemma — Code for the Classification Stage

Reference implementation accompanying:

> **MedGemma for Ophthalmology: Integrating SAM-Segmented Retinal Images and Conditioned Descriptions for Multimodal Medical Report Generation**
> M. Abreu-Cárdenas, D. Xie-Li, D. Alexander-Vargas, J.P. Pérez-Delgado, S. Cabrera-Tabash, S. Calderon-Ramirez, M. Masis-Solano, A. Garcia-Pedrero, K. Dueñas Ángeles, G.A. Osorio Navarro.
> Instituto Tecnológico de Costa Rica · Cornell Tech · Instituto Conde de Valencia · Universidad Politécnica de Madrid.

The paper proposes a three-stage pipeline that conditions MedGemma on automatically generated optic-disc segmentations and predicted disease labels for structured glaucoma report generation. **This repository holds the code for Stage I — the CNN classifier that routes each image and supplies the label used downstream — together with the backbone-selection study and the evaluation code behind Tables 2, 3 and 4 of the paper.** The Stage II (SAM 2.1 + LoRA) and Stage III (MedGemma) training and inference scripts are not versioned here.

Dataset: [MM-ODIR-129](https://huggingface.co/datasets/TheBug95/MM-ODIR-129) on Hugging Face.

---

## 1. The pipeline, and where this code sits

| Stage | What the paper does | Code here |
|---|---|---|
| **I — Disease classification & routing** | A lightweight CNN predicts `glaucoma` / `normal`. Glaucoma cases are routed to segmentation; normal cases bypass it. The predicted label is also passed to MedGemma as prior clinical context. | ✅ **This repository** |
| **II — Conditional image preparation** | SAM 2.1 adapted with LoRA predicts a binary optic-disc mask, composited as a semi-transparent overlay `x_overlay`. | ❌ Not versioned here |
| **III — Report generation** | MedGemma 1.5 4B receives the (optionally overlaid) image plus a prompt carrying the predicted label; six conditioning configurations are compared. | ❌ Not versioned here |

Stage I is the part of the pipeline under limited supervision: the design question is which backbone and which few-shot regime give a usable classifier from 77 training images, which is what the code in this repository answers.

---

## 2. Paper → code map

Every claim in the paper's §4.1 traces to a specific file here.

| Paper element | Implemented by |
|---|---|
| Prototypical Networks, `p_c = (1/\|S_c\|) Σ f_θ(x_i)`, `ŷ = arg min_c ‖f_θ(x_q) − p_c‖²` | [`prototype_classifier.py`](Experiments/Classifier_Selection/modules/prototype_classifier.py) (EasyFSL `PrototypicalNetworks`) |
| *frozen encoder* setting (ImageNet backbone fixed, nearest-prototype inference) | `PrototypeClassifier` with method `euclidean` — the code's name for the paper's `frozen` |
| *episodic fine-tuning* setting (100 2-way episodes, N support + 10 query per class, cross-entropy, Adam lr 1e-3, augmentation) | `PrototypeClassifier.meta_train()`, parameterized in [`config.yaml`](Experiments/Classifier_Selection/config.yaml) under `meta_training` |
| Three ImageNet-pretrained backbones (ResNet-18, DenseNet-121, EfficientNet-B0) | [`backbones.py`](Experiments/Classifier_Selection/modules/backbones.py) |
| Class-balanced support set sampled from REFUGE train, N ∈ {1..5} | `build_support_indices()` in [`few_shot.py`](Experiments/Classifier_Selection/scripts/few_shot.py) |
| Five fixed seeds (42, 123, 456, 789, 1024) with **identical support sets shared across all backbones within each seed** | seed loop in [`run_experiment.py`](Experiments/Classifier_Selection/run_experiment.py) |
| macro-F1 on the REFUGE validation split | [`evaluate.py`](Experiments/Classifier_Selection/scripts/evaluate.py) |
| Grad-CAM localization (IoU with the optic-disc mask) | `evaluate_gradcam_quality()` in `run_experiment.py`, [`extract_gradcam.py`](Experiments/Classifier_Selection/scripts/extract_gradcam.py), [`gradcam_module.py`](modules/gradcam_module.py) |
| Peak inference memory at batch size 1 | [`benchmark_inference.py`](Experiments/Classifier_Selection/scripts/benchmark_inference.py) |
| REFUGE, 1200 annotated fundus images | [`convert_refuge_format.py`](Experiments/Classifier_Selection/scripts/convert_refuge_format.py) + [`data/`](Experiments/Classifier_Selection/data) |
| **Table 2** — backbone selection on REFUGE | [`selection_summary_few_shot(224x224).json`](Experiments/Classifier_Selection/results/selection_summary_few_shot%28224x224%29.json) |
| MM-ODIR-129 loading (patient-level splits, disc/cup masks, `Pathological`/`Normal` labels) | [`private_data_module.py`](Classifier/private_data_module.py) |
| Support-size sweep N ∈ {3, 6, …, 21} on MM-ODIR-129 | [`meta_nshot_experiment.ipynb`](Classifier/meta_nshot_experiment.ipynb) |
| **Table 3** — support-size selection, N = 6 chosen | produced by that notebook (results written to Drive, not versioned) |
| Deployed classifier on the 26-image test split | [`inference_on_test.py`](Classifier/Inference/inference_on_test.py) |
| Bootstrap and Clopper–Pearson 95 % intervals | [`metrics_generation.py`](Classifier/Inference/metrics_generation.py) |
| **Table 4** — deployed classifier performance | [`test_metrics.json`](Classifier/Inference/test_metrics.json) |
| Per-image label and glaucoma probability consumed by Stage III | `per_image` array in `test_metrics.json` |

---

## 3. Repository layout

```
.
├── modules/
│   └── gradcam_module.py            Dataset-agnostic Grad-CAM extractor
│
├── Experiments/
│   ├── Classifier_Selection/        §4.1 — backbone selection on REFUGE (Table 2)
│   │   ├── modules/                 backbones · cnn_classifier · prototype_classifier · data_module
│   │   ├── scripts/                 convert_refuge_format · few_shot · evaluate · extract_gradcam · benchmark_inference
│   │   ├── data/                    annotations.json + splits.json (REFUGE, 1200 images)
│   │   ├── results/                 selection_summary_*.json
│   │   ├── run_experiment.py        CLI orchestrator
│   │   └── notebook/                Colab orchestrator
│   │
│   └── gradcam_resnet18_refuge/     Grad-CAM localization study (IoU / SSIM / pointing)
│
├── Classifier/                      Transfer to MM-ODIR-129
│   ├── private_data_module.py       DataModule for the MM-ODIR-129 format
│   ├── meta_nshot_experiment.ipynb  Support-size sweep (Table 3)
│   └── Inference/                   Deployed-classifier inference + CIs (Table 4)
│
├── config.yaml                      Full-pipeline configuration (includes unimplemented stages)
├── experimental_design.md           Earlier design document (see §9)
├── pipeline_mejorado_informe.md     Earlier design document (see §9)
└── AGENTS.md                        Code conventions
```

---

## 4. Stage I implementation

### 4.1 Data

**REFUGE** — [`data_module.py`](Experiments/Classifier_Selection/modules/data_module.py). `convert_refuge_format.py` turns the raw dataset into the project schema (`annotations.json` + `splits.json`, 1200 images, 400 per split, 40 glaucoma + 360 normal each). Two details matter for correctness: REFUGE's `index.json` is a dict keyed `"0".."399"`, not a list, and the class label must be read from the `Label` field rather than the filename prefix, because only the train split encodes the class in the name (`g####`/`n####`; val uses `V####`, test `T####` and carries no label). Three-class masks (0 = background, 1 = rim, 2 = cup) are flattened to a binary optic disc with `mask > 0`.

**MM-ODIR-129** — [`private_data_module.py`](Classifier/private_data_module.py). Exposes exactly the same public interface as the REFUGE module, so `train_few_shot`, `PrototypeClassifier`, `evaluate_classification` and the Grad-CAM evaluation run unchanged on either dataset. It absorbs the format differences: a `split.json` holding full records per split (its `validation` key is normalized to `val`), `Pathological`/`Normal` labels mapped to `glaucoma`/`normal`, flat image directories, and masks stored as `.npy` pairs (`<stem>_disc.npy`, `<stem>_cup.npy`) whose union `disc | cup` is the ground-truth optic disc. Images without a disc mask are excluded from the IoU computation — the paper's MM-ODIR-129 masks exist only for the 69 glaucomatous eyes.

Both modules load splits from disk and never regenerate them, which is what keeps the paper's patient-level stratification intact.

### 4.2 Backbones

[`backbones.py`](Experiments/Classifier_Selection/modules/backbones.py) is the single source of truth for what a backbone is. It exposes two levels: the raw torchvision model with its parts (Grad-CAM target layer, last block to unfreeze, feature dimension, head attribute), and a frozen feature extractor with an `Identity` head returning `(B, D)` embeddings for the prototype path.

| Backbone | Embedding dim | Grad-CAM target | Last block (fine-tuning) |
|---|---|---|---|
| `resnet18` | 512 | `model.layer4` | `model.layer4` |
| `efficientnet_b0` | 1280 | `model.features` | `model.features[-1]` |
| `densenet121` | 1024 | `model.features` | `model.features.denseblock4` |

All three come from torchvision rather than `timm`, so the documented target layers hold and the dependency surface stays small.

### 4.3 The two configurations compared in Table 2

[`prototype_classifier.py`](Experiments/Classifier_Selection/modules/prototype_classifier.py) implements both, on top of EasyFSL's `PrototypicalNetworks`:

- **`euclidean`** (the paper's *frozen*) — the backbone never trains. `fit(support_loader)` only computes the two class prototypes as mean support embeddings; a query is assigned to the nearest prototype under Euclidean distance. No backprop anywhere.
- **`meta`** (the paper's *episodic fine-tuning*) — `meta_train(data_module, n_way=2, n_shot=N, n_query=10, n_episodes=100, lr=1e-3, augment=True)` optimizes the backbone over episodes that simulate the deployment task, then the deployment prototypes are built from a balanced N-shot support set.

Two implementation points worth knowing when reading the code:

*Score convention.* `PrototypicalNetworks(use_softmax=False)` returns the **negative** Euclidean distance to each prototype, so `argmax` still selects the nearest prototype and the shared evaluation code needs no special case. `predict()` applies its own softmax to produce the probability distribution — this is the `P(glaucoma)` that `inference_on_test.py` reports and that Stage III consumes.

*Class ordering.* EasyFSL orders prototypes by ascending label, so prototype 0 = normal, prototype 1 = glaucoma, matching `CLASS_TO_IDX` in both DataModules. Getting this wrong silently inverts every metric, so it is asserted by construction rather than by convention.

*Grad-CAM with a frozen backbone.* When no parameter requires grad, activations carry no gradient and Grad-CAM yields nothing. The module therefore routes the gradient through the **image** (`batch.requires_grad_(True)`) using a forward-hook + `tensor.register_hook` pattern, which works for all three backbones — including DenseNet, whose in-place ReLU breaks naive backward-hook implementations.

`PrototypeClassifier` deliberately mirrors `CNNClassifier`'s interface (`.model`, `.class_names`, `.device`, `.target_layer`, `predict`, `get_gradcam`, `save`, `load`), with `fit()` replacing `train()`. That drop-in property is what lets the same orchestrator, evaluator and Grad-CAM code serve both methods.

### 4.4 Fine-tuning baseline (not in the paper)

[`cnn_classifier.py`](Experiments/Classifier_Selection/modules/cnn_classifier.py) implements a conventional supervised baseline: trainable head, everything frozen except the last conv block, cross-entropy, early stopping, `ReduceLROnPlateau`. It is exercised by the `finetune` method and reported in §7 below. The paper's Table 2 covers only the two prototype configurations.

### 4.5 Grad-CAM

The Grad-CAM/optic-disc overlap in Table 2 is computed by `evaluate_gradcam_quality()` in [`run_experiment.py`](Experiments/Classifier_Selection/run_experiment.py): heatmap for the glaucoma class → threshold at the 95th percentile → IoU against the ground-truth disc mask, plus pointing accuracy (does the hottest pixel fall inside the disc?). Note that this path uses each classifier's **own** `get_gradcam`, not the shared module, for backbone safety.

[`modules/gradcam_module.py`](modules/gradcam_module.py) is the standalone, dataset-agnostic extractor: model + image in, heatmap out. It does not train, load data, or compute segmentation metrics.

```python
from modules import GradCAMConfig, GradCAMExtractorModule

extractor = GradCAMExtractorModule(model, GradCAMConfig(target_layer="layer4[-1]", percentile=95))
heatmap = extractor.extract(image)          # ndarray (H, W) in [0, 1]
binary  = extractor.extract_binary(image)   # ndarray (H, W) in {0, 1}
```

It follows a Strategy pattern (`BaseCAMExtractor` → `GradCAMExtractorModule`) so Grad-CAM++, Score-CAM or Eigen-CAM can be added without touching existing code; `LayerResolver` resolves layer specs such as `"layer4[-1]"` or `"features.denseblock4"` and can suggest a target layer automatically; and either a manual hook implementation or `pytorch-grad-cam` can serve as backend.

[`Experiments/gradcam_resnet18_refuge/`](Experiments/gradcam_resnet18_refuge) is the standalone localization study behind that metric: ResNet-18 trained in few-shot regime on REFUGE, evaluated against ground-truth disc masks with IoU, SSIM (manual implementation, Gaussian window 11×11) and pointing accuracy, with 2×2 panels per image (original · heatmap · GT mask · GT-vs-contour overlay). Its extractor also implements fundus auto-crop for datasets that are not pre-cropped.

### 4.6 Deployed-classifier evaluation

[`Classifier/Inference/`](Classifier/Inference) produces Table 4. `inference_on_test.py` loads the selected checkpoint, runs `predict` over the test split and returns `y_true`, `P(glaucoma)`, `y_pred` (threshold 0.5) and per-image ids. `metrics_generation.py` computes point estimates plus two interval types appropriate for n = 26: **stratified bootstrap** (5000 resamples drawn within each class, preserving 12 normal / 14 glaucoma) and **Clopper–Pearson** exact intervals for the proportion-based metrics. The bootstrap is seeded; Clopper–Pearson is deterministic.

---

## 5. Reproducing the paper's tables

### Table 2 — backbone selection on REFUGE

```bash
python Experiments/Classifier_Selection/run_experiment.py --data-dir /path/to/REFUGE
```

Expects `<data-dir>/{train,val,test}/{Images,Masks,index.json}`. The first run converts REFUGE into `annotations.json` + `splits.json`; pass `--skip-convert` to reuse them. The run sweeps the methods listed in `config.yaml` (`methods: [euclidean, meta]`) × 3 backbones × N ∈ {1..5} × 5 seeds, benchmarks VRAM once per architecture, and writes `results/selection_summary.json`.

To match the paper exactly, keep `data.image_size: [224, 224]` as shipped in [`config.yaml`](Experiments/Classifier_Selection/config.yaml).

### Table 3 — support-size selection on MM-ODIR-129

Run [`Classifier/meta_nshot_experiment.ipynb`](Classifier/meta_nshot_experiment.ipynb), setting `DATASET_DIR`, `SPLIT_JSON` and `RESULTS_DIR`. It sweeps `N_SHOT ∈ {3,6,9,12,15,21}` for EfficientNet-B0 + meta over five seeds, evaluating macro-F1 on the full validation split (non-episodic evaluation, matching the REFUGE protocol) and Grad-CAM IoU against `disc ∪ cup`.

### Table 4 — deployed classifier on the test split

```bash
python Classifier/Inference/inference_on_test.py --pth /path/to/model_N6.pth --dataset-dir /path/to/images --split-json /path/to/split.json
```

Or run [`inference_metrics_orchestrator.ipynb`](Classifier/Inference/inference_metrics_orchestrator.ipynb), which additionally writes `test_metrics.json` with the per-image predictions.

### Grad-CAM localization study

```bash
python Experiments/gradcam_resnet18_refuge/scripts/train_fewshot.py --config config.yaml --data-dir /path/to/REFUGE
```

```bash
python Experiments/gradcam_resnet18_refuge/scripts/evaluate_fewshot.py --config config.yaml --data-dir /path/to/REFUGE --n-viz 10
```

### Module tests

```bash
python Experiments/gradcam_resnet18_refuge/scripts/test_gradcam_module.py
```

---

## 6. Results as recorded in this repository

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

EfficientNet-B0, meta-trained, N = 6; test split n = 26 (14 glaucoma, 12 normal); threshold 0.5. From [`test_metrics.json`](Classifier/Inference/test_metrics.json).

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

---

## 8. Reproducibility

- Global seed `SEED = 42` in `config.yaml`; every module receives `seed` in its config and calls `set_global_seed()` on construction.
- Five fixed seeds `{42, 123, 456, 789, 1024}` for all multi-seed results, reported as mean ± std.
- **Within iteration `i`, every backbone and method uses `seeds[i]`**, so all configurations see exactly the same support set — this is what makes the Table 2 comparison fair, and it is enforced by the loop order in `run_experiment.py`, not left to chance.
- Splits are always loaded from `splits.json` / `split.json` and never regenerated.
- `torch.backends.cudnn.deterministic = True`; DataLoader workers are seeded through `worker_init_fn`.
- The bootstrap in `metrics_generation.py` is seeded; Clopper–Pearson is exact and deterministic.

---

## 9. Installation

```bash
git clone https://github.com/TheBug95/Medgemma_Segmentation.git
cd Medgemma_Segmentation
pip install torch torchvision easyfsl numpy scipy scikit-learn pyyaml pillow opencv-python matplotlib tqdm
```

Python 3.10+, PyTorch 2.x, and a GPU with ≥16 GB VRAM for the full sweeps (experiments were run on Colab Pro / GCP with L4 or T4). `grad-cam` is optional — `gradcam_module.py` uses it when `use_external: true` and falls back to its manual hook implementation otherwise. `easyfsl` is required for the prototype classifier.

### Notebooks

| Notebook | Purpose |
|---|---|
| [`experiment_orchestrator.ipynb`](Experiments/Classifier_Selection/notebook/experiment_orchestrator.ipynb) | Backbone selection (Table 2) on Colab |
| [`CAM_generation.ipynb`](Experiments/gradcam_resnet18_refuge/CAM_generation.ipynb) | Grad-CAM generation and visualization |
| [`meta_nshot_experiment.ipynb`](Classifier/meta_nshot_experiment.ipynb) | Support-size sweep (Table 3) |
| [`inference_metrics_orchestrator.ipynb`](Classifier/Inference/inference_metrics_orchestrator.ipynb) | Deployed-classifier inference and CIs (Table 4) |

The notebooks currently clone `Medgemma_Segmentation_CIARP_2026` (branch `Classifier-Implementation`); that reference is pending an update to this repository.

---

## 10. Datasets

**REFUGE** — public. 1200 fundus images, 400 per split, 40 glaucoma + 360 normal in each; the test split is unlabelled. Masks are PNGs with 0 = background, 1 = rim, 2 = cup. Used only for backbone selection.

**MM-ODIR-129** — [public on Hugging Face](https://huggingface.co/datasets/TheBug95/MM-ODIR-129). 129 fundus images from 76 patients curated and re-annotated from ODIR-5K: 60 normal, 69 glaucomatous. Five board-certified ophthalmologists annotated binary diagnosis, ordinal scoring of seven glaucoma signs (cup-to-disc ratio, neuroretinal rim, disc hemorrhage, peripapillary atrophy, RNFL defect, disc pallor, vessel changes) and free-text clinical transcriptions; optic disc and cup masks were traced in ImageJ for the 69 glaucomatous images. Splits are patient-level (both eyes of a patient stay in one partition): train 77, validation 26, test 26.

No images or clinical annotations are versioned in this repository.

---

## 13. Known gaps

- Stage II (SAM 2.1 + LoRA) and Stage III (MedGemma conditioning, LoRA adaptation, and the report-evaluation protocol behind Tables 5–7) are not versioned in this repository.
- `Experiments/Classifier_Selection/results/selection_report.md` is still an unfilled template; the real numbers live in the `selection_summary_*.json` files.
- `Experiments/Classifier_Selection/SPEC.md` describes support sizes N = 25/30/35, superseded by the N ∈ {1..5} protocol actually run and published.
- `Experiments/gradcam_resnet18_refuge/README.md` documents `train.py` and `extract_and_evaluate.py`, which no longer exist; the current scripts are `train_fewshot.py` and `evaluate_fewshot.py`.
- There is no `requirements.txt`; dependencies are listed in §9.
- In the paper's Table 2, the IoU standard deviations for DenseNet-121/meta and EfficientNet-B0/meta appear transposed relative to the recorded results (`0.056 ± 0.024` and `0.036 ± 0.058` in print, against `0.056 ± 0.059` and `0.036 ± 0.024` in `selection_summary_few_shot(224x224).json`). Means and all other entries match exactly.
