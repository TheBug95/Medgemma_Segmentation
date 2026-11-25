# Stage II — Optic-disc segmentation

This stage trains the SAM ViT-B segmenter with LoRA, runs inference on MM-ODIR-129, and computes the IoU and Dice results in Table 5. The code comes from the [original SAM-Finetuning-LoRA experiment](https://github.com/TheBug95/SAM-Finetuning-LoRA/tree/f4274b14bb00845d233baf3db40a3859372fb559).

The checkpoint used for Table 5 identifies the architecture as SAM ViT-B (`vit_b`). Figure 1 of the paper labels this stage SAM 2.1.

## Setup

From the repository root, install the dependencies in a Python 3.10+ virtual environment.

```bash
python -m pip install -r Stage_II/requirements.txt
```

Download the [official SAM ViT-B weights](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `Stage_II/checkpoints/sam_vit_b_01ec64.pth`. The commands also accept `--sam-checkpoint`.

The default dataset is the flat copy of MM-ODIR-129 in [Stage_III/dataset](../Stage_III/dataset). The loader uses the existing `split.json`, the original photographs, and `*_disc.npy` masks. Images without an expert mask receive an empty target, as in the archived experiment. Use `--dataset-root` for another flat copy.

## Train and evaluate

```bash
python -m Stage_II.scripts.train
python -m Stage_II.scripts.inference
python -m Stage_II.scripts.inference --checkpoint Stage_II/results/sam_lora/final_model.pt
python -m Stage_II.scripts.ci_iou_dice
```

The first inference command evaluates the pretrained baseline. The second loads a complete fine-tuned checkpoint. To rerun the archived model, pass its `final_model.pt` as `--checkpoint`.

The two inference runs write `metrics.json`, `detailed_results.json` and `sam_masks/` under `Stage_II/results/sam_base_test/` and `Stage_II/results/sam_lora_test/`. Add `--save-visualizations` to save the original comparison plots. The final command calculates 95% Student's t intervals and writes `Stage_II/results/table5.json`. It also accepts `--base-results` and `--lora-results` for existing inference outputs.

The documented settings are in [config.yaml](config.yaml). Paths in YAML resolve relative to that file. CUDA is the default device, and train and inference also accept `--device cpu`.

## Table 5 protocol

Both runs use the same 26-image test split. The experiment derives point and box prompts from the annotated masks and uses the original empty-target prompt logic for normal images. Here, *zero-shot* refers to evaluating the pretrained SAM without domain-specific weights. Stage III reads the archived overlays for its report-generation experiments.

| Model | Mean IoU | 95% CI | Mean Dice | 95% CI |
|---|---:|---:|---:|---:|
| SAM ViT-B | 0.441 | [0.270, 0.611] | 0.483 | [0.298, 0.668] |
| SAM ViT-B + LoRA | 0.910 | [0.844, 0.976] | 0.942 | [0.887, 0.997] |
