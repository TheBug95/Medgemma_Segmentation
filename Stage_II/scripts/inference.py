"""Evalúa SAM base o ajustado con LoRA y exporta sus máscaras."""

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from ..modules.config import DEFAULT_CONFIG, load_config, run_metadata
from ..modules.dataset import GlaucomaDataset, collate_fn, get_transforms
from ..modules.inference import SAMInference
from ..modules.model import create_sam_lora_model
from ..modules.utils import get_device, save_metrics, set_seed


logger = logging.getLogger(__name__)


def main() -> None:
    """Evalúa un split y escribe las métricas por imagen y el agregado."""
    parser = argparse.ArgumentParser(description="Evaluate optic-disc segmentation on MM-ODIR-129")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path, help="Pretrained SAM ViT-B weights")
    parser.add_argument("--checkpoint", type=Path, help="Complete fine-tuned LoRA checkpoint")
    parser.add_argument("--split", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--save-visualizations", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    data = config.glaucoma_data
    data.dataset_root = args.dataset_root or data.dataset_root
    config.device = args.device or config.device
    config.evaluation.split = args.split or config.evaluation.split
    output_dir = args.output_dir or config.output_dir / (
        "sam_lora_test" if args.checkpoint else "sam_base_test"
    )
    config.output_dir = output_dir
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Fine-tuned checkpoint does not exist: {args.checkpoint}")
    if args.checkpoint or args.sam_checkpoint:
        config.model.checkpoint_path = args.sam_checkpoint
    if config.model.checkpoint_path and not config.model.checkpoint_path.is_file():
        parser.error(f"Pretrained SAM weights do not exist: {config.model.checkpoint_path}")
    if not args.checkpoint and not config.model.checkpoint_path:
        parser.error("Provide pretrained SAM weights via --sam-checkpoint.")

    set_seed(config.seed)
    device = get_device(config.device)
    config.device = str(device)
    dataset = GlaucomaDataset(
        split_file=data.get_split_file_path(),
        dataset_root=data.dataset_root,
        split=config.evaluation.split,
        transforms=get_transforms(data.image_size, is_train=False),
        image_size=data.image_size,
        use_point_prompts=True,
        use_box_prompts=True,
        mask_format=data.mask_format,
        mask_suffixes=data.mask_suffixes,
        use_only_labeled=data.use_only_labeled,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=data.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = create_sam_lora_model(config, device=device)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = SAMInference(model, device=device, threshold=config.evaluation.threshold)
    metrics, details = evaluator.predict_batch(
        dataloader,
        save_dir=output_dir / "visualizations" if args.save_visualizations else None,
        masks_dir=output_dir / "sam_masks",
    )
    save_metrics(
        {**run_metadata(config), "fine_tuned_checkpoint": args.checkpoint, **metrics},
        output_dir / "metrics.json",
    )
    save_metrics(details, output_dir / "detailed_results.json")
    logger.info(
        "IoU %.4f, Dice %.4f, images %d",
        metrics["mean_iou"],
        metrics["mean_dice"],
        metrics["num_images"],
    )
    logger.info("Saved masks and scores to %s", output_dir)


if __name__ == "__main__":
    main()
