"""Ajusta SAM ViT-B con LoRA usando las particiones de MM-ODIR-129."""

import argparse
import logging
from pathlib import Path

from ..modules.config import DEFAULT_CONFIG, load_config, run_metadata
from ..modules.dataset import create_glaucoma_dataloaders
from ..modules.model import create_sam_lora_model
from ..modules.trainer import SAMTrainer
from ..modules.utils import get_device, save_metrics, set_seed


logger = logging.getLogger(__name__)


def main() -> None:
    """Ejecuta el entrenamiento y guarda el historial y el checkpoint."""
    parser = argparse.ArgumentParser(description="Fine-tune SAM ViT-B on MM-ODIR-129")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Parent directory for the training run")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    config.glaucoma_data.dataset_root = args.dataset_root or config.glaucoma_data.dataset_root
    config.model.checkpoint_path = args.sam_checkpoint or config.model.checkpoint_path
    config.output_dir = args.output_dir or config.output_dir
    config.device = args.device or config.device
    checkpoint = config.model.checkpoint_path
    if checkpoint is None or not checkpoint.is_file():
        parser.error(f"Provide the official ViT-B weights at {checkpoint} or via --sam-checkpoint.")

    set_seed(config.seed)
    device = get_device(config.device)
    config.device = str(device)
    train_loader, val_loader, _ = create_glaucoma_dataloaders(
        config, num_workers=config.glaucoma_data.num_workers
    )
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    model = create_sam_lora_model(config, device=device)
    trainer = SAMTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device
    )
    history = trainer.train()
    save_metrics(
        {**run_metadata(config), **history},
        config.experiment_dir / "training_history.json",
    )
    final_path = config.experiment_dir / "final_model.pt"
    model.save_checkpoint(final_path, epoch=trainer.current_epoch + 1)
    logger.info("Saved trained checkpoint to %s", final_path)


if __name__ == "__main__":
    main()
