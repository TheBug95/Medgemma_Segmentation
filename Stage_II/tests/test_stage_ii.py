"""Pruebas de segmentación con archivos sintéticos y un modelo pequeño."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from Stage_II.modules.config import DEFAULT_CONFIG, load_config, run_metadata
from Stage_II.modules.dataset import GlaucomaDataset, collate_fn, get_transforms
from Stage_II.modules.inference import SAMInference
from Stage_II.modules.model import create_sam_lora_model
from Stage_II.modules.trainer import SAMTrainer
from Stage_II.modules.utils import save_metrics, set_seed
from Stage_II.scripts.ci_iou_dice import summarize


class TinySAM(nn.Module):
    """Modelo que permite verificar LoRA y el checkpoint en CPU."""

    def __init__(self) -> None:
        """Crea los bloques que espera el wrapper de SAM."""
        super().__init__()
        self.image_encoder = nn.ModuleDict({"qkv": nn.Linear(4, 4), "proj": nn.Linear(4, 4)})
        self.prompt_encoder = nn.Linear(4, 4)
        self.mask_decoder = nn.Linear(4, 4)


class EmptyMaskModel(nn.Module):
    """Segmentador sintético para probar la ruta de evaluación."""

    def forward(self, image: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve logits negativos y un IoU estimado por imagen."""
        batch, _, height, width = image.shape
        return torch.full((batch, 1, height, width), -8.0), torch.zeros((batch, 1))


class StageIITest(unittest.TestCase):
    """Verifica la carga, adaptación, evaluación y exportación."""

    def setUp(self) -> None:
        """Crea imágenes, una máscara binaria y un split de prueba."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.split = self.root / "split.json"
        items = [
            {"id": 1, "image_filename": "one.jpg", "label": "Normal"},
            {"id": 2, "image_filename": "two.jpg", "label": "Pathological"},
        ]
        self.split.write_text(json.dumps({"test": items}), encoding="utf-8")
        for item in items:
            Image.new("RGB", (32, 32)).save(self.root / item["image_filename"])
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:19, 9:21] = 1
        np.save(self.root / "two_disc.npy", mask)

    def dataset(self) -> GlaucomaDataset:
        """Devuelve las dos imágenes del split y sus máscaras."""
        return GlaucomaDataset(
            split_file=self.split,
            dataset_root=self.root,
            split="test",
            transforms=get_transforms(32, is_train=False),
            image_size=32,
            mask_suffixes=["disc"],
            use_only_labeled=False,
        )

    def test_config_and_metadata(self) -> None:
        """Comprueba las rutas que usa la etapa al ejecutarse desde el repositorio."""
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.model.model_type, "vit_b")
        self.assertEqual(config.glaucoma_data.dataset_root.parent.name, "Stage_III")
        output = self.root / "metadata.json"
        save_metrics(run_metadata(config), output)
        metadata = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["config"]["model"]["lora_rank"], 8)

    def test_masks_for_normal_and_pathological_images(self) -> None:
        """Conserva el orden del split y los blancos esperados para cada caso."""
        set_seed(42)
        data = self.dataset()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["masks"].sum().item(), 0)
        self.assertEqual(data[1]["masks"].sum().item(), 9 * 12)

    def test_checkpoint_and_optimizer(self) -> None:
        """Verifica el formato del checkpoint y el optimizador de entrenamiento."""
        config = load_config()
        config.output_dir = self.root
        config.model.checkpoint_path = None
        config.training.mixed_precision = False
        with patch("Stage_II.modules.model.sam_model_registry", {"vit_b": TinySAM}):
            model = create_sam_lora_model(config, device="cpu")
            saved = self.root / "tiny.pt"
            model.save_checkpoint(saved, epoch=3)
            epoch, _ = model.load_checkpoint(saved)
        self.assertEqual(epoch, 3)
        loader = DataLoader(self.dataset(), batch_size=1, collate_fn=collate_fn)
        trainer = SAMTrainer(
            model, train_loader=loader, val_loader=loader, config=config, device=torch.device("cpu")
        )
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

    def test_inference_and_ci(self) -> None:
        """Comprueba métricas, PNGs e intervalos con blancos conocidos."""
        loader = DataLoader(self.dataset(), batch_size=1, collate_fn=collate_fn)
        mask_dir = self.root / "masks"
        inference = SAMInference(EmptyMaskModel(), device="cpu")
        metrics, rows = inference.predict_batch(loader, masks_dir=mask_dir)
        self.assertEqual(metrics["num_images"], 2)
        self.assertAlmostEqual(metrics["mean_dice"], 0.5, places=5)
        self.assertEqual(len(list(mask_dir.glob("*.png"))), 2)
        scores = self.root / "detailed_results.json"
        save_metrics(rows, scores)
        self.assertAlmostEqual(summarize(scores, 0.95)["iou"]["mean"], 0.5, places=5)
