"""Configuración de la etapa de segmentación del disco óptico."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@dataclass
class GlaucomaDataConfig:
    """Datos de MM-ODIR-129 y opciones para construir las máscaras."""

    dataset_root: Path
    split_file: Path
    image_size: int
    num_workers: int
    mask_format: str
    mask_suffixes: list[str]
    use_only_labeled: bool

    def get_split_file_path(self) -> Path:
        """Devuelve el archivo con las particiones originales."""
        return self.dataset_root / self.split_file


@dataclass
class ModelConfig:
    """Arquitectura, checkpoint y adaptadores de SAM."""

    model_type: str
    checkpoint_path: Path | None
    freeze_image_encoder: bool
    freeze_prompt_encoder: bool
    freeze_mask_decoder: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]


@dataclass
class TrainingConfig:
    """Hiperparámetros del ajuste con LoRA."""

    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    mixed_precision: bool
    optimizer: str
    betas: list[float]
    scheduler: str
    focal_loss_weight: float
    dice_loss_weight: float
    iou_loss_weight: float
    patience: int
    min_delta: float
    save_every: int
    keep_last_n: int


@dataclass
class EvaluationConfig:
    """Configuración de inferencia y de los intervalos de confianza."""

    split: str
    threshold: float
    batch_size: int
    confidence: float


@dataclass
class Config:
    """Configuración completa de la etapa, con rutas locales resueltas."""

    glaucoma_data: GlaucomaDataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    seed: int
    device: str
    output_dir: Path
    experiment_name: str
    wandb_project: str | None = None
    wandb_entity: str | None = None

    @property
    def experiment_dir(self) -> Path:
        """Devuelve el directorio de la corrida de entrenamiento."""
        return self.output_dir / self.experiment_name


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    """Carga las opciones y resuelve las rutas relativas al YAML."""
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)

    data_options = settings["data"]
    data_options["dataset_root"] = (path.parent / data_options["dataset_root"]).resolve()
    data_options["split_file"] = Path(data_options["split_file"])
    model_options = settings["model"]
    checkpoint = model_options["checkpoint_path"]
    model_options["checkpoint_path"] = (path.parent / checkpoint).resolve() if checkpoint else None

    return Config(
        glaucoma_data=GlaucomaDataConfig(**data_options),
        model=ModelConfig(**model_options),
        training=TrainingConfig(**settings["training"]),
        evaluation=EvaluationConfig(**settings["evaluation"]),
        seed=settings["seed"],
        device=settings["device"],
        output_dir=(path.parent / settings["output_dir"]).resolve(),
        experiment_name=settings["experiment_name"],
    )


def run_metadata(config: Config) -> dict:
    """Devuelve la semilla, la configuración y el momento de la corrida."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "config": asdict(config),
    }
