"""Recalcula la Tabla 5 a partir de las predicciones por imagen."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import t

from ..modules.config import DEFAULT_CONFIG, load_config, run_metadata


logger = logging.getLogger(__name__)


def summarize(path: Path, confidence: float) -> dict:
    """Calcula las medias y los intervalos t sobre las imágenes de un split."""
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if len(rows) < 2:
        raise ValueError(f"At least two images are required for a confidence interval: {path}")

    result = {"num_images": len(rows)}
    for metric in ("iou", "dice"):
        scores = np.asarray([row["objects"][0][metric] if row["objects"] else 0.0 for row in rows])
        mean = float(scores.mean())
        sem = float(scores.std(ddof=1) / np.sqrt(len(scores)))
        interval = t.interval(confidence, df=len(scores) - 1, loc=mean, scale=sem) if sem else (
            mean,
            mean,
        )
        result[metric] = {"mean": mean, "confidence_interval": [float(x) for x in interval]}
    return result


def main() -> None:
    """Guarda el resumen de las dos condiciones de la Tabla 5."""
    parser = argparse.ArgumentParser(description="Recalculate Table 5 segmentation metrics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--lora-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    confidence = config.evaluation.confidence
    baseline = args.base_results or config.output_dir / "sam_base_test" / "detailed_results.json"
    fine_tuned = args.lora_results or config.output_dir / "sam_lora_test" / "detailed_results.json"
    output = args.output or config.output_dir / "table5.json"
    results = {
        **run_metadata(config),
        "confidence_level": confidence,
        "sam_base": summarize(baseline, confidence),
        "sam_lora": summarize(fine_tuned, confidence),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, default=str)
        stream.write("\n")
    logger.info("Saved per-method scores and confidence intervals to %s", output)


if __name__ == "__main__":
    main()
