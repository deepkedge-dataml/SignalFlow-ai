import json
from datetime import datetime
from pathlib import Path

import cv2
from anomalib.data import MVTecLOCO
from anomalib.engine import Engine
from anomalib.models import Patchcore

from src.export_registry import RegistryExporter
from src.overlay_generator import generate_overlay


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = 256
CATEGORY = "juice_bottle"
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_loco_anomaly_detection"


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{CATEGORY}_res{IMAGE_SIZE}_{run_id}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    heatmap_dir = exp_dir / "heatmaps"
    overlay_dir = exp_dir / "overlays"

    exp_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MVTecLOCO(
        root=DATA_ROOT,
        category=CATEGORY,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
    )

    datamodule.setup()
    model = Patchcore(backbone="resnet18", layers=("layer2", "layer3"))

    engine = Engine(
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
    )

    with open(exp_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(f"Resolution: {IMAGE_SIZE}\n")
        f.write("Model: Patchcore\n")
        f.write(f"Category: {CATEGORY}\n")

    engine.fit(model=model, datamodule=datamodule)
    predictions = engine.predict(model=model, datamodule=datamodule)

    exporter = RegistryExporter(
        exp_dir=exp_dir,
        heatmap_dir=heatmap_dir,
        category=CATEGORY,
        run_id=run_id,
        threshold=0.9952539205551147,
    )

    registry_path = exporter.export(predictions)
    print("Registry saved at:", registry_path)

    with open(registry_path, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data["entries"]:
        overlay = generate_overlay(entry["image_path"], entry["heatmap_path"])
        save_path = overlay_dir / f"{entry['image_id']}_overlay.png"
        cv2.imwrite(str(save_path), overlay)
        entry["overlay_path"] = str(save_path)

    return registry_path
