import json
from pathlib import Path
import numpy as np
from PIL import Image

class RegistryExporter:
    def __init__(self, exp_dir:Path,heatmap_dir:Path, category: str, run_id: str,threshold: float):
        self.exp_dir=exp_dir
        self.heatmap_dir=heatmap_dir
        self.category=category
        self.run_id=run_id
        self.threshold = threshold


    def save_heatmap(self, anomaly_map_2d, out_path: Path) -> None:
        heat = anomaly_map_2d.detach().cpu().numpy()
        heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_img = (255 * heat_norm).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(heat_img, mode="L").save(out_path)

    def export(self, preds):
        # Ensure directories exist
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        entries = []

        for batch in preds:
            for i in range(len(batch.image_path)):
                image_path = batch.image_path[i]
                image_id = Path(image_path).stem

                out_path = self.heatmap_dir / f"{image_id}_heat.png"

                anomaly_map = batch.anomaly_map
                # needs to check the way the maop i saved and extract it
                # either it has been saved as [batch, channel, height,width] or [batch, height, width]
                anomaly_map_2d = anomaly_map[i] if anomaly_map.ndim == 3 else anomaly_map[i, 0]
                self.save_heatmap(anomaly_map_2d, out_path)
                score = float(batch.pred_score[i])
                label = int(batch.pred_label[i])
                gt_label= int(batch.gt_label[i])
                calibrated_label = int(score >= self.threshold)
                entries.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "heatmap_path": str(out_path),
                    "pred_score": score,
                    "pred_label": label,
                    "pred_label_calibrated": calibrated_label,
                    'gt_label': gt_label,
                    "threshold_used": self.threshold
                })

                entries = sorted(entries, key=lambda x: x["pred_score"], reverse=True)

                for rank, entry in enumerate(entries, start=1):
                    entry["rank"] = rank

                normal_scores = [entry["pred_score"] for entry in entries if entry["gt_label"] == 0]
                threshold = float(np.percentile(normal_scores, 95))

                for entry in entries:
                    entry["threshold"] = threshold
                    entry["score_based_label"] = 1 if entry["pred_score"] >= threshold else 0


        registry = {
            "metadata": {
                "project": "SignalFlow",
                "category": self.category,
                "run_id": self.run_id,
                "num_entries": len(entries),
            },
            "entries": entries
        }

        registry_path=self.exp_dir/"anomaly_registry.json"

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4) # indent for readability otherwise it is not readable

        return registry_path



