import sys
import csv
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
ROOT = Path.home() / "Downloads/RUGD_sample-data"
IMAGE_DIR = ROOT / "images"
ANN_DIR = ROOT / "annotations"

OUT = REPO / "experiments/traversability/rugd_sample_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

VINT_CKPT = REPO / "deployment/model_weights/vint.pth"
VINT_CONFIG = REPO / "train/config/vint.yaml"
DECODER_CKPT = (
    REPO / "experiments/traversability/multiscale_v2_1000/best_decoder.pt"
)
TRAIN_SCRIPT = (
    REPO / "experiments/traversability/train_multiscale_v2.py"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RUGD RGB colors from RUGD_annotation-colormap.txt.
COLORS = {
    "void": (0, 0, 0),
    "dirt": (108, 64, 20),
    "sand": (255, 229, 204),
    "grass": (0, 102, 0),
    "tree": (0, 255, 0),
    "pole": (0, 153, 153),
    "water": (0, 128, 255),
    "sky": (0, 0, 255),
    "vehicle": (255, 255, 0),
    "container/generic-object": (255, 0, 127),
    "asphalt": (64, 64, 64),
    "gravel": (255, 128, 0),
    "building": (255, 0, 0),
    "mulch": (153, 76, 0),
    "rock-bed": (102, 102, 0),
    "log": (102, 0, 0),
    "bicycle": (0, 255, 128),
    "person": (204, 153, 255),
    "fence": (102, 0, 204),
    "bush": (255, 153, 204),
    "sign": (0, 102, 102),
    "rock": (153, 204, 255),
    "bridge": (102, 255, 255),
    "concrete": (101, 101, 11),
    "picnic-table": (114, 85, 47),
}

# Provisional classes only. These are not physical ground truth.
GROUND = {
    "dirt", "sand", "grass", "asphalt",
    "gravel", "mulch", "concrete",
}
OBSTACLE = {
    "tree", "pole", "vehicle", "container/generic-object",
    "building", "log", "bicycle", "person", "fence",
    "sign", "picnic-table",
}
UNCERTAIN = {"water", "rock-bed", "rock", "bush", "bridge", "void", "sky"}

sys.path.insert(0, str(REPO / "train"))
from vint_train.models.vint.vint import ViNT


def load_module(path):
    spec = importlib.util.spec_from_file_location("ms_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model():
    cfg = yaml.safe_load(VINT_CONFIG.read_text())
    ckpt = torch.load(VINT_CKPT, map_location="cpu", weights_only=False)

    model = ViNT(
        context_size=cfg["context_size"],
        len_traj_pred=cfg["len_traj_pred"],
        learn_angle=cfg["learn_angle"],
        obs_encoder=cfg["obs_encoder"],
        obs_encoding_size=cfg["obs_encoding_size"],
        late_fusion=cfg["late_fusion"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
    )
    model.load_state_dict(ckpt["model"].state_dict(), strict=True)
    return model.to(DEVICE).eval()


def decode_annotation(annotation):
    # Exact RGB matching; unknown colors remain unclassified.
    arr = np.asarray(annotation.convert("RGB"))
    class_map = np.full(arr.shape[:2], "unknown", dtype=object)
    for name, rgb in COLORS.items():
        class_map[np.all(arr == np.asarray(rgb), axis=-1)] = name
    return class_map


def main():
    module = load_module(TRAIN_SCRIPT)
    model = load_model()

    decoder = module.MultiScaleDecoder().to(DEVICE)
    saved = torch.load(DECODER_CKPT, map_location=DEVICE, weights_only=False)
    state = saved["decoder"] if "decoder" in saved else saved
    decoder.load_state_dict(state, strict=True)
    decoder.eval()

    extractor = module.FeatureExtractor(model.obs_encoder)
    transform = transforms.Compose([
        transforms.Resize((64, 85)),
        transforms.ToTensor(),
    ])

    per_class = {
        name: {"pixels": 0, "score_sum": 0.0, "images": 0}
        for name in COLORS
    }
    per_image = []

    try:
        for image_path in sorted(IMAGE_DIR.glob("*.png")):
            ann_path = ANN_DIR / image_path.name
            if not ann_path.exists():
                print("Skipping missing annotation:", image_path.name)
                continue

            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                tensor = transform(rgb).unsqueeze(0).to(DEVICE)

            with Image.open(ann_path) as ann:
                classes = decode_annotation(ann)

            with torch.inference_mode():
                features = extractor(tensor)
                logits = decoder(features, output_size=classes.shape)
                scores = torch.sigmoid(logits)[0, 0].cpu().numpy()

            if scores.shape != classes.shape:
                raise RuntimeError(
                    f"Shape mismatch for {image_path.name}: "
                    f"{scores.shape} vs {classes.shape}"
                )

            row = {"image": image_path.name}
            valid_ground = np.zeros(classes.shape, dtype=bool)
            valid_obstacle = np.zeros(classes.shape, dtype=bool)

            for name in COLORS:
                mask = classes == name
                count = int(mask.sum())
                if count == 0:
                    continue

                mean_score = float(scores[mask].mean())
                row[f"{name}_pixels"] = count
                row[f"{name}_mean_score"] = mean_score

                per_class[name]["pixels"] += count
                per_class[name]["score_sum"] += float(scores[mask].sum())
                per_class[name]["images"] += 1

                if name in GROUND:
                    valid_ground |= mask
                elif name in OBSTACLE:
                    valid_obstacle |= mask

            row["ground_proxy_pixels"] = int(valid_ground.sum())
            row["ground_proxy_mean_score"] = (
                float(scores[valid_ground].mean())
                if valid_ground.any() else None
            )
            row["obstacle_proxy_pixels"] = int(valid_obstacle.sum())
            row["obstacle_proxy_mean_score"] = (
                float(scores[valid_obstacle].mean())
                if valid_obstacle.any() else None
            )

            # A position-only baseline: average prediction per image row.
            # It is diagnostic, not a valid traversability model.
            row_means = scores.mean(axis=1)
            position_only = np.repeat(
                row_means[:, None], scores.shape[1], axis=1
            )
            row["position_baseline_mae_vs_decoder"] = float(
                np.abs(position_only - scores).mean()
            )
            per_image.append(row)

            print(
                f"{image_path.name}: "
                f"ground-proxy mean={row['ground_proxy_mean_score']}, "
                f"obstacle-proxy mean={row['obstacle_proxy_mean_score']}"
            )

    finally:
        extractor.close()

    # Aggregate per-class means.
    aggregate = {}
    for name, values in per_class.items():
        aggregate[name] = {
            "pixels": values["pixels"],
            "images_present": values["images"],
            "mean_score": (
                values["score_sum"] / values["pixels"]
                if values["pixels"] else None
            ),
        }

    csv_path = OUT / "semantic_per_image.csv"
    fields = sorted({k for row in per_image for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image)

    summary = {
        "dataset": "RUGD sample",
        "image_count": len(per_image),
        "ground_proxy_classes": sorted(GROUND),
        "obstacle_proxy_classes": sorted(OBSTACLE),
        "uncertain_classes": sorted(UNCERTAIN),
        "per_class": aggregate,
        "interpretation_warning": (
            "These are semantic proxy correlations, not ground-truth "
            "traversability metrics. Scores are learned from SAM-TP "
            "pseudo-labels. Class membership alone does not establish "
            "physical traversability for a particular robot."
        ),
    }
    (OUT / "semantic_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nSaved:", csv_path)
    print("Saved:", OUT / "semantic_summary.json")
    print("\nPer-class mean scores:")
    for name, values in aggregate.items():
        if values["mean_score"] is not None:
            print(f"  {name:25s} {values['mean_score']:.4f}")

if __name__ == "__main__":
    main()
