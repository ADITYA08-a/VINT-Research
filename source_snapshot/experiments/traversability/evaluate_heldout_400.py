import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
DATA_ROOT = Path.home() / "Downloads/go_stanford"
LABEL_ROOT = Path.home() / "GENIE-SAMTP/experiments/heldout_labels_100"

CONFIG = REPO / "train/config/vint.yaml"
VINT_CKPT = REPO / "deployment/model_weights/vint.pth"
MULTI_CKPT = REPO / "experiments/traversability/multiscale_v2_1000/best_decoder.pt"
BASE_CKPT = REPO / "experiments/traversability/final_feature_v2_1000/best_decoder.pt"

OUTPUT = REPO / "experiments/traversability/heldout_evaluation_400"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(REPO / "train"))

from vint_train.models.vint.vint import ViNT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multi_mod = load_module(
    "multiscale_eval",
    REPO / "experiments/traversability/train_multiscale_v2.py",
)
base_mod = load_module(
    "baseline_eval",
    REPO / "experiments/traversability/train_final_baseline_v2.py",
)

MultiScaleDecoder = multi_mod.MultiScaleDecoder
FinalFeatureDecoder = base_mod.FinalFeatureDecoder


class MultiFeatureExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.features = {}
        self.handles = []

        for name, index in {"block_1": 1, "block_3": 3, "block_5": 5}.items():
            self.handles.append(
                encoder._blocks[index].register_forward_hook(
                    self.make_hook(name)
                )
            )

    def make_hook(self, name):
        def hook(module, inputs, output):
            self.features[name] = output.detach()
        return hook

    @torch.no_grad()
    def __call__(self, image):
        self.features.clear()
        final = self.encoder.extract_features(image)
        result = dict(self.features)
        result["final"] = final.detach()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


class FinalFeatureExtractor:
    def __init__(self, encoder):
        self.encoder = encoder

    @torch.no_grad()
    def __call__(self, image):
        return {"final": self.encoder.extract_features(image).detach()}

    def close(self):
        pass


def resize_target(target, shape):
    if target.shape == shape:
        return target.astype(np.float32)
    image = Image.fromarray(target.astype(np.float32), mode="F")
    image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def calculate_metrics(prediction, target):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.abs(error).mean()), float(np.sqrt(np.mean(error ** 2)))


def describe(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def main():
    required = [
        DATA_ROOT, LABEL_ROOT / "manifest.json", CONFIG,
        VINT_CKPT, MULTI_CKPT, BASE_CKPT,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    manifest = json.loads((LABEL_ROOT / "manifest.json").read_text())
    samples = manifest["samples"]

    train_manifest_path = (
        Path.home() / "GENIE-SAMTP/experiments/dataset_v2_1000/manifest.json"
    )
    train_manifest = json.loads(train_manifest_path.read_text())

    overlap = set(train_manifest["selected_trajectories"]).intersection(
        manifest["selected_trajectories"]
    )
    if overlap:
        raise RuntimeError(f"Training/held-out trajectory overlap: {overlap}")

    cfg = yaml.safe_load(CONFIG.read_text())
    pretrained = torch.load(VINT_CKPT, map_location="cpu", weights_only=False)

    vint = ViNT(
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
    vint.load_state_dict(pretrained["model"].state_dict(), strict=True)
    vint.to(DEVICE).eval()
    for parameter in vint.parameters():
        parameter.requires_grad = False

    multi_data = torch.load(MULTI_CKPT, map_location=DEVICE, weights_only=False)
    multi = MultiScaleDecoder().to(DEVICE)
    multi.load_state_dict(multi_data["decoder"], strict=True)
    multi.eval()

    base_data = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
    baseline = FinalFeatureDecoder().to(DEVICE)
    baseline.load_state_dict(base_data["decoder"], strict=True)
    baseline.eval()

    multi_features = MultiFeatureExtractor(vint.obs_encoder)
    base_features = FinalFeatureExtractor(vint.obs_encoder)

    transform = transforms.Compose([
        transforms.Resize((64, 85)),
        transforms.ToTensor(),
    ])

    rows, skipped = [], []
    print("Device:", DEVICE)
    print("Reference images:", len(samples))
    print("Trajectory overlap:", len(overlap))

    try:
        for i, item in enumerate(samples, 1):
            image_path = DATA_ROOT / item["image"]
            label_path = LABEL_ROOT / item["label"]

            if not image_path.is_file() or not label_path.is_file():
                skipped.append({
                    "image": item["image"],
                    "reason": "image or label missing",
                })
                continue

            try:
                with Image.open(image_path) as im:
                    rgb_image = im.convert("RGB")
                    rgb = np.asarray(rgb_image)

                target = resize_target(
                    np.load(label_path).astype(np.float32), rgb.shape[:2]
                )
                tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    fm = multi_features(tensor)
                    pm = torch.sigmoid(
                        multi(fm, output_size=rgb.shape[:2])
                    )[0, 0].cpu().numpy()

                    fb = base_features(tensor)
                    pb = torch.sigmoid(
                        baseline(fb, output_size=rgb.shape[:2])
                    )[0, 0].cpu().numpy()

                mae_m, rmse_m = calculate_metrics(pm, target)
                mae_b, rmse_b = calculate_metrics(pb, target)

                rows.append({
                    "trajectory": item["trajectory"],
                    "image": item["image"],
                    "reference_mean": float(target.mean()),
                    "multiscale_mean": float(pm.mean()),
                    "baseline_mean": float(pb.mean()),
                    "multiscale_mae": mae_m,
                    "baseline_mae": mae_b,
                    "multiscale_rmse": rmse_m,
                    "baseline_rmse": rmse_b,
                    "multiscale_better_mae": mae_m < mae_b,
                })
            except Exception as exc:
                skipped.append({"image": item["image"], "reason": repr(exc)})
                print("FAILED:", item["image"], repr(exc))

            if i % 25 == 0 or i == len(samples):
                print(f"Processed {i}/{len(samples)}; valid={len(rows)}")

    finally:
        multi_features.close()
        base_features.close()

    if not rows:
        raise RuntimeError("No images evaluated successfully.")

    with (OUTPUT / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trajectory"]].append(row)

    trajectory_rows = []
    for trajectory, items in sorted(grouped.items()):
        trajectory_rows.append({
            "trajectory": trajectory,
            "num_images": len(items),
            "multiscale_mae": float(np.mean([x["multiscale_mae"] for x in items])),
            "baseline_mae": float(np.mean([x["baseline_mae"] for x in items])),
            "multiscale_rmse": float(np.mean([x["multiscale_rmse"] for x in items])),
            "baseline_rmse": float(np.mean([x["baseline_rmse"] for x in items])),
        })

    with (OUTPUT / "per_trajectory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    m_mae = [r["multiscale_mae"] for r in rows]
    b_mae = [r["baseline_mae"] for r in rows]
    m_rmse = [r["multiscale_rmse"] for r in rows]
    b_rmse = [r["baseline_rmse"] for r in rows]

    tm_mae = [r["multiscale_mae"] for r in trajectory_rows]
    tb_mae = [r["baseline_mae"] for r in trajectory_rows]
    tm_rmse = [r["multiscale_rmse"] for r in trajectory_rows]
    tb_rmse = [r["baseline_rmse"] for r in trajectory_rows]

    wins = sum(r["multiscale_better_mae"] for r in rows)
    trajectory_wins = sum(a < b for a, b in zip(tm_mae, tb_mae))

    report = {
        "teacher": "SAM-TP pseudo-labels",
        "evaluated_images": len(rows),
        "skipped_images": len(skipped),
        "unique_trajectories": len(trajectory_rows),
        "training_trajectory_overlap": len(overlap),
        "image_level": {
            "multiscale_mae": describe(m_mae),
            "baseline_mae": describe(b_mae),
            "multiscale_rmse": describe(m_rmse),
            "baseline_rmse": describe(b_rmse),
            "multiscale_wins_by_mae": wins,
            "multiscale_win_fraction": wins / len(rows),
        },
        "trajectory_level": {
            "multiscale_mae": describe(tm_mae),
            "baseline_mae": describe(tb_mae),
            "multiscale_rmse": describe(tm_rmse),
            "baseline_rmse": describe(tb_rmse),
            "multiscale_wins_by_mae": trajectory_wins,
            "trajectory_win_fraction": trajectory_wins / len(trajectory_rows),
        },
        "skipped": skipped,
        "caveat": "These metrics measure agreement with SAM-TP pseudo-labels, not verified physical traversability.",
    }

    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))

    print("\n========== FULL HELD-OUT EVALUATION ==========")
    print(f"Valid images: {len(rows)}; skipped: {len(skipped)}")
    print("Unique trajectories:", len(trajectory_rows))
    print(f"Multiscale image-level MAE wins: {wins}/{len(rows)}")
    print(f"Multiscale trajectory-level MAE wins: {trajectory_wins}/{len(trajectory_rows)}")
    print("\nIMAGE-LEVEL")
    print(f"Multiscale MAE:  {np.mean(m_mae):.5f} ± {np.std(m_mae):.5f}")
    print(f"Baseline MAE:    {np.mean(b_mae):.5f} ± {np.std(b_mae):.5f}")
    print(f"Multiscale RMSE: {np.mean(m_rmse):.5f} ± {np.std(m_rmse):.5f}")
    print(f"Baseline RMSE:   {np.mean(b_rmse):.5f} ± {np.std(b_rmse):.5f}")
    print("\nTRAJECTORY-LEVEL")
    print(f"Multiscale MAE:  {np.mean(tm_mae):.5f}")
    print(f"Baseline MAE:    {np.mean(tb_mae):.5f}")
    print(f"Multiscale RMSE: {np.mean(tm_rmse):.5f}")
    print(f"Baseline RMSE:   {np.mean(tb_rmse):.5f}")
    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()
