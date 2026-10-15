import csv
import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
SAM_ROOT = Path.home() / "GENIE-SAMTP"
IMAGE_ROOT = Path.home() / "Downloads/RUGD_sample-data/images"
TEACHER_ROOT = SAM_ROOT / "experiments/rugd_teacher_maps/maps"

EVAL_SCRIPT = REPO / "experiments/traversability/evaluate_rugd_sample.py"
MS_SCRIPT = REPO / "experiments/traversability/train_multiscale_v2.py"
BASE_SCRIPT = REPO / "experiments/traversability/train_final_baseline_v2.py"

OUT = REPO / "experiments/traversability/rugd_teacher_comparison"
VIS = OUT / "visualizations"
VIS.mkdir(parents=True, exist_ok=True)

MAP_SIZE = (120, 160)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spatial_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def metrics(pred, teacher):
    error = pred - teacher
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "spatial_pearson": spatial_corr(pred, teacher),
        "prediction_mean": float(pred.mean()),
        "teacher_mean": float(teacher.mean()),
        "mean_bias": float(error.mean()),
    }


def main():
    start = time.time()

    for path in [EVAL_SCRIPT, MS_SCRIPT, BASE_SCRIPT]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(IMAGE_ROOT)

    if not TEACHER_ROOT.is_dir():
        raise FileNotFoundError(TEACHER_ROOT)

    # Reuse the existing, tested ViNT loading and preprocessing code.
    ev = load_module("existing_rugd_evaluator", EVAL_SCRIPT)
    ms_mod = ev.import_module("train_multiscale_v2", MS_SCRIPT)
    base_mod = ev.import_module("train_final_baseline_v2", BASE_SCRIPT)

    device = ev.DEVICE
    model = ev.load_vint()

    ms_decoder = ev.load_decoder(
        ms_mod,
        ev.MULTISCALE_CHECKPOINT,
        ms_mod.MultiScaleDecoder,
    )
    base_decoder = ev.load_decoder(
        base_mod,
        ev.BASELINE_CHECKPOINT,
        base_mod.FinalFeatureDecoder,
    )

    ms_extractor = ms_mod.FeatureExtractor(model.obs_encoder)
    base_extractor = base_mod.FeatureExtractor(model.obs_encoder)

    transform = transforms.Compose([
        transforms.Resize((64, 85)),
        transforms.ToTensor(),
    ])

    image_paths = sorted(IMAGE_ROOT.glob("*.png"))
    rows = []

    print("Device:", device)
    print("Images:", len(image_paths))
    print("Teacher maps:", TEACHER_ROOT)
    print("Output:", OUT)

    try:
        for i, image_path in enumerate(image_paths, start=1):
            teacher_path = TEACHER_ROOT / f"{image_path.stem}.npy"

            if not teacher_path.is_file():
                print(f"[{i}/{len(image_paths)}] MISSING TEACHER: {image_path.name}")
                continue

            print(f"[{i}/{len(image_paths)}] {image_path.name}", flush=True)

            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                original = np.asarray(rgb)
                image_tensor = transform(rgb).unsqueeze(0).to(device)

            teacher_native = np.load(teacher_path).astype(np.float32)

            if teacher_native.ndim != 2:
                raise ValueError(
                    f"Invalid teacher map shape for {image_path.name}: "
                    f"{teacher_native.shape}"
                )

            # Resize the teacher map to the student's 120x160 grid.
            # The saved native-resolution teacher map remains unchanged.
            teacher_t = torch.from_numpy(teacher_native)[None, None]
            teacher = F.interpolate(
                teacher_t,
                size=MAP_SIZE,
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            with torch.inference_mode():
                ms_features = ms_extractor(image_tensor)
                ms_logits = ms_decoder(ms_features, output_size=MAP_SIZE)
                ms_pred = torch.sigmoid(ms_logits)[0, 0].cpu().numpy()

                base_features = base_extractor(image_tensor)
                base_logits = base_decoder(
                    base_features, output_size=MAP_SIZE
                )
                base_pred = torch.sigmoid(
                    base_logits
                )[0, 0].cpu().numpy()

            ms_metrics = metrics(ms_pred, teacher)
            base_metrics = metrics(base_pred, teacher)

            row = {
                "image": image_path.name,
                **{f"multiscale_{k}": v for k, v in ms_metrics.items()},
                **{f"baseline_{k}": v for k, v in base_metrics.items()},
            }
            rows.append(row)

            # Six-panel visualization for quick qualitative inspection.
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))

            axes[0, 0].imshow(original)
            axes[0, 0].set_title("RGB")

            axes[0, 1].imshow(
                teacher, cmap="viridis", vmin=0, vmax=1
            )
            axes[0, 1].set_title("SAM-TP teacher")

            axes[0, 2].imshow(
                ms_pred, cmap="viridis", vmin=0, vmax=1
            )
            axes[0, 2].set_title(
                f"Multiscale\nMAE={ms_metrics['mae']:.3f}, "
                f"r={ms_metrics['spatial_pearson']:.3f}"
            )

            axes[1, 0].imshow(
                base_pred, cmap="viridis", vmin=0, vmax=1
            )
            axes[1, 0].set_title(
                f"Final-feature baseline\nMAE={base_metrics['mae']:.3f}, "
                f"r={base_metrics['spatial_pearson']:.3f}"
            )

            axes[1, 1].imshow(
                np.abs(ms_pred - teacher),
                cmap="magma", vmin=0, vmax=1
            )
            axes[1, 1].set_title("Multiscale absolute error")

            axes[1, 2].imshow(
                np.abs(base_pred - teacher),
                cmap="magma", vmin=0, vmax=1
            )
            axes[1, 2].set_title("Baseline absolute error")

            for ax in axes.flat:
                ax.axis("off")

            fig.suptitle(image_path.name)
            fig.tight_layout()
            fig.savefig(
                VIS / f"{image_path.stem}_teacher_comparison.png",
                dpi=130,
                bbox_inches="tight",
            )
            plt.close(fig)

            print(
                f"  MS MAE={ms_metrics['mae']:.4f}, "
                f"RMSE={ms_metrics['rmse']:.4f}, "
                f"r={ms_metrics['spatial_pearson']:.4f}"
            )
            print(
                f"  Baseline MAE={base_metrics['mae']:.4f}, "
                f"RMSE={base_metrics['rmse']:.4f}, "
                f"r={base_metrics['spatial_pearson']:.4f}"
            )

    finally:
        ms_extractor.close()
        base_extractor.close()

    if not rows:
        raise RuntimeError("No images were successfully evaluated.")

    csv_path = OUT / "per_image_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def aggregate(prefix):
        result = {}
        for metric in ["mae", "rmse", "spatial_pearson", "mean_bias"]:
            values = np.array(
                [r[f"{prefix}_{metric}"] for r in rows],
                dtype=float,
            )
            valid = values[np.isfinite(values)]
            result[metric + "_mean"] = (
                float(valid.mean()) if len(valid) else None
            )
            result[metric + "_std"] = (
                float(valid.std()) if len(valid) else None
            )
        return result

    summary = {
        "dataset": "RUGD sample",
        "images_evaluated": len(rows),
        "images_missing_teacher": len(image_paths) - len(rows),
        "teacher_maps": str(TEACHER_ROOT),
        "student_output_size": list(MAP_SIZE),
        "teacher_resize": "bilinear",
        "multiscale": aggregate("multiscale"),
        "baseline": aggregate("baseline"),
        "elapsed_seconds": round(time.time() - start, 2),
        "interpretation": (
            "Metrics measure agreement with SAM-TP pseudo-labels, "
            "not physical traversability. Spatial Pearson correlation "
            "measures pattern similarity, not score calibration."
        ),
    }

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n===== AGGREGATE RESULTS =====")
    print(json.dumps(summary, indent=2))
    print("\nCSV:", csv_path)
    print("Summary:", summary_path)
    print("Visualizations:", VIS)


if __name__ == "__main__":
    main()
