import sys
import csv
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
RUGD_ROOT = Path.home() / "Downloads/RUGD_sample-data"
IMAGE_ROOT = RUGD_ROOT / "images"
ANNOTATION_ROOT = RUGD_ROOT / "annotations"

VINT_CHECKPOINT = REPO / "deployment/model_weights/vint.pth"
VINT_CONFIG = REPO / "train/config/vint.yaml"

MULTISCALE_SCRIPT = (
    REPO / "experiments/traversability/train_multiscale_v2.py"
)
BASELINE_SCRIPT = (
    REPO / "experiments/traversability/train_final_baseline_v2.py"
)

MULTISCALE_CHECKPOINT = (
    REPO / "experiments/traversability/multiscale_v2_1000/best_decoder.pt"
)
BASELINE_CHECKPOINT = (
    REPO / "experiments/traversability/final_feature_v2_1000/best_decoder.pt"
)

OUTPUT_ROOT = (
    REPO / "experiments/traversability/rugd_sample_evaluation"
)
VIS_DIR = OUTPUT_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAP_SIZE = (120, 160)  # height, width

sys.path.insert(0, str(REPO / "train"))
from vint_train.models.vint.vint import ViNT


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_vint():
    with open(VINT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    checkpoint = torch.load(
        VINT_CHECKPOINT, map_location="cpu", weights_only=False
    )

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
    model.load_state_dict(
        checkpoint["model"].state_dict(), strict=True
    )
    return model.to(DEVICE).eval()


def load_decoder(module, checkpoint_path, decoder_class):
    decoder = decoder_class().to(DEVICE)

    state = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )

    # Training saves either a wrapped checkpoint or a raw state_dict.
    if isinstance(state, dict) and "decoder" in state:
        state = state["decoder"]

    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    return decoder


def overlay_error(ax, error, title):
    im = ax.imshow(error, cmap="magma", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    return im


def main():
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(IMAGE_ROOT)

    for path in [
        VINT_CHECKPOINT,
        MULTISCALE_CHECKPOINT,
        BASELINE_CHECKPOINT,
        MULTISCALE_SCRIPT,
        BASELINE_SCRIPT,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    ms_module = import_module("train_multiscale_v2", MULTISCALE_SCRIPT)
    base_module = import_module("train_final_baseline_v2", BASELINE_SCRIPT)

    model = load_vint()
    ms_decoder = load_decoder(
        ms_module, MULTISCALE_CHECKPOINT, ms_module.MultiScaleDecoder
    )
    base_decoder = load_decoder(
        base_module, BASELINE_CHECKPOINT, base_module.FinalFeatureDecoder
    )

    ms_extractor = ms_module.FeatureExtractor(model.obs_encoder)
    base_extractor = base_module.FeatureExtractor(model.obs_encoder)

    transform = transforms.Compose([
        transforms.Resize((64, 85)),
        transforms.ToTensor(),
    ])

    image_paths = sorted(IMAGE_ROOT.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {IMAGE_ROOT}")

    rows = []

    print(f"Device: {DEVICE}")
    print(f"Images: {len(image_paths)}")
    print(f"Output: {OUTPUT_ROOT}")

    try:
        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{index}/{len(image_paths)}] {image_path.name}")

            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                original = np.asarray(rgb)
                image_tensor = transform(rgb).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                ms_features = ms_extractor(image_tensor)
                ms_logits = ms_decoder(
                    ms_features, output_size=MAP_SIZE
                )
                ms_prediction = torch.sigmoid(ms_logits)[0, 0].cpu().numpy()

                base_features = base_extractor(image_tensor)
                base_logits = base_decoder(
                    base_features, output_size=MAP_SIZE
                )
                base_prediction = (
                    torch.sigmoid(base_logits)[0, 0].cpu().numpy()
                )

            annotation_path = ANNOTATION_ROOT / image_path.name
            annotation = None

            if annotation_path.exists():
                with Image.open(annotation_path) as ann:
                    annotation = np.asarray(ann.convert("RGB"))

            # Teacher inference is performed separately in the SAM-TP
            # environment. Do not substitute RUGD semantic labels for it.
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))

            axes[0, 0].imshow(original)
            axes[0, 0].set_title("RUGD RGB")

            if annotation is not None:
                axes[0, 1].imshow(annotation)
                axes[0, 1].set_title("RUGD semantic annotation")
            else:
                axes[0, 1].text(
                    0.5, 0.5, "Annotation unavailable",
                    ha="center", va="center"
                )
                axes[0, 1].set_title("RUGD annotation")

            axes[0, 2].imshow(
                ms_prediction, cmap="viridis", vmin=0, vmax=1
            )
            axes[0, 2].set_title("ViNT multiscale prediction")

            axes[1, 0].imshow(
                base_prediction, cmap="viridis", vmin=0, vmax=1
            )
            axes[1, 0].set_title("Final-feature baseline")

            axes[1, 1].imshow(
                ms_prediction - base_prediction,
                cmap="coolwarm", vmin=-1, vmax=1
            )
            axes[1, 1].set_title("Multiscale − baseline")

            axes[1, 2].imshow(
                np.abs(ms_prediction - base_prediction),
                cmap="magma", vmin=0, vmax=1
            )
            axes[1, 2].set_title("Absolute model disagreement")

            for ax in axes.flat:
                ax.axis("off")

            fig.suptitle(
                f"{image_path.name} — cross-dataset inference",
                fontsize=13
            )
            fig.tight_layout()

            output_path = VIS_DIR / f"{image_path.stem}_comparison.png"
            fig.savefig(output_path, dpi=130, bbox_inches="tight")
            plt.close(fig)

            row = {
                "image": image_path.name,
                "multiscale_mean": float(ms_prediction.mean()),
                "multiscale_std": float(ms_prediction.std()),
                "baseline_mean": float(base_prediction.mean()),
                "baseline_std": float(base_prediction.std()),
                "mean_absolute_disagreement": float(
                    np.abs(ms_prediction - base_prediction).mean()
                ),
                "visualization": str(output_path),
            }
            rows.append(row)

            print(
                f"  multiscale mean={row['multiscale_mean']:.4f}, "
                f"baseline mean={row['baseline_mean']:.4f}, "
                f"disagreement={row['mean_absolute_disagreement']:.4f}"
            )

    finally:
        ms_extractor.close()
        base_extractor.close()

    csv_path = OUTPUT_ROOT / "per_image_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "RUGD sample",
        "image_count": len(rows),
        "teacher_maps_generated": False,
        "teacher_metrics_available": False,
        "note": (
            "This first pass compares the two decoder predictions on RUGD. "
            "SAM-TP teacher inference has not been run. RUGD semantic "
            "annotations are not ground-truth traversability scores."
        ),
        "mean_multiscale_score": float(
            np.mean([r["multiscale_mean"] for r in rows])
        ),
        "mean_baseline_score": float(
            np.mean([r["baseline_mean"] for r in rows])
        ),
        "mean_absolute_model_disagreement": float(
            np.mean([r["mean_absolute_disagreement"] for r in rows])
        ),
    }

    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Build a contact sheet from the individual comparison panels.
    from PIL import Image as PILImage, ImageOps, ImageDraw

    thumbs = []
    for row in rows:
        with PILImage.open(row["visualization"]) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((480, 300))
            canvas = PILImage.new("RGB", (500, 330), "white")
            canvas.paste(thumb, ((500 - thumb.width) // 2, 5))
            ImageDraw.Draw(canvas).text(
                (10, 310), row["image"], fill="black"
            )
            thumbs.append(canvas)

    cols = 2
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = PILImage.new(
        "RGB", (cols * 500, rows_count * 330), (225, 225, 225)
    )
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 500, (i // cols) * 330))

    sheet.save(OUTPUT_ROOT / "contact_sheet.png")

    print("\nEvaluation complete.")
    print("Visualizations:", VIS_DIR)
    print("Contact sheet:", OUTPUT_ROOT / "contact_sheet.png")
    print("CSV:", csv_path)
    print("Summary:", OUTPUT_ROOT / "summary.json")
    print("Note: no SAM-TP teacher maps or teacher-error metrics were generated.")


if __name__ == "__main__":
    main()
