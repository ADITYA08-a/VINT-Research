import json
import random
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

REPO = Path.home() / "visualnav-transformer"
DATA_ROOT = Path.home() / "Downloads/go_stanford"

LABEL_ROOT = (
    Path.home() / "GENIE-SAMTP/experiments/heldout_labels_100"
)

CONFIG = REPO / "train/config/vint.yaml"
VINT_CHECKPOINT = REPO / "deployment/model_weights/vint.pth"

MULTISCALE_CHECKPOINT = (
    REPO / "experiments/traversability/"
    "multiscale_v2_1000/best_decoder.pt"
)

BASELINE_CHECKPOINT = (
    REPO / "experiments/traversability/"
    "final_feature_v2_1000/best_decoder.pt"
)

MANIFEST = LABEL_ROOT / "manifest.json"

OUTPUT = (
    REPO / "experiments/traversability/"
    "go_stanford_heldout_visualizations"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 24
SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(REPO / "train"))

from vint_train.models.vint.vint import ViNT
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multi_module = load_module(
    "train_multiscale_v2",
    REPO / "experiments/traversability/train_multiscale_v2.py",
)

base_module = load_module(
    "train_final_baseline_v2",
    REPO / "experiments/traversability/train_final_baseline_v2.py",
)

MultiScaleDecoder = multi_module.MultiScaleDecoder
FinalFeatureDecoder = base_module.FinalFeatureDecoder


def resize_target(target, shape):
    """Resize a score map without changing its [0,1] value range."""
    if target.shape == shape:
        return target.astype(np.float32)

    resized = Image.fromarray(target.astype(np.float32), mode="F")
    resized = resized.resize(
        (shape[1], shape[0]),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    return torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )


# ---------------------------------------------------------
# Validate paths and manifest
# ---------------------------------------------------------

for path in (
    DATA_ROOT,
    CONFIG,
    VINT_CHECKPOINT,
    MULTISCALE_CHECKPOINT,
    BASELINE_CHECKPOINT,
    MANIFEST,
):
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

manifest = json.loads(MANIFEST.read_text())
labelled = manifest["samples"]

if len(labelled) == 0:
    raise RuntimeError("The held-out label manifest contains no samples.")

# Make sure these trajectories do not overlap the student-training data.
training_manifest_path = (
    Path.home() / "GENIE-SAMTP/experiments/dataset_v2_1000/manifest.json"
)
training_manifest = json.loads(training_manifest_path.read_text())

training_trajectories = set(training_manifest["selected_trajectories"])
heldout_trajectories = set(manifest["selected_trajectories"])

overlap = training_trajectories.intersection(heldout_trajectories)

if overlap:
    raise RuntimeError(
        f"Data leakage: {len(overlap)} trajectories occur in both datasets."
    )

# Deterministically sample the teacher-labelled held-out images.
rng = random.Random(SEED)
samples = list(labelled)
rng.shuffle(samples)
samples = samples[:min(NUM_IMAGES, len(samples))]

print("Device:", DEVICE)
print("Held-out labelled images available:", len(labelled))
print("Selected images for visualization:", len(samples))
print("Training/held-out trajectory overlap:", len(overlap))
print("Output directory:", OUTPUT)


# ---------------------------------------------------------
# Load pretrained ViNT
# ---------------------------------------------------------

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

checkpoint = torch.load(
    VINT_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

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

vint.load_state_dict(checkpoint["model"].state_dict(), strict=True)
vint.to(DEVICE).eval()

for parameter in vint.parameters():
    parameter.requires_grad = False


# ---------------------------------------------------------
# Load traversability decoders
# ---------------------------------------------------------

multi_saved = load_checkpoint(MULTISCALE_CHECKPOINT)
multiscale = MultiScaleDecoder().to(DEVICE)
multiscale.load_state_dict(multi_saved["decoder"], strict=True)
multiscale.eval()

base_saved = load_checkpoint(BASELINE_CHECKPOINT)
baseline = FinalFeatureDecoder().to(DEVICE)
baseline.load_state_dict(base_saved["decoder"], strict=True)
baseline.eval()


# ---------------------------------------------------------
# Extract features from the frozen ViNT encoder
# ---------------------------------------------------------

class MultiFeatureExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.features = {}
        self.handles = []

        for name, block_index in {
            "block_1": 1,
            "block_3": 3,
            "block_5": 5,
        }.items():
            handle = encoder._blocks[block_index].register_forward_hook(
                self.make_hook(name)
            )
            self.handles.append(handle)

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
        return {
            "final": self.encoder.extract_features(image).detach()
        }

    def close(self):
        pass


multi_features = MultiFeatureExtractor(vint.obs_encoder)
base_features = FinalFeatureExtractor(vint.obs_encoder)

transform = transforms.Compose([
    transforms.Resize((64, 85)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------
# Inference and quantitative evaluation
# ---------------------------------------------------------

summary = []

try:
    for index, item in enumerate(samples, start=1):
        relative = item["image"]
        image_path = DATA_ROOT / relative
        label_path = LABEL_ROOT / item["label"]

        if not image_path.is_file():
            print("Skipping missing image:", image_path)
            continue

        if not label_path.is_file():
            print("Skipping missing label:", label_path)
            continue

        with Image.open(image_path) as im:
            rgb_image = im.convert("RGB")
            rgb = np.asarray(rgb_image)

        target = np.load(label_path).astype(np.float32)
        target = resize_target(target, rgb.shape[:2])

        input_tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features_multi = multi_features(input_tensor)
            logits_multi = multiscale(
                features_multi,
                output_size=rgb.shape[:2],
            )
            prediction_multi = (
                torch.sigmoid(logits_multi)[0, 0].cpu().numpy()
            )

            features_base = base_features(input_tensor)
            logits_base = baseline(
                features_base,
                output_size=rgb.shape[:2],
            )
            prediction_base = (
                torch.sigmoid(logits_base)[0, 0].cpu().numpy()
            )

        multi_error = prediction_multi - target
        base_error = prediction_base - target

        multi_mae = float(np.abs(multi_error).mean())
        base_mae = float(np.abs(base_error).mean())
        multi_rmse = float(np.sqrt(np.mean(multi_error ** 2)))
        base_rmse = float(np.sqrt(np.mean(base_error ** 2)))

        # Consistent colour limits make predictions visually comparable.
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(rgb)
        axes[0].set_title("Go Stanford RGB")

        axes[1].imshow(target, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("SAM-TP reference")

        axes[2].imshow(
            prediction_multi, cmap="viridis", vmin=0, vmax=1
        )
        axes[2].set_title(f"Multiscale | MAE {multi_mae:.3f}")

        axes[3].imshow(
            prediction_base, cmap="viridis", vmin=0, vmax=1
        )
        axes[3].set_title(f"Final-feature | MAE {base_mae:.3f}")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Held-out trajectory: {item['trajectory']} | {relative}"
        )
        fig.tight_layout()

        output_file = OUTPUT / (
            f"{index:03d}_{item['sample_id']}.png"
        )
        fig.savefig(output_file, dpi=140, bbox_inches="tight")
        plt.close(fig)

        summary.append({
            "trajectory": item["trajectory"],
            "image": relative,
            "reference_mean": float(target.mean()),
            "multiscale_mean": float(prediction_multi.mean()),
            "baseline_mean": float(prediction_base.mean()),
            "multiscale_mae": multi_mae,
            "baseline_mae": base_mae,
            "multiscale_rmse": multi_rmse,
            "baseline_rmse": base_rmse,
            "visualization": output_file.name,
        })

        print(
            f"[{index}/{len(samples)}] {relative} | "
            f"multiscale MAE={multi_mae:.4f}, RMSE={multi_rmse:.4f} | "
            f"baseline MAE={base_mae:.4f}, RMSE={base_rmse:.4f}"
        )

finally:
    multi_features.close()
    base_features.close()


# Save per-image results.
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))

# Save a compact CSV for subsequent analysis.
if summary:
    with (OUTPUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nHeld-out results across visualized images:")
    print(
        "Multiscale mean MAE:",
        f"{np.mean([x['multiscale_mae'] for x in summary]):.5f}",
    )
    print(
        "Baseline mean MAE:",
        f"{np.mean([x['baseline_mae'] for x in summary]):.5f}",
    )
    print(
        "Multiscale mean RMSE:",
        f"{np.mean([x['multiscale_rmse'] for x in summary]):.5f}",
    )
    print(
        "Baseline mean RMSE:",
        f"{np.mean([x['baseline_rmse'] for x in summary]):.5f}",
    )

print("\nSaved visualizations to:", OUTPUT)
print("Summary JSON:", OUTPUT / "summary.json")
print("Summary CSV:", OUTPUT / "summary.csv")
