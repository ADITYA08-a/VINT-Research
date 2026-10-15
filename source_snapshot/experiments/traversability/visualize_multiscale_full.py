import json
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
DATA_ROOT = Path.home() / "Downloads/go_stanford"
LABEL_ROOT = Path.home() / "GENIE-SAMTP/experiments/heldout_labels_100"

VINT_CHECKPOINT = REPO / "deployment/model_weights/vint.pth"
CONFIG = REPO / "train/config/vint.yaml"

NEW_CHECKPOINT = (
    REPO / "experiments/traversability/"
    "multiscale_full_features_1000/best_decoder.pt"
)
OLD_CHECKPOINT = (
    REPO / "experiments/traversability/"
    "multiscale_v2_1000/best_decoder.pt"
)

OUTPUT = (
    REPO / "experiments/traversability/"
    "multiscale_full_features_1000/visualizations"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_IMAGES = 24

sys.path.insert(0, str(REPO / "train"))
from vint_train.models.vint.vint import ViNT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_module = load_module(
    "train_multiscale_v2",
    REPO / "experiments/traversability/train_multiscale_v2.py",
)
new_module = load_module(
    "train_multiscale_full",
    REPO / "experiments/traversability/train_multiscale_full.py",
)

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

vint_checkpoint = torch.load(
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
    vint_checkpoint["model"].state_dict(), strict=True
)
model.to(DEVICE).eval()

old_data = torch.load(
    OLD_CHECKPOINT, map_location=DEVICE, weights_only=False
)
new_data = torch.load(
    NEW_CHECKPOINT, map_location=DEVICE, weights_only=False
)

old_decoder = old_module.MultiScaleDecoder().to(DEVICE)
old_decoder.load_state_dict(old_data["decoder"], strict=True)
old_decoder.eval()

new_decoder = new_module.MultiScaleDecoder().to(DEVICE)
new_decoder.load_state_dict(new_data["decoder"], strict=True)
new_decoder.eval()

old_extractor = old_module.FeatureExtractor(model.obs_encoder)
new_extractor = new_module.FeatureExtractor(model.obs_encoder)

transform = transforms.Compose([
    transforms.Resize((64, 85)),
    transforms.ToTensor(),
])

manifest_path = LABEL_ROOT / "manifest.json"
manifest = json.loads(manifest_path.read_text())
samples = manifest["samples"][:NUM_IMAGES]

print("Device:", DEVICE)
print("Images to visualize:", len(samples))
print("Teacher manifest:", manifest_path)

try:
    for i, item in enumerate(samples, 1):
        image_path = DATA_ROOT / item["image"]
        label_path = LABEL_ROOT / item["label"]

        if not image_path.exists() or not label_path.exists():
            print("Skipping missing file:", item["sample_id"])
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            old_features = old_extractor(image_tensor)
            old_logits = old_decoder(
                old_features, output_size=(120, 160)
            )
            old_prediction = torch.sigmoid(
                old_logits
            )[0, 0].cpu().numpy()

            new_features = new_extractor(image_tensor)
            new_logits = new_decoder(
                new_features, output_size=(120, 160)
            )
            new_prediction = torch.sigmoid(
                new_logits
            )[0, 0].cpu().numpy()

        target = np.load(label_path).astype(np.float32)
        target = np.squeeze(target)

        if target.shape != (120, 160):
            target = np.asarray(
                Image.fromarray(target).resize(
                    (160, 120), Image.Resampling.BILINEAR
                )
            )

        rgb = np.asarray(image.resize((160, 120)))
        new_error = np.abs(new_prediction - target)
        old_error = np.abs(old_prediction - target)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        panels = [
            (rgb, "RGB input", None),
            (target, "SAM-TP teacher", "viridis"),
            (new_prediction, "Full-feature prediction", "viridis"),
            (old_prediction, "Previous multiscale prediction", "viridis"),
            (new_error, "Full-feature absolute error", "magma"),
            (old_error, "Previous multiscale absolute error", "magma"),
        ]

        for ax, (data, title, cmap) in zip(axes.flat, panels):
            if cmap is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"{item['sample_id']} | Held-out comparison",
            fontsize=14,
        )
        fig.tight_layout()

        output_path = OUTPUT / f"{item['sample_id']}_comparison.png"
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print(
            f"[{i}/{len(samples)}] {output_path.name} | "
            f"Full MAE={new_error.mean():.4f} | "
            f"Previous MAE={old_error.mean():.4f}"
        )

finally:
    old_extractor.close()
    new_extractor.close()

print("\nAll available comparison images saved to:")
print(OUTPUT)
