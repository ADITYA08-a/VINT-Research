import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
DATA_ROOT = Path.home() / "Downloads/go_stanford"
LABEL_ROOT = Path.home() / "GENIE-SAMTP/experiments/pilot_dataset"

CHECKPOINT = REPO / "deployment/model_weights/vint.pth"
CONFIG = REPO / "train/config/vint.yaml"

EXPERIMENT = REPO / "experiments/traversability/multiscale_pilot"
DECODER_CHECKPOINT = EXPERIMENT / "best_decoder.pt"
MANIFEST = LABEL_ROOT / "manifest.json"

OUTPUT = EXPERIMENT / "visualizations"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(REPO / "train"))

from vint_train.models.vint.vint import ViNT
import importlib.util

training_script = (
    REPO / "experiments/traversability/train_multiscale.py"
)

spec = importlib.util.spec_from_file_location(
    "train_multiscale", training_script
)
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

MultiScaleDecoder = training_module.MultiScaleDecoder
FeatureExtractor = training_module.FeatureExtractor

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

checkpoint = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=False
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

model.load_state_dict(checkpoint["model"].state_dict(), strict=True)
model.to(DEVICE).eval()

decoder_checkpoint = torch.load(
    DECODER_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

decoder = MultiScaleDecoder().to(DEVICE)
decoder.load_state_dict(decoder_checkpoint["decoder"])
decoder.eval()

extractor = FeatureExtractor(model.obs_encoder)

transform = transforms.Compose([
    transforms.Resize((64, 85)),
    transforms.ToTensor(),
])

manifest = json.loads(MANIFEST.read_text())
samples = manifest["samples"]

try:
    for index, item in enumerate(samples):
        image_path = DATA_ROOT / item["image"]
        label_path = LABEL_ROOT / item["label"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = extractor(image_tensor)
            logits = decoder(
                features,
                output_size=(120, 160),
            )
            prediction = torch.sigmoid(logits)[0, 0].cpu().numpy()

        target = np.load(label_path)

        original = np.asarray(
            Image.open(image_path).convert("RGB").resize((160, 120))
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(original)
        axes[0].set_title("RGB input")

        axes[1].imshow(target, vmin=0, vmax=1, cmap="viridis")
        axes[1].set_title("SAM-TP pseudo-label")

        axes[2].imshow(prediction, vmin=0, vmax=1, cmap="viridis")
        axes[2].set_title("ViNT multiscale prediction")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()

        output_path = OUTPUT / f"{item['sample_id']}.png"
        fig.savefig(output_path, dpi=140)
        plt.close(fig)

        print(
            f"{output_path.name}: "
            f"target mean={target.mean():.4f}, "
            f"prediction mean={prediction.mean():.4f}, "
            f"prediction std={prediction.std():.4f}"
        )

finally:
    extractor.close()

print(f"\nVisualizations saved to: {OUTPUT}")
