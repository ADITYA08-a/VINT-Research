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
IMAGE_PATH = Path.home() / "Downloads/tes_image.jpeg"

VINT_CHECKPOINT = REPO / "deployment/model_weights/vint.pth"
CONFIG = REPO / "train/config/vint.yaml"

EXPERIMENT = (
    REPO / "experiments/traversability/"
    "multiscale_full_features_1000"
)
DECODER_CHECKPOINT = EXPERIMENT / "best_decoder.pt"
OUTPUT_DIR = EXPERIMENT / "internet_image_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "tes_image_prediction.png"
OUTPUT_ARRAY = OUTPUT_DIR / "tes_image_prediction.npy"

for path in [IMAGE_PATH, VINT_CHECKPOINT, CONFIG, DECODER_CHECKPOINT]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(REPO / "train"))
from vint_train.models.vint.vint import ViNT

# Load the training module to reuse the exact trained architecture.
script_path = EXPERIMENT.parent / "train_multiscale_full.py"
spec = importlib.util.spec_from_file_location(
    "train_multiscale_full_inference", script_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

# Load the pretrained ViNT model.
vint_checkpoint = torch.load(
    VINT_CHECKPOINT, map_location="cpu", weights_only=False
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
vint.load_state_dict(vint_checkpoint["model"].state_dict(), strict=True)
vint.to(DEVICE).eval()

# Load the trained full-feature decoder.
checkpoint = torch.load(
    DECODER_CHECKPOINT, map_location=DEVICE, weights_only=False
)
decoder = module.MultiScaleDecoder().to(DEVICE)
decoder.load_state_dict(checkpoint["decoder"], strict=True)
decoder.eval()

# Match the training preprocessing.
image = Image.open(IMAGE_PATH).convert("RGB")
original_size = image.size

transform = transforms.Compose([
    transforms.Resize((64, 85)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Run dense prediction.
extractor = module.FeatureExtractor(vint.obs_encoder)

try:
    with torch.no_grad():
        features = extractor(image_tensor)
        logits = decoder(features, output_size=(120, 160))
        prediction = torch.sigmoid(logits)[0, 0].cpu().numpy()
finally:
    extractor.close()

# Save numerical predictions.
np.save(OUTPUT_ARRAY, prediction)

# Resize the original image to align with the predicted map.
rgb = np.asarray(image.resize((160, 120), Image.Resampling.LANCZOS))

# Generate a visual report.
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(rgb)
axes[0].set_title("Input image")
axes[0].axis("off")

heatmap = axes[1].imshow(
    prediction, cmap="viridis", vmin=0, vmax=1
)
axes[1].set_title("Predicted traversability score")
axes[1].axis("off")
fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(rgb)
overlay = axes[2].imshow(
    prediction, cmap="viridis", vmin=0, vmax=1, alpha=0.55
)
axes[2].set_title("Prediction overlay")
axes[2].axis("off")

fig.suptitle("ViNT + Full-Multiscale Traversability Decoder")
fig.tight_layout()
fig.savefig(OUTPUT_IMAGE, dpi=180, bbox_inches="tight")
plt.close(fig)

print("\nINFERENCE COMPLETE")
print("Input:", IMAGE_PATH)
print("Original dimensions:", original_size)
print("Prediction dimensions:", prediction.shape)
print("Score range: [{:.4f}, {:.4f}]".format(
    prediction.min(), prediction.max()
))
print("Mean score: {:.4f}".format(prediction.mean()))
print("Saved visualization:", OUTPUT_IMAGE)
print("Saved prediction array:", OUTPUT_ARRAY)
