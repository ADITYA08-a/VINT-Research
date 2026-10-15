import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from PIL import Image
from torchvision import transforms

REPO = Path.home() / "visualnav-transformer"
DATA_ROOT = Path.home() / "Downloads/go_stanford"
LABEL_ROOT = Path.home() / "GENIE-SAMTP/experiments/pilot_dataset"

EXPERIMENT = REPO / "experiments/traversability/multiscale_pilot"

sys.path.insert(0, str(REPO / "train"))

from vint_train.models.vint.vint import ViNT

# Load the same feature extractor and decoder used in training.
import importlib.util

script = REPO / "experiments/traversability/train_multiscale.py"
spec = importlib.util.spec_from_file_location("train_multiscale", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

MultiScaleDecoder = module.MultiScaleDecoder
FeatureExtractor = module.FeatureExtractor

with open(REPO / "train/config/vint.yaml") as f:
    cfg = yaml.safe_load(f)

checkpoint = torch.load(
    REPO / "deployment/model_weights/vint.pth",
    map_location="cpu",
    weights_only=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
vint.to(device).eval()

saved = torch.load(
    EXPERIMENT / "best_decoder.pt",
    map_location=device,
    weights_only=False,
)

decoder = MultiScaleDecoder().to(device)
decoder.load_state_dict(saved["decoder"])
decoder.eval()

extractor = FeatureExtractor(vint.obs_encoder)

transform = transforms.Compose([
    transforms.Resize((64, 85)),
    transforms.ToTensor(),
])

manifest = json.loads(
    (LABEL_ROOT / "manifest.json").read_text()
)

records = []

try:
    for item in manifest["samples"]:
        image_path = DATA_ROOT / item["image"]
        label_path = LABEL_ROOT / item["label"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        target = np.load(label_path).astype(np.float32)

        with torch.no_grad():
            features = extractor(image_tensor)
            logits = decoder(features, output_size=target.shape)
            prediction = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mae = np.abs(prediction - target).mean()
        rmse = np.sqrt(np.square(prediction - target).mean())

        records.append({
            "name": item["sample_id"],
            "target_mean": float(target.mean()),
            "prediction_mean": float(prediction.mean()),
            "target_std": float(target.std()),
            "prediction_std": float(prediction.std()),
            "MAE": float(mae),
            "RMSE": float(rmse),
        })

    print("\nPer-image evaluation")
    print("-" * 100)

    for r in records:
        print(
            f"{r['name']:<25} "
            f"target={r['target_mean']:.3f} "
            f"pred={r['prediction_mean']:.3f} "
            f"MAE={r['MAE']:.3f} "
            f"RMSE={r['RMSE']:.3f}"
        )

    print("\nAggregate evaluation")
    print("-" * 40)

    for key in ("MAE", "RMSE"):
        print(
            f"Mean image-level {key}: "
            f"{np.mean([r[key] for r in records]):.5f}"
        )

    target_means = np.array([r["target_mean"] for r in records])
    pred_means = np.array([r["prediction_mean"] for r in records])

    if target_means.std() > 1e-8 and pred_means.std() > 1e-8:
        corr = np.corrcoef(target_means, pred_means)[0, 1]
        print(f"Correlation of image-level means: {corr:.5f}")
    else:
        print("Correlation undefined: near-constant image-level means.")

finally:
    extractor.close()
