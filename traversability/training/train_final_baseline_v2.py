import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REPO = Path.home() / "visualnav-transformer"
DATA_ROOT = Path.home() / "Downloads/go_stanford"
LABEL_ROOT = Path.home() / "GENIE-SAMTP/experiments/dataset_v2_1000"
MANIFEST = LABEL_ROOT / "manifest.json"

CHECKPOINT = REPO / "deployment/model_weights/vint.pth"
CONFIG = REPO / "train/config/vint.yaml"

OUTPUT_DIR = REPO / "experiments/traversability/final_feature_v2_1000"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH = 85
IMAGE_HEIGHT = 64
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------
# Reconstruct the original ViNT model
# ---------------------------------------------------------

sys.path.insert(0, str(REPO / "train"))

from vint_train.models.vint.vint import ViNT


def load_vint():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
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
        checkpoint["model"].state_dict(),
        strict=True,
    )

    model.to(DEVICE).eval()

    # Freeze the pretrained encoder.
    for parameter in model.obs_encoder.parameters():
        parameter.requires_grad = False

    return model


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class TraversabilityDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        image_path = item["image_path"]
        label_path = item["label_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = np.load(label_path).astype(np.float32)

        if label.ndim != 2:
            raise ValueError(
                f"Expected a 2D score map, got {label.shape}: {label_path}"
            )

        if not np.isfinite(label).all():
            raise ValueError(f"Non-finite values in {label_path}")

        # SAM-TP labels should already be sigmoid-transformed.
        if label.min() < 0 or label.max() > 1:
            raise ValueError(
                f"Expected scores in [0, 1], got "
                f"[{label.min():.4f}, {label.max():.4f}]"
            )

        label = torch.from_numpy(label).unsqueeze(0)

        return image, label


def resolve_samples():
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["samples"]

    samples = []

    for item in entries:
        image_rel = Path(item["image"])
        label_rel = Path(item["label"])

        image_path = (
            image_rel if image_rel.is_absolute()
            else DATA_ROOT / image_rel
        )

        label_path = (
            label_rel if label_rel.is_absolute()
            else LABEL_ROOT / label_rel
        )

        if not label_path.exists():
            # Also support manifests storing only the label filename.
            candidates = [
                LABEL_ROOT / "labels" / label_rel.name,
                LABEL_ROOT / label_rel.name,
            ]
            label_path = next(
                (p for p in candidates if p.exists()),
                label_path,
            )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        # Group frames by trajectory to prevent frame-level leakage.
        trajectory = image_rel.parts[0] if len(image_rel.parts) > 1 else str(
            image_path.parent
        )

        samples.append({
            "image_path": image_path,
            "label_path": label_path,
            "trajectory": trajectory,
        })

    if len(samples) < 2:
        raise RuntimeError("Need at least two labelled images.")

    return samples


def split_by_trajectory(samples):
    trajectories = sorted({
        sample["trajectory"] for sample in samples
    })

    if len(trajectories) < 2:
        raise RuntimeError(
            "Only one trajectory is represented. "
            "A trajectory-held-out validation split is impossible."
        )

    rng = random.Random(SEED)
    rng.shuffle(trajectories)

    n_val = max(1, round(0.2 * len(trajectories)))
    val_trajectories = set(trajectories[:n_val])

    train_samples = [
        s for s in samples if s["trajectory"] not in val_trajectories
    ]
    val_samples = [
        s for s in samples if s["trajectory"] in val_trajectories
    ]

    if not train_samples or not val_samples:
        raise RuntimeError("Empty train or validation split.")

    print("Trajectories:", len(trajectories))
    print("Training trajectories:", len(trajectories) - n_val)
    print("Validation trajectories:", n_val)
    print("Training samples:", len(train_samples))
    print("Validation samples:", len(val_samples))

    return train_samples, val_samples


# ---------------------------------------------------------
# Multiscale decoder
# ---------------------------------------------------------

class FinalFeatureDecoder(nn.Module):
    """Baseline using only EfficientNet's final 1280-channel feature map."""

    def __init__(self, in_channels=1280):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, features, output_size):
        x = features["final"]

        # Upsample the 2x2 final feature map to 32x32.
        for _ in range(4):
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )

        x = self.decoder(x)

        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class FeatureExtractor:
    """Extract only the final spatial feature map."""

    def __init__(self, encoder):
        self.encoder = encoder

    @torch.no_grad()
    def __call__(self, image):
        return {
            "final": self.encoder.extract_features(image).detach()
        }

    def close(self):
        pass


# ---------------------------------------------------------
# Training and validation
# ---------------------------------------------------------

def run_epoch(extractor, decoder, loader, optimizer=None):
    training = optimizer is not None
    decoder.train(training)

    total_loss = 0.0
    total_pixels = 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # The pretrained encoder stays frozen.
        features = extractor(images)

        with torch.set_grad_enabled(training):
            logits = decoder(
                features,
                output_size=targets.shape[-2:],
            )

            predictions = torch.sigmoid(logits)

            loss = F.smooth_l1_loss(predictions, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        pixels = targets.numel()
        total_loss += loss.item() * pixels
        total_pixels += pixels

    return total_loss / max(total_pixels, 1)


def main():
    samples = resolve_samples()

    train_samples, val_samples = split_by_trajectory(samples)

    train_loader = DataLoader(
        TraversabilityDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        TraversabilityDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_vint()
    encoder = model.obs_encoder

    extractor = FeatureExtractor(encoder)
    decoder = FinalFeatureDecoder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss = run_epoch(
                extractor, decoder, train_loader, optimizer
            )

            with torch.no_grad():
                val_loss = run_epoch(
                    extractor, decoder, val_loader
                )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train SmoothL1: {train_loss:.6f} | "
                f"val SmoothL1: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    "decoder": decoder.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "feature_names": [
                        "block_1", "block_3", "block_5", "final"
                    ],
                    "input_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                    "target_type": "SAM-TP sigmoid pseudo-label",
                }, OUTPUT_DIR / "best_decoder.pt")

        torch.save(
            decoder.state_dict(),
            OUTPUT_DIR / "last_decoder_state.pt",
        )

        print(f"\nBest validation loss: {best_val_loss:.6f}")
        print(f"Outputs saved in: {OUTPUT_DIR}")

    finally:
        extractor.close()


if __name__ == "__main__":
    main()
