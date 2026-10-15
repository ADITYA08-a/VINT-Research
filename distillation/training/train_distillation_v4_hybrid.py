from pathlib import Path
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from vint_train.data.data_utils import img_path_to_data
from experiments.distillation.student_v4_hybrid import FastViTStudentV4Hybrid


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path.home() / "Downloads/go_stanford"
SPLITS = Path.home() / "Downloads/vint_data_splits/go_stanford"
TEACHER_PATH = Path("deployment/model_weights/vint.pth")
OUT = Path("experiments/distillation/results/training_v4_hybrid")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 8

MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES = 1000

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
DISTANCE_WEIGHT = 0.01

NUM_WORKERS = 0
IMAGE_SIZE = (85, 64)
CONTEXT_FRAMES = 6
GOAL_OFFSET = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class ImagePairDataset(Dataset):
    """Six consecutive observations and one future goal image."""

    def __init__(self, trajectory_names, max_samples, seed):
        pairs = []

        for name in sorted(trajectory_names):
            folder = ROOT / name
            if not folder.is_dir():
                continue

            frame_ids = sorted(
                int(p.stem)
                for p in folder.glob("*.jpg")
                if p.stem.isdigit()
            )
            available = set(frame_ids)

            for t in frame_ids:
                context = list(range(t - CONTEXT_FRAMES + 1, t + 1))
                goal_t = t + GOAL_OFFSET

                if not all(i in available for i in context):
                    continue
                if goal_t not in available:
                    continue

                pairs.append((name, context, goal_t))

        candidate_count = len(pairs)
        random.Random(seed).shuffle(pairs)
        self.pairs = pairs[:max_samples]

        print(
            f"Pairs: {len(self.pairs)} selected "
            f"from {candidate_count} candidates"
        )

        if not self.pairs:
            raise RuntimeError(
                "No valid image pairs found. Check dataset and split paths."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name, context, goal_t = self.pairs[index]
        folder = ROOT / name

        frames = [
            img_path_to_data(str(folder / f"{t}.jpg"), IMAGE_SIZE)
            for t in context
        ]
        goal = img_path_to_data(
            str(folder / f"{goal_t}.jpg"), IMAGE_SIZE
        )

        return torch.cat(frames, dim=0), goal


def normalize_batch(data):
    obs, goal = data

    # Apply the same ImageNet normalization independently to each frame.
    obs = torch.cat(
        [normalize(frame) for frame in torch.split(obs, 3, dim=1)],
        dim=1,
    )
    goal = normalize(goal)

    return obs.to(device), goal.to(device)


# ------------------------------------------------------------
# Teacher
# ------------------------------------------------------------

def load_teacher():
    checkpoint = torch.load(
        TEACHER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    teacher = checkpoint["model"].to(device).eval()

    # Compatibility fix for the serialized teacher in this PyTorch version.
    # Only modifies the in-memory model; the original checkpoint is untouched.
    for layer in teacher.decoder.sa_decoder.layers:
        if layer.activation is F.gelu:
            layer.activation_relu_or_gelu = 2
        elif layer.activation is F.relu:
            layer.activation_relu_or_gelu = 1
        else:
            raise RuntimeError(
                f"Unexpected teacher activation: {layer.activation}"
            )

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    print("Teacher loaded and frozen.")
    return teacher


# ------------------------------------------------------------
# Training and validation
# ------------------------------------------------------------

def run_epoch(model, teacher, loader, optimizer=None, epoch=0):
    training = optimizer is not None
    model.train(training)

    total_losses = []
    distance_losses = []
    waypoint_losses = []

    for step, data in enumerate(loader, start=1):
        obs, goal = normalize_batch(data)

        # Generate teacher targets without computing teacher gradients.
        with torch.no_grad():
            teacher_dist, teacher_action = teacher(obs, goal)

        if training:
            optimizer.zero_grad(set_to_none=True)

            student_dist, student_action = model(obs, goal)

            distance_loss = F.mse_loss(student_dist, teacher_dist)
            waypoint_loss = F.mse_loss(student_action, teacher_action)

            total_loss = (
                waypoint_loss + DISTANCE_WEIGHT * distance_loss
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {step}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

        else:
            with torch.inference_mode():
                student_dist, student_action = model(obs, goal)

                distance_loss = F.mse_loss(student_dist, teacher_dist)
                waypoint_loss = F.mse_loss(student_action, teacher_action)
                total_loss = (
                    waypoint_loss + DISTANCE_WEIGHT * distance_loss
                )

        total_losses.append(total_loss.item())
        distance_losses.append(distance_loss.item())
        waypoint_losses.append(waypoint_loss.item())

        if training and (step == 1 or step % 50 == 0):
            print(
                f"  Batch {step}/{len(loader)} | "
                f"distance={distance_loss.item():.5f} | "
                f"waypoint={waypoint_loss.item():.5f} | "
                f"total={total_loss.item():.5f}"
            )

    return {
        "total": float(np.mean(total_losses)),
        "distance": float(np.mean(distance_losses)),
        "waypoint": float(np.mean(waypoint_losses)),
    }


# ------------------------------------------------------------
# Metrics and plots
# ------------------------------------------------------------

def save_history(history):
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs,
        [r["train_waypoint_mse"] for r in history],
        marker="o",
        label="Train",
    )
    axes[0].plot(
        epochs,
        [r["val_waypoint_mse"] for r in history],
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Waypoint MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [r["train_distance_mse"] for r in history],
        marker="o",
        label="Train",
    )
    axes[1].plot(
        epochs,
        [r["val_distance_mse"] for r in history],
        marker="o",
        label="Validation",
    )
    axes[1].set_title("Distance MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Device:", device)
    print("Output directory:", OUT.resolve())

    train_file = SPLITS / "train/traj_names.txt"
    test_file = SPLITS / "test/traj_names.txt"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Expected trajectory split files under {SPLITS}"
        )

    train_names = [
        x.strip() for x in train_file.read_text().splitlines() if x.strip()
    ]
    test_names = [
        x.strip() for x in test_file.read_text().splitlines() if x.strip()
    ]

    # Deterministic 90/10 trajectory-level split of the training trajectories.
    # The official held-out test trajectories are not used for training.
    random.Random(SEED).shuffle(train_names)
    val_count = max(1, round(0.10 * len(train_names)))

    val_names = train_names[:val_count]
    fit_names = train_names[val_count:]

    if set(fit_names) & set(val_names):
        raise RuntimeError("Training and validation trajectories overlap.")
    if (set(fit_names) | set(val_names)) & set(test_names):
        raise RuntimeError("Official test trajectories leaked into training.")

    print("Fit trajectories:", len(fit_names))
    print("Validation trajectories:", len(val_names))
    print("Official held-out test trajectories:", len(test_names))

    train_dataset = ImagePairDataset(
        fit_names, MAX_TRAIN_SAMPLES, seed=SEED
    )
    val_dataset = ImagePairDataset(
        val_names, MAX_VAL_SAMPLES, seed=SEED + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    teacher = load_teacher()

    # Pretrained encoder is loaded once. Checkpoints use pretrained=False
    # at evaluation time and restore the saved state dictionary.
    model = FastViTStudentV4Hybrid(pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Student parameters: {total_params:,} ({total_params / 1e6:.3f}M)")

    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": BACKBONE_LR},
            {"params": other_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_val_waypoint = float("inf")
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 18} Epoch {epoch}/{EPOCHS} {'=' * 18}")

        train_metrics = run_epoch(
            model, teacher, train_loader, optimizer=optimizer, epoch=epoch
        )
        val_metrics = run_epoch(
            model, teacher, val_loader, optimizer=None, epoch=epoch
        )

        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_distance_mse": train_metrics["distance"],
            "train_waypoint_mse": train_metrics["waypoint"],
            "val_total": val_metrics["total"],
            "val_distance_mse": val_metrics["distance"],
            "val_waypoint_mse": val_metrics["waypoint"],
        }
        history.append(row)
        save_history(history)

        print(
            f"Epoch {epoch}: "
            f"train waypoint={row['train_waypoint_mse']:.6f}, "
            f"val waypoint={row['val_waypoint_mse']:.6f}, "
            f"train distance={row['train_distance_mse']:.6f}, "
            f"val distance={row['val_distance_mse']:.6f}"
        )

        # Select the checkpoint strictly by validation waypoint MSE.
        if row["val_waypoint_mse"] < best_val_waypoint:
            best_val_waypoint = row["val_waypoint_mse"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_waypoint_mse": best_val_waypoint,
                    "val_distance_mse": row["val_distance_mse"],
                    "config": {
                        "embed_dim": 384,
                        "num_frames": 6,
                        "num_waypoints": 5,
                        "self_attention_layers": 3,
                        "attention_heads": 4,
                    },
                    "seed": SEED,
                },
                OUT / "student_v4_hybrid_best.pth",
            )
            print("  Saved new best checkpoint.")

        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            OUT / "student_v4_hybrid_last.pth",
        )

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print(f"Best validation waypoint MSE: {best_val_waypoint:.6f}")
    print("Best checkpoint:", OUT / "student_v4_hybrid_best.pth")
    print("Metrics:", OUT / "metrics.csv")
    print("Curves:", OUT / "training_curves.png")


if __name__ == "__main__":
    main()
