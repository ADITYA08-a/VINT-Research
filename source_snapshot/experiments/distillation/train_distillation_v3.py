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
from experiments.distillation.student_v3 import FastViTStudentV3


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path.home() / "Downloads/go_stanford"
SPLITS = Path.home() / "Downloads/vint_data_splits/go_stanford"
TEACHER_PATH = Path("deployment/model_weights/vint.pth")
OUT = Path("experiments/distillation/results/training_v3")
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


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class ImagePairDataset(Dataset):
    """Six consecutive observation frames and one future goal."""

    def __init__(self, trajectory_names, max_samples, seed):
        pairs = []

        for name in sorted(trajectory_names):
            folder = ROOT / name

            frame_ids = sorted(
                int(p.stem)
                for p in folder.glob("*.jpg")
                if p.stem.isdigit()
            )
            available = set(frame_ids)

            for t in frame_ids:
                context = list(
                    range(t - CONTEXT_FRAMES + 1, t + 1)
                )
                goal_t = t + GOAL_OFFSET

                if not all(i in available for i in context):
                    continue
                if goal_t not in available:
                    continue

                pairs.append((name, context, goal_t))

        rng = random.Random(seed)
        rng.shuffle(pairs)
        self.pairs = pairs[:max_samples]

        print(
            f"Image pairs: {len(self.pairs)} "
            f"(candidates: {len(pairs)})"
        )

        if not self.pairs:
            raise RuntimeError(
                "No valid pairs found. Check ROOT and trajectory names."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name, context, goal_t = self.pairs[index]
        folder = ROOT / name

        frames = [
            img_path_to_data(
                str(folder / f"{t}.jpg"), IMAGE_SIZE
            )
            for t in context
        ]

        goal = img_path_to_data(
            str(folder / f"{goal_t}.jpg"), IMAGE_SIZE
        )

        return torch.cat(frames, dim=0), goal


# ------------------------------------------------------------
# Teacher
# ------------------------------------------------------------

def load_teacher(device):
    checkpoint = torch.load(
        TEACHER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    teacher = checkpoint["model"].to(device).eval()

    # Compatibility fix applied only to the in-memory teacher.
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

    return teacher


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

def normalize_batch(data, transform, device):
    obs, goal = data

    obs = torch.cat(
        [
            transform(frame)
            for frame in torch.split(obs, 3, dim=1)
        ],
        dim=1,
    )

    goal = transform(goal)

    return obs.to(device), goal.to(device)


# ------------------------------------------------------------
# Training and validation
# ------------------------------------------------------------

def run_epoch(
    model,
    teacher,
    loader,
    transform,
    device,
    optimizer=None,
    epoch=0,
):
    training = optimizer is not None
    model.train(training)

    distance_losses = []
    waypoint_losses = []
    total_losses = []

    for step, data in enumerate(loader, start=1):
        obs, goal = normalize_batch(data, transform, device)

        with torch.no_grad():
            teacher_dist, teacher_action = teacher(obs, goal)

        if training:
            optimizer.zero_grad(set_to_none=True)

            student_dist, student_action = model(obs, goal)

            distance_loss = F.mse_loss(
                student_dist, teacher_dist
            )
            waypoint_loss = F.mse_loss(
                student_action, teacher_action
            )

            total_loss = (
                waypoint_loss
                + DISTANCE_WEIGHT * distance_loss
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

                distance_loss = F.mse_loss(
                    student_dist, teacher_dist
                )
                waypoint_loss = F.mse_loss(
                    student_action, teacher_action
                )

                total_loss = (
                    waypoint_loss
                    + DISTANCE_WEIGHT * distance_loss
                )

        distance_losses.append(distance_loss.item())
        waypoint_losses.append(waypoint_loss.item())
        total_losses.append(total_loss.item())

        if training and (step == 1 or step % 50 == 0):
            print(
                f"  Batch {step}/{len(loader)} | "
                f"dist={distance_loss.item():.5f} | "
                f"waypoint={waypoint_loss.item():.5f} | "
                f"total={total_loss.item():.5f}"
            )

    return {
        "total": float(np.mean(total_losses)),
        "distance": float(np.mean(distance_losses)),
        "waypoint": float(np.mean(waypoint_losses)),
    }


# ------------------------------------------------------------
# Saving and plotting
# ------------------------------------------------------------

def save_history(history):
    csv_path = OUT / "metrics.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(history[0].keys())
        )
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
    axes[0].set_title("FastViT V3 waypoint distillation MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [r["train_dist_mse"] for r in history],
        marker="o",
        label="Train",
    )
    axes[1].plot(
        epochs,
        [r["val_dist_mse"] for r in history],
        marker="o",
        label="Validation",
    )
    axes[1].set_title("FastViT V3 distance distillation MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=160)
    plt.close(fig)

    print("Saved:", csv_path)
    print("Saved:", OUT / "loss_curves.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {ROOT}")

    train_file = SPLITS / "train/traj_names.txt"
    test_file = SPLITS / "test/traj_names.txt"

    train_names = [
        line.strip()
        for line in train_file.read_text().splitlines()
        if line.strip()
    ]

    test_names = {
        line.strip()
        for line in test_file.read_text().splitlines()
        if line.strip()
    }

    if set(train_names) & test_names:
        raise RuntimeError("Train/test trajectory leakage detected.")

    rng = random.Random(SEED)
    rng.shuffle(train_names)

    n_val = max(1, len(train_names) // 10)
    val_names = set(train_names[:n_val])
    fit_names = set(train_names[n_val:])

    if fit_names & val_names:
        raise RuntimeError("Training/validation trajectory leakage.")

    print(
        f"Training trajectories: {len(fit_names)} | "
        f"Validation trajectories: {len(val_names)} | "
        f"Held-out test trajectories: {len(test_names)}"
    )

    train_data = ImagePairDataset(
        fit_names, MAX_TRAIN_SAMPLES, SEED
    )
    val_data = ImagePairDataset(
        val_names, MAX_VAL_SAMPLES, SEED + 1
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    teacher = load_teacher(device)
    student = FastViTStudentV3(pretrained=True).to(device)

    backbone_parameters = list(student.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_parameters}
    head_parameters = [
        p for p in student.parameters()
        if id(p) not in backbone_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": BACKBONE_LR},
            {"params": head_parameters, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(
        "Teacher parameters:",
        f"{sum(p.numel() for p in teacher.parameters()):,}",
    )
    print(
        "Student parameters:",
        f"{sum(p.numel() for p in student.parameters()):,}",
    )
    print("Training pairs:", len(train_data))
    print("Validation pairs:", len(val_data))
    print("Backbone LR:", BACKBONE_LR)
    print("New-layer LR:", HEAD_LR)
    print("Distance loss weight:", DISTANCE_WEIGHT)
    print("Output directory:", OUT.resolve())

    history = []
    best_val_waypoint = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        train_metrics = run_epoch(
            student, teacher, train_loader, transform,
            device, optimizer=optimizer, epoch=epoch,
        )

        val_metrics = run_epoch(
            student, teacher, val_loader, transform,
            device, optimizer=None, epoch=epoch,
        )

        scheduler.step(val_metrics["waypoint"])

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total"],
            "train_dist_mse": train_metrics["distance"],
            "train_waypoint_mse": train_metrics["waypoint"],
            "val_total_loss": val_metrics["total"],
            "val_dist_mse": val_metrics["distance"],
            "val_waypoint_mse": val_metrics["waypoint"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }

        history.append(row)

        print(
            f"\nTrain | dist={train_metrics['distance']:.6f} "
            f"waypoint={train_metrics['waypoint']:.6f}"
        )
        print(
            f"Valid | dist={val_metrics['distance']:.6f} "
            f"waypoint={val_metrics['waypoint']:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "seed": SEED,
            "architecture": "FastViTStudentV3",
            "config": {
                "backbone": "fastvit_t8",
                "pretrained": True,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "distance_weight": DISTANCE_WEIGHT,
            },
        }

        epoch_path = OUT / f"student_v3_epoch_{epoch}.pth"
        torch.save(checkpoint, epoch_path)

        if val_metrics["waypoint"] < best_val_waypoint:
            best_val_waypoint = val_metrics["waypoint"]
            torch.save(checkpoint, OUT / "student_v3_best.pth")
            print("New best checkpoint:", OUT / "student_v3_best.pth")

        save_history(history)

    print("\nTraining finished.")
    print("Best validation waypoint MSE:", best_val_waypoint)
    print("Best checkpoint:", (OUT / "student_v3_best.pth").resolve())
    print("Metrics:", (OUT / "metrics.csv").resolve())
    print("Loss plot:", (OUT / "loss_curves.png").resolve())


if __name__ == "__main__":
    main()
