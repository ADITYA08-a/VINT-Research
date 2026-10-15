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
from experiments.distillation.student import ViNTStudent

ROOT = Path.home() / "Downloads/go_stanford"
SPLITS = Path.home() / "Downloads/vint_data_splits/go_stanford"
TEACHER_PATH = Path("deployment/model_weights/vint.pth")
OUT = Path("experiments/distillation/results/training")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 2
MAX_TRAIN_SAMPLES = 512
MAX_VAL_SAMPLES = 128
LR = 1e-4
NUM_WORKERS = 0
IMAGE_SIZE = (85, 64)
CONTEXT_FRAMES = 6
GOAL_OFFSET = 10


class ImagePairDataset(Dataset):
    """Image-only dataset: six past/current frames plus a future goal."""

    def __init__(self, trajectory_names, max_samples, seed):
        pairs = []
        for name in sorted(trajectory_names):
            folder = ROOT / name
            frame_ids = sorted(
                int(p.stem) for p in folder.glob("*.jpg")
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

        rng = random.Random(seed)
        rng.shuffle(pairs)
        self.pairs = pairs[:max_samples]
        print(
            f"Image pairs: {len(self.pairs)} "
            f"(candidate pairs before cap: {len(pairs)})"
        )

        if not self.pairs:
            raise RuntimeError("No valid image pairs found.")

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


def load_teacher(device):
    checkpoint = torch.load(
        TEACHER_PATH, map_location="cpu", weights_only=False
    )
    teacher = checkpoint["model"].to(device).eval()

    # Compatibility for the serialized Transformer in this environment.
    for layer in teacher.decoder.sa_decoder.layers:
        if layer.activation is F.gelu:
            layer.activation_relu_or_gelu = 2
        elif layer.activation is F.relu:
            layer.activation_relu_or_gelu = 1
        else:
            raise RuntimeError(
                f"Unexpected teacher activation: {layer.activation}"
            )

    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def normalize_batch(data, transform, device):
    obs, goal = data
    # Normalize each RGB frame independently, as in original ViNT training.
    obs = torch.cat(
        [transform(frame) for frame in torch.split(obs, 3, dim=1)],
        dim=1,
    )
    goal = transform(goal)
    return obs.to(device), goal.to(device)


def run_epoch(model, teacher, loader, transform, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    dist_losses, action_losses = [], []

    for step, data in enumerate(loader, 1):
        obs, goal = normalize_batch(data, transform, device)

        with torch.no_grad():
            teacher_dist, teacher_action = teacher(obs, goal)

        if training:
            student_dist, student_action = model(obs, goal)
            dist_loss = F.mse_loss(student_dist, teacher_dist)
            action_loss = F.mse_loss(student_action, teacher_action)
            loss = dist_loss + action_loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at batch {step}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.inference_mode():
                student_dist, student_action = model(obs, goal)
                dist_loss = F.mse_loss(student_dist, teacher_dist)
                action_loss = F.mse_loss(student_action, teacher_action)

        dist_losses.append(dist_loss.item())
        action_losses.append(action_loss.item())

        if training and (step == 1 or step % 10 == 0):
            print(
                f"Batch {step}/{len(loader)} "
                f"distance={dist_loss.item():.5f} "
                f"waypoints={action_loss.item():.5f}"
            )

    return float(np.mean(dist_losses)), float(np.mean(action_losses))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_names = [
        x.strip()
        for x in (SPLITS / "train/traj_names.txt").read_text().splitlines()
        if x.strip()
    ]
    test_names = set(
        x.strip()
        for x in (SPLITS / "test/traj_names.txt").read_text().splitlines()
        if x.strip()
    )
    if set(train_names) & test_names:
        raise RuntimeError("Train/test trajectory leakage detected.")

    rng = random.Random(SEED)
    rng.shuffle(train_names)
    n_val = max(1, len(train_names) // 10)
    val_names = set(train_names[:n_val])
    fit_names = set(train_names[n_val:])

    print(
        f"Train trajectories: {len(fit_names)}; "
        f"validation trajectories: {len(val_names)}; "
        f"held-out test trajectories: {len(test_names)}"
    )

    train_data = ImagePairDataset(
        fit_names, MAX_TRAIN_SAMPLES, SEED
    )
    val_data = ImagePairDataset(
        val_names, MAX_VAL_SAMPLES, SEED + 1
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    teacher = load_teacher(device)
    student = ViNTStudent().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

    print(
        f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}"
    )
    print(
        f"Student parameters: {sum(p.numel() for p in student.parameters()):,}"
    )

    history = []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        train_dist, train_action = run_epoch(
            student, teacher, train_loader, transform, device, optimizer
        )
        val_dist, val_action = run_epoch(
            student, teacher, val_loader, transform, device
        )

        row = {
            "epoch": epoch,
            "train_dist_mse": train_dist,
            "train_action_mse": train_action,
            "val_dist_mse": val_dist,
            "val_action_mse": val_action,
        }
        history.append(row)

        print(
            f"Train distance MSE: {train_dist:.6f}; "
            f"train waypoint MSE: {train_action:.6f}"
        )
        print(
            f"Validation distance MSE: {val_dist:.6f}; "
            f"validation waypoint MSE: {val_action:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "seed": SEED,
        }
        torch.save(checkpoint, OUT / f"student_epoch_{epoch}.pth")

        val_total = val_dist + val_action
        if val_total < best_val:
            best_val = val_total
            torch.save(checkpoint, OUT / "student_best.pth")

        with (OUT / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, train_key, val_key, title in [
            (axes[0], "train_dist_mse", "val_dist_mse", "Distance MSE"),
            (axes[1], "train_action_mse", "val_action_mse", "Waypoint MSE"),
        ]:
            ax.plot(
                [r["epoch"] for r in history],
                [r[train_key] for r in history],
                "o-", label="Train"
            )
            ax.plot(
                [r["epoch"] for r in history],
                [r[val_key] for r in history],
                "s-", label="Validation"
            )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True)
            ax.legend()

        fig.tight_layout()
        fig.savefig(OUT / "loss_curves.png", dpi=160)
        plt.close(fig)

    print("\nSmoke-training complete.")
    print("Best checkpoint:", (OUT / "student_best.pth").resolve())
    print("Metrics:", (OUT / "metrics.csv").resolve())
    print("Loss plot:", (OUT / "loss_curves.png").resolve())


if __name__ == "__main__":
    main()
