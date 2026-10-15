from pathlib import Path
import csv
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from vint_train.data.data_utils import img_path_to_data
from experiments.distillation.student import ViNTStudent


ROOT = Path.home() / "Downloads/go_stanford"
SPLITS = Path.home() / "Downloads/vint_data_splits/go_stanford"
TEACHER_PATH = Path("deployment/model_weights/vint.pth")
STUDENT_PATH = Path(
    "experiments/distillation/results/training/student_best.pth"
)
OUT = Path("experiments/distillation/results/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 123
IMAGE_SIZE = (85, 64)
N_BENCHMARK = 100
WARMUP = 15
NUM_VISUALS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def load_teacher():
    checkpoint = torch.load(
        TEACHER_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = checkpoint["model"].to(device).eval()

    # Compatibility fix for the serialized teacher's Transformer.
    for layer in model.decoder.sa_decoder.layers:
        if layer.activation is F.gelu:
            layer.activation_relu_or_gelu = 2
        elif layer.activation is F.relu:
            layer.activation_relu_or_gelu = 1
        else:
            raise RuntimeError(
                f"Unexpected teacher activation: {layer.activation}"
            )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def load_student():
    model = ViNTStudent().to(device)
    checkpoint = torch.load(
        STUDENT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("student_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def make_pair(folder, obs_time, goal_time):
    """Load six observation frames and one goal image."""
    frames = [
        img_path_to_data(
            str(folder / f"{t}.jpg"), IMAGE_SIZE
        )
        for t in range(obs_time - 5, obs_time + 1)
    ]

    goal = img_path_to_data(
        str(folder / f"{goal_time}.jpg"), IMAGE_SIZE
    )

    # Normalize each RGB frame independently.
    obs = torch.cat(
        [normalize(frame) for frame in frames],
        dim=0,
    ).unsqueeze(0)

    goal = normalize(goal).unsqueeze(0)

    return obs.to(device), goal.to(device)


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def predict(model, obs, goal):
    return model(obs, goal)


def benchmark(model, inputs, name):
    """Measure model-forward latency, excluding preprocessing."""
    obs, goal = inputs

    for _ in range(WARMUP):
        predict(model, obs, goal)

    synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times_ms = []

    for _ in range(N_BENCHMARK):
        synchronize()
        start = time.perf_counter()

        predict(model, obs, goal)

        synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    values = np.asarray(times_ms)

    return {
        "model": name,
        "batch_size": 1,
        "runs": N_BENCHMARK,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_samples_per_second": float(
            1000 / values.mean()
        ),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if device.type == "cuda"
            else None
        ),
    }


def collect_test_pairs():
    test_file = SPLITS / "test/traj_names.txt"

    test_names = [
        line.strip()
        for line in test_file.read_text().splitlines()
        if line.strip()
    ]

    print("Held-out test trajectories:", len(test_names))

    candidates = []

    for name in test_names:
        folder = ROOT / name

        frame_ids = sorted(
            int(path.stem)
            for path in folder.glob("*.jpg")
            if path.stem.isdigit()
        )
        available = set(frame_ids)

        for t in frame_ids:
            goal_t = t + 10

            if (
                all(i in available for i in range(t - 5, t + 1))
                and goal_t in available
            ):
                candidates.append((name, t, goal_t))

    if not candidates:
        raise RuntimeError(
            "No valid image pairs found in held-out trajectories."
        )

    random.Random(SEED).shuffle(candidates)

    print("Valid held-out image pairs:", len(candidates))

    return candidates[:min(NUM_VISUALS, len(candidates))]


def denormalize(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    return (image.detach().cpu() * std + mean).clamp(0, 1)


def plot_predictions(teacher, student, selected):
    """
    Compact 2x2 layout.

    Each example occupies one panel containing:
    observation image, goal image, and waypoint comparison.
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        squeeze=False,
    )
    axes = axes.flatten()

    errors = []

    with torch.inference_mode():
        for index, (traj, obs_time, goal_time) in enumerate(selected):
            ax = axes[index]

            obs, goal = make_pair(
                ROOT / traj, obs_time, goal_time
            )

            teacher_dist, teacher_action = predict(
                teacher, obs, goal
            )
            student_dist, student_action = predict(
                student, obs, goal
            )

            teacher_xy = (
                teacher_action[0, :, :2].detach().cpu().numpy()
            )
            student_xy = (
                student_action[0, :, :2].detach().cpu().numpy()
            )

            # Per-waypoint Euclidean disagreement in model output units.
            waypoint_errors = np.linalg.norm(
                teacher_xy - student_xy, axis=1
            )
            errors.extend(waypoint_errors.tolist())

            # Compact top-down view with a concise example label.
            ax.plot(
                np.r_[0, teacher_xy[:, 0]],
                np.r_[0, teacher_xy[:, 1]],
                "o-",
                linewidth=2,
                markersize=4,
                label="Teacher",
            )
            ax.plot(
                np.r_[0, student_xy[:, 0]],
                np.r_[0, student_xy[:, 1]],
                "s--",
                linewidth=2,
                markersize=4,
                label="Student",
            )
            ax.scatter(
                [0], [0],
                marker="*",
                s=90,
                label="Start",
            )

            ax.set_title(
                f"{traj}: t={obs_time} → goal={goal_time}\n"
                f"Distance output: teacher={teacher_dist.item():.2f}, "
                f"student={student_dist.item():.2f}",
                fontsize=9,
            )
            ax.set_xlabel("Relative X (model output units)")
            ax.set_ylabel("Relative Y (model output units)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.axis("equal")

    # Hide any unused panels.
    for index in range(len(selected), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        "ViNT Teacher vs Student — Held-out Waypoint Predictions",
        fontsize=14,
    )
    fig.tight_layout()

    path = OUT / "teacher_student_predictions.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    error_summary = {
        "num_examples": len(selected),
        "num_waypoints_compared": len(errors),
        "mean_teacher_student_waypoint_disagreement": (
            float(np.mean(errors)) if errors else None
        ),
        "median_teacher_student_waypoint_disagreement": (
            float(np.median(errors)) if errors else None
        ),
    }

    (OUT / "waypoint_disagreement.json").write_text(
        json.dumps(error_summary, indent=2)
    )

    print("\nSaved compact prediction plot:", path.resolve())
    print("Waypoint disagreement:", error_summary)


def plot_latency(teacher_stats, student_stats):
    """Create a separate compact latency comparison chart."""
    metrics = [
        ("mean_ms", "Mean"),
        ("median_ms", "Median"),
        ("p95_ms", "P95"),
    ]

    x = np.arange(len(metrics))
    width = 0.34

    teacher_values = [
        teacher_stats[key] for key, _ in metrics
    ]
    student_values = [
        student_stats[key] for key, _ in metrics
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        x - width / 2,
        teacher_values,
        width,
        label="Teacher",
    )
    ax.bar(
        x + width / 2,
        student_values,
        width,
        label="Student",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title("Teacher vs Student Inference Latency")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for i, value in enumerate(teacher_values):
        ax.text(
            x[i] - width / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for i, value in enumerate(student_values):
        ax.text(
            x[i] + width / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    path = OUT / "latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved latency chart:", path.resolve())


def main():
    if not TEACHER_PATH.exists():
        raise FileNotFoundError(TEACHER_PATH)

    if not STUDENT_PATH.exists():
        raise FileNotFoundError(STUDENT_PATH)

    selected = collect_test_pairs()

    teacher = load_teacher()
    student = load_student()

    teacher_params = sum(
        p.numel() for p in teacher.parameters()
    )
    student_params = sum(
        p.numel() for p in student.parameters()
    )

    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(
        f"Parameter reduction: "
        f"{(1 - student_params / teacher_params) * 100:.2f}%"
    )

    # Use the same held-out image pair for both benchmarks.
    first_name, first_obs, first_goal = selected[0]
    benchmark_inputs = make_pair(
        ROOT / first_name,
        first_obs,
        first_goal,
    )

    print("\nBenchmarking teacher...")
    teacher_stats = benchmark(
        teacher, benchmark_inputs, "ViNT teacher"
    )
    print(json.dumps(teacher_stats, indent=2))

    print("\nBenchmarking student...")
    student_stats = benchmark(
        student, benchmark_inputs, "ViNT student"
    )
    print(json.dumps(student_stats, indent=2))

    results = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "teacher_parameters": teacher_params,
        "student_parameters": student_params,
        "parameter_reduction_percent": (
            (1 - student_params / teacher_params) * 100
        ),
        "latency_scope": (
            "Model forward only; preprocessing excluded; "
            "batch size 1; synchronized timings after warm-up."
        ),
        "latency": [teacher_stats, student_stats],
    }

    (OUT / "latency_results.json").write_text(
        json.dumps(results, indent=2)
    )

    with (OUT / "latency_results.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(teacher_stats.keys()),
        )
        writer.writeheader()
        writer.writerows([teacher_stats, student_stats])

    plot_predictions(teacher, student, selected)
    plot_latency(teacher_stats, student_stats)

    print("\nAll evaluation files saved under:")
    print(OUT.resolve())


if __name__ == "__main__":
    main()
