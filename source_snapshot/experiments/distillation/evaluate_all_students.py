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
from experiments.distillation.student_v2 import ViNTStudentV2


ROOT = Path.home() / "Downloads/go_stanford"
SPLITS = Path.home() / "Downloads/vint_data_splits/go_stanford"

TEACHER_PATH = Path("deployment/model_weights/vint.pth")
V1_PATH = Path(
    "experiments/distillation/results/training/student_best.pth"
)
V2_PATH = Path(
    "experiments/distillation/results/training_v2/student_v2_best.pth"
)

OUT = Path("experiments/distillation/results/evaluation_v2")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 123
IMAGE_SIZE = (85, 64)
MAX_TEST_PAIRS = 512
NUM_VISUALS = 4
WARMUP = 15
N_BENCHMARK = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_teacher():
    checkpoint = torch.load(
        TEACHER_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = checkpoint["model"].to(device).eval()

    # PyTorch compatibility fix for the serialized teacher.
    for layer in model.decoder.sa_decoder.layers:
        if layer.activation is F.gelu:
            layer.activation_relu_or_gelu = 2
        elif layer.activation is F.relu:
            layer.activation_relu_or_gelu = 1
        else:
            raise RuntimeError(
                f"Unexpected teacher activation: {layer.activation}"
            )

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def load_v1():
    model = ViNTStudent().to(device)
    checkpoint = torch.load(
        V1_PATH,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("student_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.eval()


def load_v2():
    # The checkpoint contains trained weights; don't download or
    # initialize ImageNet weights again during evaluation.
    model = ViNTStudentV2(pretrained=False).to(device)
    checkpoint = torch.load(
        V2_PATH,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("student_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.eval()


def make_pair(folder, obs_time, goal_time):
    frames = [
        img_path_to_data(
            str(folder / f"{t}.jpg"), IMAGE_SIZE
        )
        for t in range(obs_time - 5, obs_time + 1)
    ]

    goal = img_path_to_data(
        str(folder / f"{goal_time}.jpg"), IMAGE_SIZE
    )

    obs = torch.cat(
        [normalize(frame) for frame in frames], dim=0
    ).unsqueeze(0)

    goal = normalize(goal).unsqueeze(0)

    return obs.to(device), goal.to(device)


def collect_test_pairs():
    test_file = SPLITS / "test/traj_names.txt"
    test_names = [
        x.strip()
        for x in test_file.read_text().splitlines()
        if x.strip()
    ]

    candidates = []

    for name in test_names:
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
            goal_t = t + 10

            if (
                all(i in available for i in range(t - 5, t + 1))
                and goal_t in available
            ):
                candidates.append((name, t, goal_t))

    if not candidates:
        raise RuntimeError("No held-out test pairs found.")

    random.Random(SEED).shuffle(candidates)

    selected = candidates[:min(MAX_TEST_PAIRS, len(candidates))]

    print("Held-out test trajectories:", len(test_names))
    print("Available held-out pairs:", len(candidates))
    print("Pairs used for quantitative evaluation:", len(selected))

    return selected


@torch.inference_mode()
def predict(model, obs, goal):
    return model(obs, goal)


def waypoint_xy(actions):
    return actions[0, :, :2].detach().float().cpu().numpy()


def evaluate_imitation(teacher, student, pairs, model_name):
    waypoint_mses = []
    distance_sq_errors = []
    displacement_errors = []
    per_pair_rows = []

    for index, (traj, obs_time, goal_time) in enumerate(pairs, 1):
        obs, goal = make_pair(ROOT / traj, obs_time, goal_time)

        teacher_dist, teacher_action = predict(teacher, obs, goal)
        student_dist, student_action = predict(student, obs, goal)

        t_xy = waypoint_xy(teacher_action)
        s_xy = waypoint_xy(student_action)

        delta = np.linalg.norm(t_xy - s_xy, axis=1)

        waypoint_mse = float(np.mean((t_xy - s_xy) ** 2))
        distance_sq_error = float(
            (teacher_dist.item() - student_dist.item()) ** 2
        )

        waypoint_mses.append(waypoint_mse)
        distance_sq_errors.append(distance_sq_error)
        displacement_errors.extend(delta.tolist())

        per_pair_rows.append({
            "model": model_name,
            "trajectory": traj,
            "obs_time": obs_time,
            "goal_time": goal_time,
            "waypoint_mse": waypoint_mse,
            "mean_waypoint_disagreement": float(np.mean(delta)),
            "max_waypoint_disagreement": float(np.max(delta)),
            "teacher_distance": float(teacher_dist.item()),
            "student_distance": float(student_dist.item()),
            "distance_squared_error": distance_sq_error,
        })

        if index % 100 == 0:
            print(f"  {model_name}: evaluated {index}/{len(pairs)}")

    summary = {
        "model": model_name,
        "num_test_pairs": len(pairs),
        "waypoint_mse_vs_teacher": float(np.mean(waypoint_mses)),
        "distance_mse_vs_teacher": float(np.mean(distance_sq_errors)),
        "mean_waypoint_disagreement": float(
            np.mean(displacement_errors)
        ),
        "median_waypoint_disagreement": float(
            np.median(displacement_errors)
        ),
        "p95_waypoint_disagreement": float(
            np.percentile(displacement_errors, 95)
        ),
        "max_waypoint_disagreement": float(
            np.max(displacement_errors)
        ),
    }

    return summary, per_pair_rows


def benchmark(model, name):
    # Fixed-size model-forward benchmark; preprocessing is excluded.
    folder = ROOT / "sim39h_0_2"
    obs, goal = make_pair(folder, 5, 15)

    for _ in range(WARMUP):
        predict(model, obs, goal)

    synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings = []

    for _ in range(N_BENCHMARK):
        synchronize()
        start = time.perf_counter()
        predict(model, obs, goal)
        synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    values = np.asarray(timings)

    return {
        "model": name,
        "batch_size": 1,
        "runs": N_BENCHMARK,
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_samples_per_second": float(
            1000.0 / np.mean(values)
        ),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if device.type == "cuda" else None
        ),
    }


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_waypoints(teacher, v1, v2, selected):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    with torch.inference_mode():
        for index, (traj, obs_time, goal_time) in enumerate(selected):
            ax = axes[index]
            obs, goal = make_pair(ROOT / traj, obs_time, goal_time)

            _, ta = predict(teacher, obs, goal)
            _, a1 = predict(v1, obs, goal)
            _, a2 = predict(v2, obs, goal)

            outputs = [
                ("Teacher", waypoint_xy(ta), "o-"),
                ("V1", waypoint_xy(a1), "s--"),
                ("V2", waypoint_xy(a2), "^-."),
            ]

            for label, xy, style in outputs:
                ax.plot(
                    np.r_[0, xy[:, 0]],
                    np.r_[0, xy[:, 1]],
                    style,
                    linewidth=1.8,
                    markersize=4,
                    label=label,
                )

            ax.scatter([0], [0], marker="*", s=75, label="Start")
            ax.set_title(f"{traj}: t={obs_time} → goal={goal_time}")
            ax.set_xlabel("Relative X (model output units)")
            ax.set_ylabel("Relative Y (model output units)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.axis("equal")

    for index in range(len(selected), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Held-out waypoint imitation: teacher, V1, V2")
    fig.tight_layout()
    path = OUT / "waypoint_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path.resolve())


def plot_latency(stats):
    names = [row["model"] for row in stats]
    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for offset, key, label in [
        (-width, "mean_ms", "Mean"),
        (0, "median_ms", "Median"),
        (width, "p95_ms", "P95"),
    ]:
        values = [row[key] for row in stats]
        ax.bar(x + offset, values, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Batch-size-one inference latency")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUT / "latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path.resolve())


def main():
    for path in [TEACHER_PATH, V1_PATH, V2_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Device:", device)

    pairs = collect_test_pairs()
    visual_pairs = pairs[:min(NUM_VISUALS, len(pairs))]

    teacher = load_teacher()
    v1 = load_v1()
    v2 = load_v2()

    models = [
        ("Teacher", teacher),
        ("V1", v1),
        ("V2", v2),
    ]

    parameter_counts = {
        name: sum(p.numel() for p in model.parameters())
        for name, model in models
    }

    print("\nParameter counts:")
    for name, count in parameter_counts.items():
        print(f"  {name}: {count:,}")

    summaries = []
    pair_rows = []

    for name, model in [("V1", v1), ("V2", v2)]:
        print(f"\nEvaluating {name} against teacher...")
        summary, rows = evaluate_imitation(
            teacher, model, pairs, name
        )
        summaries.append(summary)
        pair_rows.extend(rows)
        print(json.dumps(summary, indent=2))

    save_csv(OUT / "per_pair_metrics.csv", pair_rows)

    # The teacher is the reference, so its disagreement with itself is zero.
    reference = {
        "model": "Teacher",
        "num_test_pairs": len(pairs),
        "waypoint_mse_vs_teacher": 0.0,
        "distance_mse_vs_teacher": 0.0,
        "mean_waypoint_disagreement": 0.0,
        "median_waypoint_disagreement": 0.0,
        "p95_waypoint_disagreement": 0.0,
        "max_waypoint_disagreement": 0.0,
    }

    full_summary = {
        "device": str(device),
        "seed": SEED,
        "test_pairs": len(pairs),
        "parameter_counts": parameter_counts,
        "imitation_metrics": [reference] + summaries,
    }

    (OUT / "imitation_metrics.json").write_text(
        json.dumps(full_summary, indent=2)
    )
    save_csv(OUT / "imitation_metrics.csv", [reference] + summaries)

    print("\nBenchmarking model-forward latency...")
    latency = []

    for name, model in models:
        stats = benchmark(model, name)
        stats["parameters"] = parameter_counts[name]
        latency.append(stats)
        print(json.dumps(stats, indent=2))

    (OUT / "latency_results.json").write_text(
        json.dumps(latency, indent=2)
    )
    save_csv(OUT / "latency_results.csv", latency)

    plot_waypoints(teacher, v1, v2, visual_pairs)
    plot_latency(latency)

    print("\nEvaluation finished.")
    print("Results directory:", OUT.resolve())
    print("Imitation metrics:", (OUT / "imitation_metrics.csv").resolve())
    print("Per-pair metrics:", (OUT / "per_pair_metrics.csv").resolve())
    print("Latency metrics:", (OUT / "latency_results.csv").resolve())


if __name__ == "__main__":
    main()
