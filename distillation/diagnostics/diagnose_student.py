from pathlib import Path
import json
import random

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
OUT = Path("experiments/distillation/results/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 123
MAX_PAIRS = 64
IMAGE_SIZE = (85, 64)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def load_teacher():
    ckpt = torch.load(
        TEACHER_PATH, map_location="cpu", weights_only=False
    )
    model = ckpt["model"].to(device).eval()

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


def load_student():
    model = ViNTStudent().to(device)
    ckpt = torch.load(
        STUDENT_PATH, map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt.get("student_state_dict", ckpt))
    return model.eval()


def make_pair(folder, t, goal_t):
    frames = [
        img_path_to_data(str(folder / f"{i}.jpg"), IMAGE_SIZE)
        for i in range(t - 5, t + 1)
    ]
    goal = img_path_to_data(
        str(folder / f"{goal_t}.jpg"), IMAGE_SIZE
    )
    obs = torch.cat([normalize(f) for f in frames], dim=0).unsqueeze(0)
    goal = normalize(goal).unsqueeze(0)
    return obs.to(device), goal.to(device)


def collect_pairs():
    names = [
        x.strip()
        for x in (
            SPLITS / "test/traj_names.txt"
        ).read_text().splitlines()
        if x.strip()
    ]

    candidates = []
    for name in names:
        folder = ROOT / name
        ids = {
            int(p.stem)
            for p in folder.glob("*.jpg")
            if p.stem.isdigit()
        }
        for t in sorted(ids):
            goal_t = t + 10
            if all(i in ids for i in range(t - 5, t + 1)) and goal_t in ids:
                candidates.append((name, t, goal_t))

    random.Random(SEED).shuffle(candidates)
    selected = candidates[:min(MAX_PAIRS, len(candidates))]
    if not selected:
        raise RuntimeError("No valid held-out image pairs found.")

    return selected


def summarize(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


@torch.inference_mode()
def main():
    teacher = load_teacher()
    student = load_student()
    pairs = collect_pairs()

    teacher_distances = []
    student_distances = []
    teacher_xy = []
    student_xy = []

    print("Pairs evaluated:", len(pairs))

    for index, (traj, t, goal_t) in enumerate(pairs, 1):
        obs, goal = make_pair(ROOT / traj, t, goal_t)

        td, ta = teacher(obs, goal)
        sd, sa = student(obs, goal)

        teacher_distances.append(td.cpu().numpy().reshape(-1))
        student_distances.append(sd.cpu().numpy().reshape(-1))
        teacher_xy.append(ta[0, :, :2].cpu().numpy())
        student_xy.append(sa[0, :, :2].cpu().numpy())

        if index <= 5:
            print(
                f"{index}. {traj}, t={t}, goal={goal_t}: "
                f"distance T={td.item():.3f}, S={sd.item():.3f}; "
                f"mean |XY| T={ta[0, :, :2].norm(dim=-1).mean().item():.3f}, "
                f"S={sa[0, :, :2].norm(dim=-1).mean().item():.3f}"
            )

    td = np.concatenate(teacher_distances)
    sd = np.concatenate(student_distances)
    tx = np.concatenate(teacher_xy, axis=0)
    sx = np.concatenate(student_xy, axis=0)

    # All statistics use identical held-out image pairs.
    waypoint_delta = tx - sx
    waypoint_error = np.linalg.norm(waypoint_delta, axis=-1)

    result = {
        "pairs": len(pairs),
        "teacher_distance": summarize("teacher_distance", td),
        "student_distance": summarize("student_distance", sd),
        "teacher_waypoint_xy": summarize("teacher_xy", tx),
        "student_waypoint_xy": summarize("student_xy", sx),
        "teacher_student_waypoint_error": {
            "mean": float(waypoint_error.mean()),
            "median": float(np.median(waypoint_error)),
            "p95": float(np.percentile(waypoint_error, 95)),
        },
        "student_distance_nearly_constant": bool(sd.std() < 1e-3),
        "student_waypoints_nearly_constant": bool(
            np.std(sx, axis=0).mean() < 1e-3
        ),
    }

    print("\n=== Output diagnostics ===")
    print(json.dumps(result, indent=2))
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=2))

    # Compact figure: output distributions and teacher/student agreement.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    axes[0].hist(td, bins=18, alpha=0.65, label="Teacher")
    axes[0].hist(sd, bins=18, alpha=0.65, label="Student")
    axes[0].set_title("Distance output distribution")
    axes[0].set_xlabel("Predicted distance output")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].scatter(
        tx[:, 0], sx[:, 0], s=12, alpha=0.6, label="X"
    )
    axes[1].scatter(
        tx[:, 1], sx[:, 1], s=12, alpha=0.6, label="Y"
    )
    axes[1].set_title("Student vs teacher coordinates")
    axes[1].set_xlabel("Teacher output")
    axes[1].set_ylabel("Student output")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    axes[2].hist(
        waypoint_error, bins=18, alpha=0.8
    )
    axes[2].set_title("Waypoint disagreement")
    axes[2].set_xlabel("Euclidean output difference")
    axes[2].set_ylabel("Count")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    plot_path = OUT / "output_diagnostics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved diagnostics:")
    print((OUT / "diagnostics.json").resolve())
    print(plot_path.resolve())


if __name__ == "__main__":
    main()
