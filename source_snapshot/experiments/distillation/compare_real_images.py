from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vint_train.data.data_utils import img_path_to_data
from student import ViNTStudent


ROOT = Path.home() / "Downloads/go_stanford"
TRAJECTORY = "sim39h_0_2"
FRAME_IDS = [0, 1, 2, 3, 4, 5]
GOAL_ID = 20

TEACHER_PATH = Path("deployment/model_weights/vint.pth")
OUTPUT_PATH = Path(
    "experiments/distillation/results/teacher_student_real_images.png"
)

IMAGE_SIZE = (85, 64)  # width, height
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path):
    """Apply ViNT's crop/resize and ImageNet normalization."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    tensor = img_path_to_data(str(path), IMAGE_SIZE)
    return (tensor - MEAN) / STD


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load teacher checkpoint without modifying it on disk.
    checkpoint = torch.load(
        TEACHER_PATH, map_location="cpu", weights_only=False
    )
    teacher = checkpoint["model"].to(device).eval()

    # Compatibility for serialized Transformer layers from the checkpoint.
    for layer in teacher.decoder.sa_decoder.layers:
        if layer.activation is F.gelu:
            layer.activation_relu_or_gelu = 2
        elif layer.activation is F.relu:
            layer.activation_relu_or_gelu = 1
        else:
            raise RuntimeError(
                f"Unexpected decoder activation: {layer.activation}"
            )

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # This is a fresh, UNTRAINED student.
    student = ViNTStudent().to(device).eval()

    folder = ROOT / TRAJECTORY
    image_paths = [folder / f"{i}.jpg" for i in FRAME_IDS]
    goal_path = folder / f"{GOAL_ID}.jpg"

    # [1, 18, 64, 85]: six RGB frames, matching the teacher.
    obs = torch.cat([load_image(p) for p in image_paths], dim=0)
    obs = obs.unsqueeze(0).to(device)
    goal = load_image(goal_path).unsqueeze(0).to(device)

    print("Trajectory:", TRAJECTORY)
    print("Observation frames:", FRAME_IDS)
    print("Goal frame:", GOAL_ID)
    print("Observation tensor:", tuple(obs.shape))
    print("Goal tensor:", tuple(goal.shape))

    with torch.inference_mode():
        teacher_dist, teacher_action = teacher(obs, goal)
        student_dist, student_action = student(obs, goal)

    for name, tensor in [
        ("Teacher distance", teacher_dist),
        ("Teacher waypoints", teacher_action),
        ("Student distance", student_dist),
        ("Student waypoints", student_action),
    ]:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or infinity")

    teacher_dist_np = teacher_dist[0].float().cpu().numpy().reshape(-1)
    student_dist_np = student_dist[0].float().cpu().numpy().reshape(-1)
    teacher_wp = teacher_action[0].float().cpu().numpy()
    student_wp = student_action[0].float().cpu().numpy()

    print("\n=== MODEL OUTPUTS ===")
    print("Teacher distance:", teacher_dist_np)
    print("Student distance:", student_dist_np)
    print("\nTeacher waypoints [5, 4]:\n", teacher_wp)
    print("\nStudent waypoints [5, 4]:\n", student_wp)

    distance_mse = F.mse_loss(student_dist, teacher_dist).item()
    waypoint_mse = F.mse_loss(student_action, teacher_action).item()
    position_rmse = np.sqrt(
        np.mean((student_wp[:, :2] - teacher_wp[:, :2]) ** 2)
    )

    print("\n=== INITIAL AGREEMENT (UNTRAINED STUDENT) ===")
    print(f"Distance MSE: {distance_mse:.6f}")
    print(f"Waypoint MSE: {waypoint_mse:.6f}")
    print(f"Position-coordinate RMSE: {position_rmse:.6f}")
    print("These are initial discrepancies, not navigation metrics.")

    # Convert normalized input tensors back to RGB for display.
    def display_image(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"ViNT Teacher vs Untrained Student — {TRAJECTORY}\n"
        f"Observation frames {FRAME_IDS}; goal frame {GOAL_ID}",
        fontsize=13,
    )

    # Six actual observation frames.
    for i, (ax, path) in enumerate(zip(axes[0, :3].tolist() +
                                      axes[1, :3].tolist(), image_paths)):
        ax.imshow(display_image(path))
        ax.set_title(f"Observation frame {FRAME_IDS[i]}")
        ax.axis("off")

    # Put the goal in the remaining top-right panel.
    axes[0, 3].imshow(display_image(goal_path))
    axes[0, 3].set_title(f"Goal frame {GOAL_ID}")
    axes[0, 3].axis("off")

    # The final bottom-right panel compares predicted local XY coordinates.
    ax = axes[1, 3]
    ax.plot(
        [0] + teacher_wp[:, 0].tolist(),
        [0] + teacher_wp[:, 1].tolist(),
        "o-",
        label="Teacher",
    )
    ax.plot(
        [0] + student_wp[:, 0].tolist(),
        [0] + student_wp[:, 1].tolist(),
        "s--",
        label="Untrained student",
    )
    ax.scatter([0], [0], marker="*", s=120, label="Current position")
    ax.set_title(
        "Predicted local XY waypoints\n"
        f"Teacher dist={teacher_dist_np[0]:.2f}; "
        f"Student dist={student_dist_np[0]:.2f}"
    )
    ax.set_xlabel("Local X (model output units)")
    ax.set_ylabel("Local Y (model output units)")
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nVisualization saved to:", OUTPUT_PATH.resolve())
    print("Open it with:")
    print(f"xdg-open '{OUTPUT_PATH.resolve()}'")


if __name__ == "__main__":
    main()
