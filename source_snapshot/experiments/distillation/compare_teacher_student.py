import torch
import torch.nn.functional as F

from experiments.distillation.student import ViNTStudent


TEACHER_PATH = "deployment/model_weights/vint.pth"


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load the serialized teacher model.
    checkpoint = torch.load(
        TEACHER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    teacher = checkpoint["model"]
    teacher = teacher.to(device).eval()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # Load a fresh student. No student checkpoint exists yet.
    student = ViNTStudent().to(device).eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"Teacher parameters: {teacher_params:,} ({teacher_params/1e6:.2f}M)")
    print(f"Student parameters: {student_params:,} ({student_params/1e6:.2f}M)")
    print(f"Parameter reduction: {(1-student_params/teacher_params)*100:.2f}%")

    # ViNT config: width=85, height=64.
    # Synthetic tensors are used only for interface validation.
    obs = torch.randn(2, 15, 64, 85, device=device)
    goal = torch.randn(2, 3, 64, 85, device=device)

    with torch.inference_mode():
        teacher_dist, teacher_action = teacher(obs, goal)
        student_dist, student_action = student(obs, goal)

    print("\nOutput shapes")
    print("Teacher distance:", tuple(teacher_dist.shape))
    print("Student distance:", tuple(student_dist.shape))
    print("Teacher actions:", tuple(teacher_action.shape))
    print("Student actions:", tuple(student_action.shape))

    assert teacher_dist.shape == student_dist.shape == (2, 1)
    assert teacher_action.shape == student_action.shape == (2, 5, 4)

    tensors = {
        "teacher distance": teacher_dist,
        "teacher actions": teacher_action,
        "student distance": student_dist,
        "student actions": student_action,
    }

    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite values in {name}")

    # These are initial, untrained-student discrepancies.
    dist_mse = F.mse_loss(student_dist, teacher_dist).item()
    action_mse = F.mse_loss(student_action, teacher_action).item()

    print("\nSynthetic-input comparison (student is untrained)")
    print(f"Distance prediction MSE: {dist_mse:.6f}")
    print(f"Waypoint prediction MSE: {action_mse:.6f}")
    print("\nInterface and numerical checks: PASS")
    print("Note: prediction errors are not meaningful quality metrics yet.")


if __name__ == "__main__":
    main()
