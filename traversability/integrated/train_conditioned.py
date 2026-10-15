import os
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from vint_train.models.vint.vint import ViNT
from vint_train.data.vint_dataset import ViNT_Dataset

from experiments.traversability.integrated.vint_traversability_conditioned import (
    ViNTTraversabilityConditioned,
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path.home() / "visualnav-transformer"
OUT = ROOT / "experiments/traversability/integrated/run1"

DATA_ROOT = Path.home() / "Downloads/go_stanford"
SPLITS_ROOT = Path.home() / "Downloads/vint_data_splits/go_stanford"

VINT_CKPT = ROOT / "deployment/model_weights/vint.pth"
DECODER_CKPT = (
    ROOT
    / "experiments/traversability/multiscale_v2_1000/best_decoder.pt"
)
CONFIG_PATH = ROOT / "train/config/vint.yaml"
DATA_CONFIG_PATH = ROOT / "train/vint_train/data/data_config.yaml"

SEED = 42
BATCH_SIZE = 4
NUM_WORKERS = 2
EPOCHS = 1
LR = 1e-5

# Smoke-test limits. Increase only after this run succeeds.
MAX_TRAIN_BATCHES = 200
MAX_VAL_BATCHES = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUT.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Experiment directory:", OUT)


# ============================================================
# Validate inputs
# ============================================================

for path in [
    DATA_ROOT,
    SPLITS_ROOT / "train/traj_names.txt",
    SPLITS_ROOT / "test/traj_names.txt",
    VINT_CKPT,
    DECODER_CKPT,
    CONFIG_PATH,
    DATA_CONFIG_PATH,
]:
    if not path.exists():
        raise FileNotFoundError("Required path not found: {}".format(path))


# ============================================================
# Create a deterministic trajectory-level train/validation split.
# The official test split is never used for model selection.
# ============================================================

with open(SPLITS_ROOT / "train/traj_names.txt", "r") as f:
    official_train = [
        line.strip() for line in f if line.strip()
    ]

official_train = sorted(set(official_train))
rng = random.Random(SEED)
rng.shuffle(official_train)

n_val = max(1, int(round(0.10 * len(official_train))))
val_names = sorted(official_train[:n_val])
train_names = sorted(official_train[n_val:])

split_root = OUT / "splits"
train_split_dir = split_root / "train"
val_split_dir = split_root / "val"

train_split_dir.mkdir(parents=True, exist_ok=True)
val_split_dir.mkdir(parents=True, exist_ok=True)

with open(train_split_dir / "traj_names.txt", "w") as f:
    f.write("\n".join(train_names) + "\n")

with open(val_split_dir / "traj_names.txt", "w") as f:
    f.write("\n".join(val_names) + "\n")

with open(OUT / "split_manifest.pkl", "wb") as f:
    pickle.dump(
        {
            "seed": SEED,
            "train_trajectories": train_names,
            "val_trajectories": val_names,
            "official_test_used": False,
        },
        f,
    )

print(
    "Trajectory split: train={}, val={}".format(
        len(train_names), len(val_names)
    )
)
print("Official test trajectories are not used for model selection.")


# ============================================================
# Load configuration
# ============================================================

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

with open(DATA_CONFIG_PATH, "r") as f:
    all_data_cfg = yaml.safe_load(f)

if "go_stanford" not in all_data_cfg:
    raise KeyError(
        "go_stanford is missing from {}".format(DATA_CONFIG_PATH)
    )

dcfg = all_data_cfg["go_stanford"]

common = dict(
    data_folder=str(DATA_ROOT),
    dataset_name="go_stanford",
    image_size=tuple(cfg["image_size"]),
    waypoint_spacing=dcfg.get("waypoint_spacing", 1),
    min_dist_cat=dcfg["distance"]["min_dist_cat"],
    max_dist_cat=dcfg["distance"]["max_dist_cat"],
    min_action_distance=dcfg["action"]["min_dist_cat"],
    max_action_distance=dcfg["action"]["max_dist_cat"],
    negative_mining=dcfg.get("negative_mining", True),
    len_traj_pred=cfg["len_traj_pred"],
    learn_angle=cfg["learn_angle"],
    context_size=cfg["context_size"],
    context_type=cfg.get("context_type", "temporal"),
    end_slack=dcfg.get("end_slack", 0),
    goals_per_obs=dcfg.get("goals_per_obs", 2),
    normalize=cfg.get("normalize", True),
)


# ============================================================
# Normalize Go-Stanford metadata in memory only.
#
# Some trajectory files store yaw as arrays inside object arrays.
# Positions may also have object dtype. ViNT's waypoint math
# expects ordinary numeric arrays.
#
# No source trajectory files are modified.
# ============================================================

_original_get_trajectory = ViNT_Dataset._get_trajectory


def _get_trajectory_numeric(self, trajectory_name):
    data = _original_get_trajectory(self, trajectory_name)

    if "yaw" in data:
        yaw = np.asarray(data["yaw"], dtype=object).reshape(-1)
        yaw_values = []

        for value in yaw:
            scalar_array = np.asarray(value).reshape(-1)
            if scalar_array.size == 0:
                raise ValueError(
                    "Empty yaw value in trajectory {}".format(
                        trajectory_name
                    )
                )
            yaw_values.append(float(scalar_array[0]))

        data["yaw"] = np.asarray(yaw_values, dtype=np.float64)

    if "position" in data:
        position = np.asarray(data["position"], dtype=object)
        position_values = []

        for value in position:
            point = np.asarray(value, dtype=np.float64).reshape(-1)

            if point.size < 2:
                raise ValueError(
                    "Invalid position in trajectory {}: {}".format(
                        trajectory_name, point
                    )
                )

            position_values.append(point[:2])

        data["position"] = np.asarray(
            position_values, dtype=np.float64
        ).reshape(-1, 2)

    return data


ViNT_Dataset._get_trajectory = _get_trajectory_numeric


# ============================================================
# Build datasets
# ============================================================

print("Building ViNT datasets and image caches...")

train_ds = ViNT_Dataset(
    data_split_folder=str(train_split_dir),
    **common
)

val_ds = ViNT_Dataset(
    data_split_folder=str(val_split_dir),
    **common
)

print(
    "Dataset samples: train={}, val={}".format(
        len(train_ds), len(val_ds)
    )
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)


# ============================================================
# Load original pretrained ViNT strictly
# ============================================================

vint = ViNT(
    context_size=cfg["context_size"],
    len_traj_pred=cfg["len_traj_pred"],
    learn_angle=cfg["learn_angle"],
    obs_encoder=cfg["obs_encoder"],
    obs_encoding_size=cfg["obs_encoding_size"],
    late_fusion=cfg["late_fusion"],
    mha_num_attention_heads=cfg[" mha_num_attention_heads"]
    if " mha_num_attention_heads" in cfg
    else cfg["mha_num_attention_heads"],
    mha_num_attention_layers=cfg["mha_num_attention_layers"],
).to(DEVICE)

checkpoint = torch.load(
    str(VINT_CKPT),
    map_location="cpu",
    weights_only=False,
)

saved_model = checkpoint.get("model", checkpoint.get("state_dict"))

if isinstance(saved_model, torch.nn.Module):
    vint.load_state_dict(saved_model.state_dict(), strict=True)
elif isinstance(saved_model, dict):
    if saved_model and all(
        key.startswith("module.") for key in saved_model
    ):
        saved_model = {
            key[len("module."):]: value
            for key, value in saved_model.items()
        }
    vint.load_state_dict(saved_model, strict=True)
else:
    raise TypeError(
        "Unexpected ViNT checkpoint payload: {}".format(
            type(saved_model)
        )
    )

print("Original ViNT checkpoint loaded strictly.")


# ============================================================
# Construct conditioned model
# ============================================================

model = ViNTTraversabilityConditioned(
    pretrained_vint=vint,
    decoder_checkpoint=str(DECODER_CKPT),
).to(DEVICE)

# Freeze the original navigation model and the pretrained map decoder.
for parameter in model.vint.parameters():
    parameter.requires_grad = False

for parameter in model.decoder.parameters():
    parameter.requires_grad = False

# Only the traversability-conditioning adapter is trained.
for module_name in [
    "spatial_encoder",
    "conditioning_projection",
]:
    module = getattr(model, module_name)
    for parameter in module.parameters():
        parameter.requires_grad = True

model.conditioning_gate.requires_grad = True

trainable_params = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print(
    "Trainable parameter count: {}".format(
        sum(p.numel() for p in trainable_params)
    )
)

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LR,
    weight_decay=1e-4,
)


# ============================================================
# Prepare data
# ============================================================

image_mean = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(1, 3, 1, 1)

image_std = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(1, 3, 1, 1)


def prepare(batch):
    obs_img, goal_img, actions, distance, goal_pos, idx, action_mask = batch

    obs_img = obs_img.to(DEVICE, non_blocking=True).float()
    goal_img = goal_img.to(DEVICE, non_blocking=True).float()
    actions = actions.to(DEVICE, non_blocking=True).float()
    distance = distance.to(DEVICE, non_blocking=True).long()
    action_mask = action_mask.to(DEVICE, non_blocking=True).bool()

    # Dataset images are channel-first histories. Normalize every
    # RGB frame using the original ViNT ImageNet preprocessing.
    if obs_img.ndim != 5:
        raise ValueError(
            "Expected observation shape (B, context, C, H, W); got {}".format(
                tuple(obs_img.shape)
            )
        )

    batch_size, context, channels, height, width = obs_img.shape

    if channels != 3:
        raise ValueError(
            "Expected RGB observations; got {} channels".format(channels)
        )

    obs_flat = obs_img.reshape(
        batch_size * context, channels, height, width
    )
    obs_flat = (
        obs_flat - image_mean.to(DEVICE)
    ) / image_std.to(DEVICE)

    obs_img = obs_flat.reshape(
        batch_size, context, channels, height, width
    )

    goal_img = (
        goal_img - image_mean.to(DEVICE)
    ) / image_std.to(DEVICE)

    return obs_img, goal_img, actions, distance, action_mask


# ============================================================
# Loss
# ============================================================

alpha = float(cfg.get("alpha", 0.5))


def compute_loss(outputs, actions, distance, action_mask):
    predicted_distance = outputs["distance"]
    predicted_actions = outputs["waypoints"]

    distance_loss = torch.nn.functional.mse_loss(
        predicted_distance, distance.float()
    )

    if predicted_actions.shape != actions.shape:
        raise ValueError(
            "Action shape mismatch: prediction {}, target {}".format(
                tuple(predicted_actions.shape), tuple(actions.shape)
            )
        )

    per_element = (
        predicted_actions - actions
    ).pow(2)

    # action_mask is per example. Exclude masked examples.
    mask = action_mask.float().view(-1, 1, 1)
    denominator = (
        mask.sum() * actions.shape[1] * actions.shape[2]
    ).clamp_min(1.0)

    action_loss = (per_element * mask).sum() / denominator

    total_loss = (
        alpha * 1e-2 * distance_loss
        + (1.0 - alpha) * action_loss
    )

    return total_loss, distance_loss.detach(), action_loss.detach()


# ============================================================
# Gradient sanity check
# ============================================================

print("Running gradient sanity check...")

model.train()
# Keep the frozen pretrained components in evaluation mode.
model.vint.eval()
model.decoder.eval()

first_batch = next(iter(train_loader))
obs, goal, actions, distance, action_mask = prepare(first_batch)

optimizer.zero_grad(set_to_none=True)
outputs = model(obs, goal)
loss, dist_loss, action_loss = compute_loss(
    outputs, actions, distance, action_mask
)

if not torch.isfinite(loss):
    raise RuntimeError("Non-finite loss in gradient sanity check.")

loss.backward()

grad_sq_sum = 0.0
grad_count = 0

for parameter in trainable_params:
    if parameter.grad is not None:
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite adapter gradient detected.")
        grad_sq_sum += grad.norm().item() ** 2
        grad_count += 1

grad_norm = grad_sq_sum ** 0.5

print(
    "Gradient check: loss={:.6f}, distance={:.6f}, "
    "action={:.6f}, grad_norm={:.6e}, tensors={}".format(
        loss.item(),
        dist_loss.item(),
        action_loss.item(),
        grad_norm,
        grad_count,
    )
)

if grad_count == 0 or grad_norm == 0.0:
    raise RuntimeError(
        "No nonzero gradients reached the conditioning adapter."
    )

optimizer.step()
print("Gradient check passed; one optimizer step completed.")


# ============================================================
# Train / validation smoke epoch
# ============================================================

def run_epoch(loader, training, max_batches):
    if training:
        model.train()
        # The navigation backbone and decoder remain frozen/eval.
        model.vint.eval()
        model.decoder.eval()
    else:
        model.eval()

    total_loss = 0.0
    total_dist = 0.0
    total_action = 0.0
    batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        obs, goal, actions, distance, action_mask = prepare(batch)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(obs, goal)
            loss, dist_loss, action_loss = compute_loss(
                outputs, actions, distance, action_mask
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss at batch {}".format(batch_idx)
                )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0
                )
                optimizer.step()

        total_loss += loss.detach().item()
        total_dist += dist_loss.item()
        total_action += action_loss.item()
        batches += 1

        if (batch_idx + 1) % 25 == 0:
            print(
                "{} batch {}/{} loss={:.6f}".format(
                    "train" if training else "val",
                    batch_idx + 1,
                    max_batches if max_batches is not None else len(loader),
                    loss.item(),
                )
            )

    if batches == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "loss": total_loss / batches,
        "distance_loss": total_dist / batches,
        "action_loss": total_action / batches,
        "batches": batches,
    }


history = []

for epoch in range(EPOCHS):
    print("\nEpoch {}/{}".format(epoch + 1, EPOCHS))

    train_metrics = run_epoch(
        train_loader, True, MAX_TRAIN_BATCHES
    )
    val_metrics = run_epoch(
        val_loader, False, MAX_VAL_BATCHES
    )

    record = {
        "epoch": epoch + 1,
        "train": train_metrics,
        "val": val_metrics,
    }
    history.append(record)

    print("Train:", train_metrics)
    print("Val:  ", val_metrics)

    # Save only this experiment's adapter/model state.
    save_path = OUT / "latest.pth"
    torch.save(
        {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "seed": SEED,
            "train_trajectories": len(train_names),
            "val_trajectories": len(val_names),
            "official_test_used": False,
        },
        str(save_path),
    )

    with open(OUT / "history.yaml", "w") as f:
        yaml.safe_dump(history, f, sort_keys=False)

    print("Saved:", save_path)

print("\nSmoke training completed.")
print("No official test trajectories were used.")
