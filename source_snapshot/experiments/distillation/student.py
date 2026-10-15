import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


class PositionalEncoding(nn.Module):
    def __init__(self, dim=256, max_len=7):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )

        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ViNTStudent(nn.Module):
    """
    Compact student for ViNT knowledge distillation.

    Inputs:
        obs_img:  [B, 18, H, W] — six RGB observation frames
        goal_img: [B, 3, H, W]  — goal image

    Outputs:
        dist_pred:   [B, 1]
        action_pred: [B, 5, 4]
    """

    def __init__(
        self,
        context_size=5,
        len_traj_pred=5,
        embed_dim=256,
        transformer_layers=2,
        attention_heads=4,
        learn_angle=True,
    ):
        super().__init__()

        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.num_action_params = 4 if learn_angle else 2
        self.embed_dim = embed_dim

        # One shared visual encoder for observation frames and goal.
        backbone = mobilenet_v3_large(weights=None)
        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Large's final feature map has 960 channels.
        self.projection = nn.Sequential(
            nn.Linear(960, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.positional_encoding = PositionalEncoding(
            dim=embed_dim,
            max_len=context_size + 2,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attention_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )

        # Flattened sequence is not necessary: use the final goal token.
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, len_traj_pred * self.num_action_params),
        )

    def encode_images(self, images):
        features = self.encoder(images)
        features = self.pool(features).flatten(1)
        return self.projection(features)

    def forward(self, obs_img, goal_img):
        if obs_img.ndim != 4 or goal_img.ndim != 4:
            raise ValueError("Expected batched image tensors [B,C,H,W].")

        num_obs_frames = self.context_size + 1
        expected_channels = 3 * num_obs_frames
        if obs_img.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} observation channels, "
                f"received {obs_img.shape[1]}."
            )
        if goal_img.shape[1] != 3:
            raise ValueError("Goal image must have 3 RGB channels.")

        batch_size = obs_img.shape[0]
        if goal_img.shape[0] != batch_size:
            raise ValueError("Observation and goal batch sizes must match.")

        # [B, 18, H, W] -> [B*6, 3, H, W]
        height, width = obs_img.shape[-2:]
        num_obs_frames = self.context_size + 1
        obs_frames = obs_img.reshape(
            batch_size, num_obs_frames, 3, height, width
        )
        obs_frames = obs_frames.reshape(
            batch_size * num_obs_frames, 3, height, width
        )

        # Encode all observations and goals using the same weights.
        obs_features = self.encode_images(obs_frames)
        obs_features = obs_features.reshape(
            batch_size, self.context_size + 1, self.embed_dim
        )
        goal_features = self.encode_images(goal_img).unsqueeze(1)

        tokens = torch.cat((obs_features, goal_features), dim=1)
        tokens = self.positional_encoding(tokens)
        tokens = self.temporal_fusion(tokens)

        # Goal-conditioned representation.
        representation = self.fusion_norm(tokens[:, -1])

        dist_pred = self.distance_head(representation)
        action_pred = self.action_head(representation).reshape(
            batch_size, self.len_traj_pred, self.num_action_params
        )

        # Match ViNT's output conventions.
        # Avoid in-place slice modifications: autograd needs the original
        # predictions to compute gradients correctly.
        position_deltas = action_pred[:, :, :2]
        positions = torch.cumsum(position_deltas, dim=1)

        if self.learn_angle:
            angle_vectors = F.normalize(
                action_pred[:, :, 2:], dim=-1, eps=1e-6
            )
            action_pred = torch.cat(
                (positions, angle_vectors), dim=-1
            )
        else:
            action_pred = positions

        return dist_pred, action_pred


if __name__ == "__main__":
    torch.manual_seed(42)

    model = ViNTStudent()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total:,} ({total / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()

    # Synthetic smoke test only; this is not training data.
    obs = torch.randn(2, 18, 64, 85, device=device)
    goal = torch.randn(2, 3, 64, 85, device=device)

    dist_pred, action_pred = model(obs, goal)

    assert dist_pred.shape == (2, 1), dist_pred.shape
    assert action_pred.shape == (2, 5, 4), action_pred.shape
    assert torch.isfinite(dist_pred).all()
    assert torch.isfinite(action_pred).all()

    loss = dist_pred.square().mean() + action_pred.square().mean()
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )

    print(f"Device:               {device}")
    print(f"Distance output:      {tuple(dist_pred.shape)}")
    print(f"Waypoint output:      {tuple(action_pred.shape)}")
    print(f"Smoke-test loss:      {loss.item():.6f}")
    print("Forward/backward test: PASS")
