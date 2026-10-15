import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


class PositionalEncoding(nn.Module):
    def __init__(self, dim=256, max_len=7):
        super().__init__()

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

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


class ViNTStudentV2(nn.Module):
    """
    Compact ViNT student with pretrained MobileNetV3-Large
    and teacher-style full-sequence fusion.

    Inputs:
        obs_img:  [B, 18, H, W] -- six RGB frames
        goal_img: [B, 3, H, W]

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
        pretrained=True,
    ):
        super().__init__()

        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.embed_dim = embed_dim
        self.num_action_params = 4

        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained else None
        )
        backbone = mobilenet_v3_large(weights=weights)

        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(960, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.positional_encoding = PositionalEncoding(
            dim=embed_dim,
            max_len=context_size + 2,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attention_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )

        # Teacher-style fusion: use all seven fused tokens.
        fused_dim = (context_size + 2) * embed_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.distance_head = nn.Linear(32, 1)

        self.action_head = nn.Linear(
            32, len_traj_pred * self.num_action_params
        )

    def encode_images(self, images):
        features = self.encoder(images)
        features = self.pool(features).flatten(1)
        return self.projection(features)

    def forward(self, obs_img, goal_img):
        if obs_img.ndim != 4 or goal_img.ndim != 4:
            raise ValueError("Inputs must be batched image tensors.")

        batch_size = obs_img.shape[0]
        num_frames = self.context_size + 1

        if obs_img.shape[1] != 3 * num_frames:
            raise ValueError(
                f"Expected {3 * num_frames} observation channels, "
                f"got {obs_img.shape[1]}"
            )

        if goal_img.shape[1] != 3:
            raise ValueError("Goal image must have three channels.")

        if goal_img.shape[0] != batch_size:
            raise ValueError("Observation and goal batch sizes differ.")

        height, width = obs_img.shape[-2:]

        obs_frames = obs_img.reshape(
            batch_size, num_frames, 3, height, width
        ).reshape(batch_size * num_frames, 3, height, width)

        obs_features = self.encode_images(obs_frames)
        obs_features = obs_features.reshape(
            batch_size, num_frames, self.embed_dim
        )

        goal_features = self.encode_images(goal_img).unsqueeze(1)

        tokens = torch.cat(
            (obs_features, goal_features), dim=1
        )
        tokens = self.positional_encoding(tokens)
        tokens = self.temporal_fusion(tokens)

        # [B, 7, 256] -> [B, 1792]
        fused = tokens.reshape(batch_size, -1)
        representation = self.fusion_mlp(fused)

        dist_pred = self.distance_head(representation)

        action_pred = self.action_head(representation).reshape(
            batch_size,
            self.len_traj_pred,
            self.num_action_params,
        )

        # Match ViNT output conventions without in-place operations.
        positions = torch.cumsum(action_pred[:, :, :2], dim=1)
        angles = F.normalize(
            action_pred[:, :, 2:], dim=-1, eps=1e-6
        )

        action_pred = torch.cat((positions, angles), dim=-1)

        return dist_pred, action_pred


if __name__ == "__main__":
    torch.manual_seed(42)

    model = ViNTStudentV2(pretrained=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    obs = torch.randn(2, 18, 64, 85, device=device)
    goal = torch.randn(2, 3, 64, 85, device=device)

    with torch.no_grad():
        dist, actions = model(obs, goal)

    print("Device:", device)
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Distance shape:", tuple(dist.shape))
    print("Action shape:", tuple(actions.shape))
    print("Outputs finite:", bool(
        torch.isfinite(dist).all() and torch.isfinite(actions).all()
    ))
