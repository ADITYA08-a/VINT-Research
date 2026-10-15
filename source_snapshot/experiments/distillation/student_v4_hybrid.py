import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FastViTStudentV4Hybrid(nn.Module):
    """
    Hybrid spatial-temporal attention student.

    Inputs:
        obs_img: [B, 18, H, W] -- six RGB observations
        goal_img: [B,  3, H, W] -- goal RGB image

    Outputs:
        distance: [B, 1]
        actions:  [B, 5, 4]

    Architecture:
        FastViT-T8
        -> retain 2x3 spatial feature grid per image
        -> observation self-attention
        -> goal-to-observation cross-attention
        -> distance and waypoint heads
    """

    def __init__(
        self,
        pretrained=True,
        embed_dim=384,
        num_frames=6,
        num_waypoints=5,
        self_attention_layers=3,
        attention_heads=4,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.num_waypoints = num_waypoints
        self.embed_dim = embed_dim

        # Shared visual encoder for observations and goal.
        self.encoder = timm.create_model(
            "fastvit_t8",
            pretrained=pretrained,
            num_classes=0,
        )
        encoder_dim = self.encoder.num_features

        # Project each spatial location into a 384-D token.
        self.token_projection = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # FastViT-T8 produces a 2x3 spatial feature grid at 64x85.
        self.spatial_positions = nn.Parameter(
            torch.zeros(1, 1, 6, embed_dim)
        )
        self.temporal_positions = nn.Parameter(
            torch.zeros(1, num_frames, 1, embed_dim)
        )

        nn.init.trunc_normal_(self.spatial_positions, std=0.02)
        nn.init.trunc_normal_(self.temporal_positions, std=0.02)

        # Contextualize the observation history across space and time.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attention_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.observation_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self_attention_layers,
        )

        # Goal spatial tokens query the contextualized observation tokens.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(embed_dim)

        # Refine each goal-conditioned spatial token.
        self.refinement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.refinement_norm = nn.LayerNorm(embed_dim)

        # Pool goal-conditioned tokens before predicting.
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_waypoints * 4),
        )

    def encode_spatial_tokens(self, images):
        """
        images: [B, N, 3, H, W]
        returns: [B, N, 6, embed_dim]
        """
        batch_size, num_images, channels, height, width = images.shape

        images = images.reshape(
            batch_size * num_images,
            channels,
            height,
            width,
        )

        features = self.encoder.forward_features(images)

        if features.ndim != 4:
            raise RuntimeError(
                "Expected FastViT spatial feature map [B,C,H,W], "
                f"got {tuple(features.shape)}"
            )

        # Preserve the spatial grid instead of global-average pooling.
        features = F.adaptive_avg_pool2d(features, output_size=(2, 3))

        _, channels, grid_h, grid_w = features.shape
        features = features.flatten(2).transpose(1, 2)

        # [B*N, 6, encoder_dim] -> [B, N, 6, embed_dim]
        features = self.token_projection(features)
        return features.reshape(
            batch_size, num_images, grid_h * grid_w, self.embed_dim
        )

    def forward(self, obs_img, goal_img):
        if obs_img.ndim != 4 or goal_img.ndim != 4:
            raise ValueError("Expected batched observation and goal images.")

        batch_size, obs_channels, height, width = obs_img.shape

        if obs_channels != self.num_frames * 3:
            raise ValueError(
                f"Expected {self.num_frames * 3} observation channels; "
                f"got {obs_channels}"
            )

        if goal_img.shape != (batch_size, 3, height, width):
            raise ValueError(
                "Goal image must have shape [B,3,H,W] matching observations."
            )

        obs_frames = obs_img.reshape(
            batch_size, self.num_frames, 3, height, width
        )

        # One shared-backbone call for all six observations and the goal.
        all_images = torch.cat(
            [obs_frames, goal_img.unsqueeze(1)], dim=1
        )
        all_tokens = self.encode_spatial_tokens(all_images)

        obs_tokens = all_tokens[:, :self.num_frames]
        goal_tokens = all_tokens[:, self.num_frames]

        # Add spatial position information to every image.
        obs_tokens = obs_tokens + self.spatial_positions
        goal_tokens = goal_tokens + self.spatial_positions[:, 0]

        # Add temporal positions to observation frames.
        obs_tokens = obs_tokens + self.temporal_positions

        # Six frames x six spatial locations = 36 observation tokens.
        obs_tokens = obs_tokens.reshape(
            batch_size, self.num_frames * 6, self.embed_dim
        )

        # Self-attention among observation tokens.
        obs_tokens = self.observation_encoder(obs_tokens)

        # Six goal spatial tokens query all 36 observation tokens.
        attended, _ = self.cross_attention(
            query=goal_tokens,
            key=obs_tokens,
            value=obs_tokens,
            need_weights=False,
        )

        goal_tokens = self.cross_attention_norm(
            goal_tokens + attended
        )

        goal_tokens = self.refinement_norm(
            goal_tokens + self.refinement(goal_tokens)
        )

        # Aggregate the six goal-conditioned spatial tokens.
        fused = goal_tokens.mean(dim=1)

        distance = self.distance_head(fused)

        actions = self.action_head(fused).reshape(
            batch_size, self.num_waypoints, 4
        )

        # Match the existing student output convention.
        xy = torch.cumsum(actions[..., :2], dim=1)
        heading = F.normalize(
            actions[..., 2:4],
            p=2,
            dim=-1,
            eps=1e-6,
        )
        actions = torch.cat([xy, heading], dim=-1)

        return distance, actions


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastViTStudentV4Hybrid(pretrained=True).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Device:", device)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Within 10-12M target:", 10_000_000 <= total_params <= 12_000_000)

    obs = torch.randn(2, 18, 64, 85, device=device)
    goal = torch.randn(2, 3, 64, 85, device=device)

    with torch.inference_mode():
        distance, actions = model(obs, goal)

    print("Distance shape:", tuple(distance.shape))
    print("Actions shape:", tuple(actions.shape))
    print("Distance finite:", bool(torch.isfinite(distance).all()))
    print("Actions finite:", bool(torch.isfinite(actions).all()))

    if device.type == "cuda":
        obs1, goal1 = obs[:1], goal[:1]

        with torch.inference_mode():
            for _ in range(10):
                model(obs1, goal1)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []

            for _ in range(30):
                start = time.perf_counter()
                model(obs1, goal1)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)

        timings.sort()
        print(f"Batch-1 mean latency: {sum(timings) / len(timings):.3f} ms")
        print(f"Batch-1 median latency: {timings[len(timings)//2]:.3f} ms")
        print(
            "Peak allocated GPU memory:",
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB",
        )

    print("Hybrid V4 smoke test completed.")
