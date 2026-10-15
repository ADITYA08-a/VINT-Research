import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FastViTStudentV3(nn.Module):
    """
    FastViT-T8 student for ViNT teacher distillation.

    Inputs:
        obs_img:  [B, 18, H, W] - six RGB observation frames
        goal_img: [B,  3, H, W] - one RGB goal image

    Outputs:
        distance: [B, 1]
        actions:  [B, 5, 4]
    """

    def __init__(
        self,
        pretrained=True,
        embed_dim=256,
        num_frames=6,
        num_waypoints=5,
        transformer_layers=2,
        transformer_heads=4,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.num_waypoints = num_waypoints
        self.embed_dim = embed_dim

        # Shared pretrained FastViT-T8 backbone.
        self.encoder = timm.create_model(
            "fastvit_t8",
            pretrained=pretrained,
            num_classes=0,
        )

        encoder_dim = self.encoder.num_features

        # Convert each frame's spatial feature map to one token.
        self.token_projection = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Six observation tokens + one goal token.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # Fuse all seven tokens, as in the V2 design.
        self.fusion = nn.Sequential(
            nn.Linear((num_frames + 1) * embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.distance_head = nn.Linear(128, 1)

        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_waypoints * 4),
        )

    def encode_images(self, images):
        """
        images: [B, N, 3, H, W]
        returns: [B, N, embed_dim]
        """
        batch_size, num_images, channels, height, width = images.shape

        images = images.reshape(
            batch_size * num_images,
            channels,
            height,
            width,
        )

        features = self.encoder.forward_features(images)

        # FastViT-T8 returns spatial feature maps [B*N, C, h, w].
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, output_size=1)
            features = features.flatten(1)
        elif features.ndim == 3:
            # Defensive support for token-based feature outputs.
            features = features.mean(dim=1)
        elif features.ndim != 2:
            raise RuntimeError(
                f"Unexpected FastViT feature shape: {tuple(features.shape)}"
            )

        features = self.token_projection(features)
        return features.reshape(batch_size, num_images, self.embed_dim)

    def forward(self, obs_img, goal_img):
        if obs_img.ndim != 4:
            raise ValueError(
                f"Expected obs_img [B,18,H,W], got {tuple(obs_img.shape)}"
            )

        if goal_img.ndim != 4:
            raise ValueError(
                f"Expected goal_img [B,3,H,W], got {tuple(goal_img.shape)}"
            )

        batch_size, obs_channels, height, width = obs_img.shape

        if obs_channels != self.num_frames * 3:
            raise ValueError(
                f"Expected {self.num_frames * 3} observation channels, "
                f"got {obs_channels}"
            )

        if goal_img.shape[0] != batch_size or goal_img.shape[1] != 3:
            raise ValueError("Goal image must have shape [B,3,H,W].")

        if goal_img.shape[-2:] != (height, width):
            raise ValueError("Observation and goal image sizes must match.")

        # [B,18,H,W] -> [B,6,3,H,W]
        obs_frames = obs_img.reshape(
            batch_size, self.num_frames, 3, height, width
        )

        # Encode all six observations and the goal in one backbone call.
        all_frames = torch.cat(
            [obs_frames, goal_img.unsqueeze(1)], dim=1
        )
        tokens = self.encode_images(all_frames)

        # [B,7,256]
        tokens = tokens + self.pos_embedding

        tokens = self.transformer(tokens)

        # [B,7*256] -> [B,128]
        fused = self.fusion(tokens.flatten(start_dim=1))

        distance = self.distance_head(fused)

        # [B,5,4]: x, y, heading component 1, heading component 2.
        actions = self.action_head(fused).reshape(
            batch_size, self.num_waypoints, 4
        )

        # Convert relative waypoint offsets into cumulative positions.
        xy = torch.cumsum(actions[..., :2], dim=1)

        # Normalize the two heading components.
        heading = F.normalize(
            actions[..., 2:4],
            p=2,
            dim=-1,
            eps=1e-6,
        )

        actions = torch.cat([xy, heading], dim=-1)

        return distance, actions


if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastViTStudentV3(pretrained=True).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Device:", device)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Parameter budget (<10M):", total_params < 10_000_000)

    batch_size = 2
    obs = torch.randn(batch_size, 18, 64, 85, device=device)
    goal = torch.randn(batch_size, 3, 64, 85, device=device)

    with torch.inference_mode():
        distance, actions = model(obs, goal)

    print("Observation shape:", tuple(obs.shape))
    print("Goal shape:", tuple(goal.shape))
    print("Distance shape:", tuple(distance.shape))
    print("Action shape:", tuple(actions.shape))
    print("Distance finite:", bool(torch.isfinite(distance).all()))
    print("Actions finite:", bool(torch.isfinite(actions).all()))

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Batch-1 inference benchmark.
        obs1 = obs[:1]
        goal1 = goal[:1]

        with torch.inference_mode():
            for _ in range(15):
                model(obs1, goal1)

            torch.cuda.synchronize()
            timings = []

            for _ in range(50):
                start = time.perf_counter()
                model(obs1, goal1)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)

        timings.sort()
        mean_ms = sum(timings) / len(timings)

        print(f"Batch-1 mean latency: {mean_ms:.3f} ms")
        print(f"Batch-1 median latency: {timings[len(timings)//2]:.3f} ms")
        print(
            "Peak allocated GPU memory:",
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB",
        )

    if total_params >= 10_000_000:
        print("WARNING: Model exceeds the 10M parameter target.")

    print("V3 smoke test completed.")
