import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FastViTStudentV4(nn.Module):
    """
    FastViT-T8 student with goal-to-observation cross-attention.

    Inputs:
        obs_img: [B, 18, H, W] -- six RGB observation frames
        goal_img: [B,  3, H, W] -- goal RGB image

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
        attention_heads=4,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.num_waypoints = num_waypoints
        self.embed_dim = embed_dim

        # Shared FastViT-T8 encoder for observations and goal.
        self.encoder = timm.create_model(
            "fastvit_t8",
            pretrained=pretrained,
            num_classes=0,
        )

        encoder_dim = self.encoder.num_features

        # Convert each image's feature map into one token.
        self.token_projection = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Six temporal observation positions + one goal position.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Goal token queries the six observation tokens.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.cross_attention_norm = nn.LayerNorm(embed_dim)

        # Compact feed-forward refinement after cross-attention.
        self.refinement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.refinement_norm = nn.LayerNorm(embed_dim)

        # Both heads use the goal-conditioned representation.
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
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

        if features.ndim == 4:
            # FastViT-T8 spatial feature maps.
            features = F.adaptive_avg_pool2d(
                features, output_size=1
            ).flatten(1)
        elif features.ndim == 3:
            # Defensive support for token-based feature outputs.
            features = features.mean(dim=1)
        elif features.ndim != 2:
            raise RuntimeError(
                f"Unexpected FastViT feature shape: {tuple(features.shape)}"
            )

        features = self.token_projection(features)

        return features.reshape(
            batch_size, num_images, self.embed_dim
        )

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

        # Encode all seven images in one backbone call.
        all_frames = torch.cat(
            [obs_frames, goal_img.unsqueeze(1)], dim=1
        )
        tokens = self.encode_images(all_frames)
        tokens = tokens + self.pos_embedding

        # Separate observation history and goal representation.
        obs_tokens = tokens[:, :self.num_frames, :]   # [B,6,256]
        goal_token = tokens[:, self.num_frames:, :]  # [B,1,256]

        # Goal is Q; observations are K and V.
        attended, attention_weights = self.cross_attention(
            query=goal_token,
            key=obs_tokens,
            value=obs_tokens,
            need_weights=False,
        )

        # Residual connection around cross-attention.
        fused = self.cross_attention_norm(
            goal_token + attended
        )

        # Feed-forward refinement with another residual connection.
        fused = self.refinement_norm(
            fused + self.refinement(fused)
        )

        # [B,1,256] -> [B,256]
        fused = fused.squeeze(1)

        distance = self.distance_head(fused)

        # [B,5,4]: x, y, heading component 1, heading component 2.
        actions = self.action_head(fused).reshape(
            batch_size, self.num_waypoints, 4
        )

        # Convert predicted relative offsets to cumulative XY positions.
        xy = torch.cumsum(actions[..., :2], dim=1)

        # Normalize each waypoint's two heading components.
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

    model = FastViTStudentV4(pretrained=True).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Device:", device)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Parameter budget (<10M):", total_params < 10_000_000)

    obs = torch.randn(2, 18, 64, 85, device=device)
    goal = torch.randn(2, 3, 64, 85, device=device)

    with torch.inference_mode():
        distance, actions = model(obs, goal)

    print("Observation shape:", tuple(obs.shape))
    print("Goal shape:", tuple(goal.shape))
    print("Distance shape:", tuple(distance.shape))
    print("Action shape:", tuple(actions.shape))
    print("Distance finite:", bool(torch.isfinite(distance).all()))
    print("Actions finite:", bool(torch.isfinite(actions).all()))

    if device.type == "cuda":
        obs1, goal1 = obs[:1], goal[:1]

        with torch.inference_mode():
            for _ in range(15):
                model(obs1, goal1)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []

            for _ in range(50):
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

    print("V4 smoke test completed.")
