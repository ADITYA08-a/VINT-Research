import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDecoder(nn.Module):
    """Multiscale traversability decoder matching the trained checkpoint."""

    def __init__(self, projection_channels=64):
        super().__init__()

        input_channels = {
            "block_1": 24,
            "block_3": 40,
            "block_5": 80,
            "final": 1280,
        }

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(channels, projection_channels, 1, bias=False),
                nn.GroupNorm(8, projection_channels),
                nn.ReLU(inplace=True),
            )
            for name, channels in input_channels.items()
        })

        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, features, output_size):
        target_size = features["block_1"].shape[-2:]
        projected = []

        for name in ("block_1", "block_3", "block_5", "final"):
            x = self.projections[name](features[name])

            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            projected.append(x)

        x = self.fusion(torch.cat(projected, dim=1))
        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class ViNTTraversabilityAware(nn.Module):
    """
    Auxiliary-only integration stage.

    Navigation predictions come directly from the original ViNT model.
    Traversability is predicted from the latest RGB observation frame.

    Input image tensors must use the same [0, 1] RGB preprocessing as the
    multiscale decoder training script.
    """

    def __init__(self, pretrained_vint, decoder_checkpoint):
        super().__init__()

        self.vint = pretrained_vint
        self.encoder = self.vint.obs_encoder
        self.traversability_decoder = MultiScaleDecoder()

        checkpoint = torch.load(
            decoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        self.traversability_decoder.load_state_dict(
            checkpoint["decoder"],
            strict=True,
        )

        self.feature_names = ("block_1", "block_3", "block_5")
        self._features = {}
        self._handles = []

        for name, index in (
            ("block_1", 1),
            ("block_3", 3),
            ("block_5", 5),
        ):
            handle = self.encoder._blocks[index].register_forward_hook(
                self._make_hook(name)
            )
            self._handles.append(handle)

        self.decoder_metadata = {
            key: checkpoint.get(key)
            for key in (
                "epoch",
                "best_val_loss",
                "feature_names",
                "input_size",
                "target_type",
            )
        }

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self._features[name] = output
        return hook

    def forward(self, obs_img, goal_img):
        # Keep the original navigation pathway and outputs unchanged.
        navigation_output = self.vint(obs_img, goal_img)

        # Use the latest RGB observation for the front-facing map.
        latest_image = obs_img[:, -3:, :, :]
        self._features.clear()

        # The pretrained encoder and decoder are inference-only initially.
        with torch.no_grad():
            final_features = self.encoder.extract_features(latest_image)

        features = dict(self._features)
        features["final"] = final_features

        expected = {
            "block_1", "block_3", "block_5", "final"
        }
        if set(features) != expected:
            raise RuntimeError(
                "Incorrect feature set: "
                f"expected {expected}, got {set(features)}"
            )

        with torch.no_grad():
            traversability_logits = self.traversability_decoder(
                features,
                output_size=latest_image.shape[-2:],
            )

        traversability = torch.sigmoid(traversability_logits)

        return {
            "distance": navigation_output[0],
            "waypoints": navigation_output[1],
            "traversability_logits": traversability_logits,
            "traversability": traversability,
        }

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
