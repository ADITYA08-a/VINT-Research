import torch
import torch.nn as nn
import torch.nn.functional as F


class TraversabilityDecoder(nn.Module):
    """
    Lightweight decoder for pixel-wise traversability prediction.

    Input:
        Spatial features from ViNT's EfficientNet-B0:
        [B, 1280, Hf, Wf]

    Output:
        Pixel-wise class logits:
        [B, num_classes, H, W]
    """

    def __init__(self, in_channels=1280, num_classes=2):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, features, output_size):
        # Gradually increase the spatial resolution.
        x = F.interpolate(
            features,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.decoder(x)

        # Match the input image's exact dimensions.
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


class ViNTTraversability(nn.Module):
    """
    Traversability branch that reuses the pretrained ViNT encoder.

    The original ViNT navigation model is not modified.
    """

    def __init__(self, pretrained_vint, freeze_encoder=True):
        super().__init__()

        # Reuse the exact encoder object from the pretrained ViNT model.
        self.encoder = pretrained_vint.obs_encoder

        self.traversability_decoder = TraversabilityDecoder(
            in_channels=self.encoder._fc.in_features,
            num_classes=2,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W]

        Returns:
            logits: [B, 2, H, W]
        """

        features = self.encoder.extract_features(image)

        logits = self.traversability_decoder(
            features,
            output_size=image.shape[-2:],
        )

        return logits
