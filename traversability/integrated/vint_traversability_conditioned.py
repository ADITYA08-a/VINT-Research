import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.traversability.integrated.vint_traversability_aware import (
    MultiScaleDecoder,
)


class ViNTTraversabilityConditioned(nn.Module):
    """
    Original ViNT plus pretrained multiscale traversability prediction
    and a learnable residual conditioning module.

    The original ViNT checkpoint and traversability decoder checkpoint
    are loaded separately. The original ViNT implementation is untouched.
    """

    def __init__(self, pretrained_vint, decoder_checkpoint):
        super().__init__()

        self.vint = pretrained_vint
        self.encoder = self.vint.obs_encoder
        self.obs_encoding_size = self.vint.obs_encoding_size

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

        # Convert the latest-frame spatial feature map into one
        # conditioning vector, with the same dimension as a ViNT token.
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, self.obs_encoding_size),
            nn.LayerNorm(self.obs_encoding_size),
        )

        # Start as an exact zero residual. Initially, traversability
        # cannot perturb the original observation token.
        self.conditioning_gate = nn.Parameter(torch.tensor([1e-3]))
        self.conditioning_projection = nn.Linear(
            self.obs_encoding_size,
            self.obs_encoding_size,
            bias=False,
        )
        nn.init.normal_(self.conditioning_projection.weight, mean=0.0, std=1e-3)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self._features[name] = output
        return hook

    def forward(self, obs_img, goal_img):
        # Original ViNT forward pass computes navigation predictions.
        # We reproduce its token construction to inject conditioning
        # before its existing Transformer decoder.
        vint = self.vint

        if vint.late_fusion:
            goal_encoding = vint.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat(
                [obs_img[:, 3 * vint.context_size:, :, :], goal_img],
                dim=1,
            )
            goal_encoding = vint.goal_encoder.extract_features(obsgoal_img)

        goal_encoding = vint.goal_encoder._avg_pooling(goal_encoding)

        if vint.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = vint.goal_encoder._dropout(goal_encoding)

        goal_encoding = vint.compress_goal_enc(goal_encoding)

        if goal_encoding.ndim == 2:
            goal_encoding = goal_encoding.unsqueeze(1)

        # Compute the original observation tokens.
        observation_frames = torch.split(obs_img, 3, dim=1)
        observation_frames = torch.cat(observation_frames, dim=0)

        obs_encoding = vint.obs_encoder.extract_features(observation_frames)
        obs_encoding = vint.obs_encoder._avg_pooling(obs_encoding)

        if vint.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = vint.obs_encoder._dropout(obs_encoding)

        obs_encoding = vint.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape(
            vint.context_size + 1,
            -1,
            vint.obs_encoding_size,
        ).transpose(0, 1)

        # Extract the spatial features from the latest frame only.
        latest_image = obs_img[:, -3:, :, :]
        self._features.clear()

        final_features = self.encoder.extract_features(latest_image)
        features = dict(self._features)
        features["final"] = final_features

        traversability_logits = self.traversability_decoder(
            features,
            output_size=latest_image.shape[-2:],
        )
        traversability = torch.sigmoid(traversability_logits)

        # A pooled representation of the predicted traversability map.
        spatial_vector = self.spatial_encoder(traversability)

        # Condition only the latest observation token.
        # The zero initialization preserves the original navigation
        # pathway at initialization.
        residual = self.conditioning_projection(spatial_vector)
        obs_encoding = torch.cat(
            [
                obs_encoding[:, :-1, :],
                obs_encoding[:, -1:, :]
                + self.conditioning_gate * residual.unsqueeze(1),
            ],
            dim=1,
        )

        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = vint.decoder(tokens)

        distance = vint.dist_predictor(final_repr)
        action = vint.action_predictor(final_repr)
        action = action.reshape(
            action.shape[0],
            vint.len_trajectory_pred,
            vint.num_action_params,
        )

        action[:, :, :2] = torch.cumsum(action[:, :, :2], dim=1)

        if vint.learn_angle:
            action[:, :, 2:] = F.normalize(
                action[:, :, 2:].clone(),
                dim=-1,
            )

        return {
            "distance": distance,
            "waypoints": action,
            "traversability_logits": traversability_logits,
            "traversability": traversability,
        }

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
