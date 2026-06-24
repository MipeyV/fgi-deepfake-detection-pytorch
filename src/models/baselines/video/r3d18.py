"""Torchvision R3D-18 video classifier."""

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from src.models.baselines.video._validation import validate_positive_int

R3D18_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
R3D18_KINETICS_STD = (0.22803, 0.22145, 0.216989)


class R3D18VideoClassifier(nn.Module):
    """Torchvision R3D-18 adapted to the shared video classifier contract."""

    def __init__(
        self,
        num_classes: int = 2,
        weights: str = "none",
        dropout: float = 0.3,
        normalize: bool = True,
    ) -> None:
        """Initialize an R3D-18 backbone and replace its classification head.

        Args:
            num_classes: Number of output classification logits.
            weights: ``none`` or ``kinetics400_v1``.
            dropout: Dropout probability before the final linear layer.
            normalize: Whether to apply Kinetics-400 channel normalization.

        Raises:
            ValueError: If a parameter is invalid or the weight name is
                unsupported.
        """
        super().__init__()

        validate_positive_int(num_classes, "num_classes")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        normalized_weights = weights.lower()
        if normalized_weights == "none":
            torchvision_weights = None
        elif normalized_weights == "kinetics400_v1":
            torchvision_weights = R3D_18_Weights.KINETICS400_V1
        else:
            raise ValueError("R3D-18 weights must be 'none' or 'kinetics400_v1'")

        self.num_classes = num_classes
        self.weights = normalized_weights
        self.dropout = dropout
        self.normalize = normalize
        self.backbone = r3d_18(weights=torchvision_weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.register_buffer(
            "input_mean",
            torch.tensor(R3D18_KINETICS_MEAN).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(R3D18_KINETICS_STD).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classify clips shaped ``[batch, frames, channels, height, width]``.

        Args:
            inputs: RGB frame sequences with values normally in ``[0, 1]``.

        Returns:
            Unnormalized class logits shaped ``[batch, num_classes]``.

        Raises:
            ValueError: If the input is not five-dimensional or does not
                contain three RGB channels.
        """
        if inputs.ndim != 5:
            raise ValueError(
                "R3D18VideoClassifier expects inputs with shape "
                "[batch_size, num_frames, channels, height, width]"
            )

        if inputs.shape[2] != 3:
            raise ValueError("R3D18VideoClassifier expects 3-channel RGB frames")

        inputs = inputs.permute(0, 2, 1, 3, 4)
        if self.normalize:
            inputs = (inputs - self.input_mean) / self.input_std
        return self.backbone(inputs)
