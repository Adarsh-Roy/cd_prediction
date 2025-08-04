import torch
import torch.nn as nn

from src.models.components.pointnet_2d import PointNet2D
from src.models.components.lstm_encoder import LSTMEncoder
from src.models.components.mlp_head import CdRegressor


class Cd_PLM_Model(nn.Module):
    """
    Full pipeline model:
    - Slice-wise point encoding via PointNet2D
    - Temporal encoding via LSTM
    - Regression head for Cd prediction
    """

    def __init__(
        self,
        slice_input_dim: int = 2,
        slice_emb_dim: int = 256,
        lstm_hidden_dim: int = 256,
        design_emb_dim: int = 512,
    ):
        super().__init__()

        self.slice_encoder = PointNet2D(
            input_dim=slice_input_dim, emb_dim=slice_emb_dim
        )

        self.temporal_encoder = LSTMEncoder(
            input_dim=slice_emb_dim, hidden_dim=lstm_hidden_dim
        )

        self.head = CdRegressor(input_dim=design_emb_dim)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: tuple of (slices, point_mask)
                - slices:     (batch_size, num_slices, points_per_slice, 2)
                - point_mask: (batch_size, num_slices, points_per_slice)

        Returns:
            Cd predictions: (batch_size,) – one scaler per batch item
        """
        slices, point_mask = x
        batch_size, num_slices, points_per_slice, dim = slices.shape

        # Flatten slices for PointNet
        flat_pts = slices.view(
            batch_size * num_slices, points_per_slice, dim
        )  # (B·S, P, 2)
        flat_mask = point_mask.view(
            batch_size * num_slices, points_per_slice
        )  # (B·S, P)

        # Encode each slice
        slice_embs = self.slice_encoder(flat_pts, flat_mask)  # (B·S, emb_dim)
        slice_embs = slice_embs.view(batch_size, num_slices, -1)  # (B, S, emb_dim)

        # Temporal encoding via LSTM
        design_emb = self.temporal_encoder(slice_embs)  # (B, emb_dim)

        # Final Cd regression
        return self.head(design_emb)  # (B,)

    @staticmethod
    def example_input(batch_size: int = 2, S: int = 80, P: int = 6500) -> tuple:
        """
        Returns a dummy batch for tracing or export.
        """
        slices = torch.randn(batch_size, S, P, 2)
        point_mask = torch.ones(batch_size, S, P)
        return slices, point_mask
