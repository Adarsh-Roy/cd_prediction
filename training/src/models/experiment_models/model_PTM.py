import torch
import torch.nn as nn

from src.models.components.pointnet_2d import PointNet2D
from src.models.components.transformer_encoder import TransformerSliceEncoder
from src.models.components.mlp_head import CdRegressor


class Cd_PTM_Model(nn.Module):
    """
    Full pipeline model:
    - Slice-wise point encoding via PointNet2D
    - Temporal encoding via Transformer
    - Regression head for Cd prediction
    """

    def __init__(
        self,
        slice_input_dim: int = 2,
        slice_emb_dim: int = 256,
        transformer_hidden_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.1,
        encoder_emb_dim: int = 256,
        max_num_slices: int = 80,
    ):
        super().__init__()

        self.slice_encoder = PointNet2D(
            input_dim=slice_input_dim, emb_dim=slice_emb_dim
        )

        self.temporal_encoder = TransformerSliceEncoder(
            input_dim=slice_emb_dim,
            hidden_dim=transformer_hidden_dim,
            num_layers=transformer_layers,
            nhead=transformer_heads,
            dropout=transformer_dropout,
            max_seq_len=max_num_slices,
        )

        self.head = CdRegressor(input_dim=encoder_emb_dim)

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: tuple of (slices, point_mask, slice_mask)
                - slices:     (B, S, P, 2)
                - point_mask: (B, S, P)
                - slice_mask: (B, S)

        Returns:
            Cd predictions: (B,) – one scalar per batch item
        """
        slices, point_mask, slice_mask = x
        B, S, P, D = slices.shape

        # Flatten slices for PointNet
        flat_pts = slices.view(B * S, P, D)  # (B·S, P, 2)
        flat_mask = point_mask.view(B * S, P)  # (B·S, P)

        # Encode each slice
        slice_embeds = self.slice_encoder(flat_pts, flat_mask)  # (B·S, emb_dim)
        slice_embeds = slice_embeds.view(B, S, -1)  # (B, S, emb_dim)

        # Zero out padded slices before transformer
        slice_embeds = slice_embeds * slice_mask.unsqueeze(-1)  # (B, S, emb_dim)

        # Temporal encoding via Transformer
        global_embedding = self.temporal_encoder(
            slice_embeds, slice_mask
        )  # (B, emb_dim)

        # Final Cd regression
        return self.head(global_embedding)  # (B,)

    @staticmethod
    def example_input(batch_size: int = 2, S: int = 80, P: int = 6500) -> tuple:
        """
        Returns a dummy batch for tracing or export.
        """
        slices = torch.randn(batch_size, S, P, 2)
        point_mask = torch.ones(batch_size, S, P)
        slice_mask = torch.ones(batch_size, S)
        return slices, point_mask, slice_mask
