import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.models.components.dgcnn import EdgeConvSliceEncoder
from src.models.components.transformer_encoder import TransformerSliceEncoder
from src.models.components.mlp_head import CdRegressor


class Cd_DTM_Model(nn.Module):
    """
    Dynamic Graph CNN + Transformer + MLP model.
    Accepts a list of 80 PyG Batches (one per slice index across batch).
    """

    def __init__(
        self,
        slice_input_dim: int = 2,
        slice_emb_dim: int = 256,
        transformer_hidden_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.1,
        max_num_slices: int = 80,
        k_neighbors: int = 20,
    ):
        super().__init__()

        self.slice_encoder = EdgeConvSliceEncoder(
            input_dim=slice_input_dim,
            emb_dim=slice_emb_dim,
            k=k_neighbors,
        )

        self.temporal_encoder = TransformerSliceEncoder(
            input_dim=slice_emb_dim,
            hidden_dim=transformer_hidden_dim,
            num_layers=transformer_layers,
            nhead=transformer_heads,
            dropout=transformer_dropout,
            max_seq_len=max_num_slices,
        )

        self.head = CdRegressor(input_dim=slice_emb_dim)

    def forward(self, car_slice_batches: list[Batch]) -> torch.Tensor:
        """
        Args:
            car_slice_batches (list of PyG Batch objects):
                Length = S (num slices). Each is a PyG Batch with:
                - x:      (N_points, 2)
                - batch:  (N_points,) → batch sample index

        Returns:
            Tensor: (B,) Cd prediction per sample
        """
        device = next(self.parameters()).device
        slice_embeddings = []

        for slice_batch in car_slice_batches:
            slice_batch = slice_batch.to(device)
            emb = self.slice_encoder(slice_batch)  # (B, emb_dim)
            slice_embeddings.append(emb)

        # Stack to (S, B, emb_dim) → then transpose to (B, S, emb_dim)
        transformer_input = torch.stack(slice_embeddings, dim=0).transpose(0, 1)

        car_emb = self.temporal_encoder(transformer_input)  # (B, emb_dim)
        return self.head(car_emb)  # (B,)

    @staticmethod
    def example_input(batch_size: int = 2, S: int = 80, P: int = 6500) -> list[Batch]:
        """
        Returns dummy input: list of S PyG batches, each containing B slices of shape (P, 2)
        """
        from torch_geometric.data import Data, Batch

        slice_batches = []

        for s in range(S):
            slice_data = []
            for b in range(batch_size):
                pts = torch.randn(P, 2)
                slice_data.append(Data(x=pts))
            slice_batches.append(Batch.from_data_list(slice_data))

        return slice_batches
