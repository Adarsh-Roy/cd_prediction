"""model_DLM.py
================
Composite model that combines:

* **DGCNN slice encoder** – encodes every 2‑D slice (variable #points) into an embedding.
* **LSTM temporal encoder** – processes the ordered slice embeddings of a car design.
* **MLP regression head** – maps global design embedding to scalar drag‑coefficient (C_d).

Assumptions
-----------
* All designs share the **same, fixed number of slices** (`num_slices`).
* Each slice can contain a **variable** number of points (handled by PyG + DGCNN).
* DataLoader supplies a *list* of length `num_slices`, where each element is a
  :class:`torch_geometric.data.Batch` that stacks that particular slice across the
  mini‑batch.  See ``src.data.dataset.ragged_collate_fn`` for the reference collate.

Example
-------
>>> model = Cd_DLM_Model()
>>> slice_batches, cd_target = next(iter(loader))  # ragged_collate_fn output
>>> cd_hat = model(slice_batches)
>>> loss = criterion(cd_hat, cd_target)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.config.constants import DEFAULT_NUM_SLICES
from src.models.components.dgcnn import EdgeConvSliceEncoder
from src.models.components.lstm_encoder import LSTMEncoder
from src.models.components.mlp_head import CdRegressor


class Cd_DLM_Model(nn.Module):
    """DGCNN + LSTM + MLP end‑to‑end C_d regressor."""

    def __init__(
        self,
        *,
        slice_input_dim: int = 2,
        slice_emb_dim: int = 256,
        k_neighbors: int = 16,
        num_slices: int = DEFAULT_NUM_SLICES,
        lstm_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # 1️⃣ Slice‑level encoder (variable points ➜ fixed embedding)
        self.slice_encoder = EdgeConvSliceEncoder(
            input_dim=slice_input_dim,
            emb_dim=slice_emb_dim,
            k=k_neighbors,
        )

        # 2️⃣ Temporal encoder over ordered slice sequence
        self.temporal_encoder = LSTMEncoder(
            input_dim=slice_emb_dim,
            hidden_dim=lstm_hidden_dim,
        )

        # 3️⃣ Regression head – maps design embedding ➜ scalar Cd
        self.head = CdRegressor(
            input_dim=lstm_hidden_dim * 2  # LSTMEncoder is bidirectional
        )

        self.num_slices = num_slices

    # --------------------------------------------------------------------- #
    #  Forward                                                              #
    # --------------------------------------------------------------------- #
    def forward(self, slice_batches: List[Batch]) -> torch.Tensor:
        """Predict C_d for a mini‑batch.

        Parameters
        ----------
        slice_batches
            List of length *S = ``self.num_slices``*.  Each element is a
            :class:`torch_geometric.data.Batch` whose graphs correspond to the
            same slice index across the mini‑batch designs.

        Returns
        -------
        torch.Tensor
            Shape *(B,)* – predicted drag coefficient for each design.
        """

        if len(slice_batches) != self.num_slices:
            raise ValueError(
                f"Expected {self.num_slices} slice Batches but got {len(slice_batches)}."
            )

        device = next(self.parameters()).device

        # Encode every slice independently --------------------------------- #
        slice_embeddings = []
        for sb in slice_batches:
            emb = self.slice_encoder(sb.to(device))  # (B, slice_emb_dim)
            slice_embeddings.append(emb)

        # Stack → (B, S, F) ------------------------------------------------- #
        slice_seq = torch.stack(slice_embeddings, dim=1)  # (B, S, F)

        # Temporal encoding ------------------------------------------------- #
        global_emb = self.temporal_encoder(slice_seq)  # (B, 2*hidden_dim)

        # Regression -------------------------------------------------------- #
        cd_pred = self.head(global_emb)  # (B,)

        return cd_pred

    @staticmethod
    def example_input(batch_size: int = 2, S: int = 80, P: int = 6500) -> list[Batch]:
        """
        Returns dummy input: list of S PyG batches, each containing B slices of shape (P, 2)
        """
        from torch_geometric.data import Batch, Data

        slice_batches = []

        for s in range(S):
            slice_data = []
            for b in range(batch_size):
                pts = torch.randn(P, 2)
                slice_data.append(Data(x=pts))
            slice_batches.append(Batch.from_data_list(slice_data))

        return slice_batches
