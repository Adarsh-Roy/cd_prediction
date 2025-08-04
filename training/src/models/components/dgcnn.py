import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, knn_graph, global_max_pool
from torch_geometric.data import Batch


class EdgeConvSliceEncoder(nn.Module):
    """
    DGCNN-style encoder for a batch of 2D point slices.

    Each slice is encoded into a fixed-size embedding vector using:
    - k-NN graph construction
    - EdgeConv over each slice
    - Global max pooling over points per slice
    """

    def __init__(self, input_dim: int = 2, emb_dim: int = 256, k: int = 20):
        """
        Args:
            input_dim (int): Dimension of input points (default: 2)
            emb_dim (int): Output embedding size for each slice
            k (int): Number of nearest neighbors for graph construction
        """
        super().__init__()
        self.k = k

        self.edge_conv = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, emb_dim),
            ),
            aggr="max",
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data (torch_geometric.data.Batch): A PyG batch containing:
                - x:      (N_total, input_dim) all points from all slices
                - batch:  (N_total,) index of which slice each point belongs to

        Returns:
            Tensor: (num_slices, emb_dim) — One embedding per slice
        """
        # Build the slice-wise k-NN graph (no inter-slice edges)
        edge_index = knn_graph(x=data.x, k=self.k, batch=data.batch)

        # Run EdgeConv over each point
        point_features = self.edge_conv(data.x, edge_index)

        # Global max pool to get a per-slice embedding
        slice_embedding = global_max_pool(point_features, data.batch)

        return slice_embedding
