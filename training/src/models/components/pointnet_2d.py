import torch
import torch.nn as nn


class PointNet2D(nn.Module):
    """
    PointNet2D with attention pooling instead of max pooling.
    Input shape: (batch_size, num_points, 2)
    Mask shape: (batch_size, num_points) with 1s for valid points and 0s for padded.
    Output shape: (batch_size, emb_dim)
    """

    def __init__(self, input_dim: int = 2, emb_dim: int = 256):
        super(PointNet2D, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 32, 1),  # 2 → 32
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 1),  # 32 → 64
            nn.LeakyReLU(),
            nn.Conv1d(64, emb_dim, 1),  # 64 → 256
            nn.LeakyReLU(),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_points, 2) - Input points
            mask: (batch_size, num_points) or None - Mask of valid points (1=real, 0=padded)

        Returns:
            (batch_size, emb_dim) slice embedding
        """
        x = x.transpose(1, 2)  # (batch_size, 2, num_points)
        features = self.mlp(x)  # (batch_size, emb_dim, num_points)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, num_points)
            features = features.masked_fill(mask == 0, float("-inf"))

        embedding, _ = torch.max(features, dim=2)  # (batch_size, emb_dim)
        return embedding
