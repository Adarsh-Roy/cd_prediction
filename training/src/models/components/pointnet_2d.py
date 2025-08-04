import torch
import torch.nn as nn


class PointNet2D(nn.Module):
    """
    PointNet2D with attention pooling instead of max pooling.
    Input shape: (B, N, 2)
    Mask shape: (B, N) with 1s for valid points and 0s for padded.
    Output shape: (B, emb_dim)
    """

    def __init__(self, input_dim: int = 2, emb_dim: int = 256):
        super(PointNet2D, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(),
            nn.Conv1d(128, emb_dim, 1),
            nn.LeakyReLU(),
        )

        self.attn = nn.Conv1d(emb_dim, 1, 1)  # Attention layer: (B, 1, N)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, 2)         - Input points
            mask: (B, N) or None - Mask of valid points (1=real, 0=padded)

        Returns:
            (B, emb_dim) slice embedding
        """
        x = x.transpose(1, 2)  # (B, 2, N)
        features = self.mlp(x)  # (B, emb_dim, N)
        attn_logits = self.attn(features)  # (B, 1, N)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, N)
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=2)  # (B, 1, N)
        embedding = torch.sum(features * attn_weights, dim=2)  # (B, emb_dim)

        return embedding
