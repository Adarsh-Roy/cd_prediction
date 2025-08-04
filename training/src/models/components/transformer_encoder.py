import math
import torch
import torch.nn as nn


def generate_sinusoidal_position_embeddings(
    max_seq_len: int, embedding_dim: int
) -> torch.Tensor:
    """
    Creates sinusoidal positional encodings of shape (1, max_seq_len, embedding_dim),
    compatible with Transformer input.
    """
    pe = torch.zeros(max_seq_len, embedding_dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: (1, max_seq_len, embedding_dim)


class TransformerSliceEncoder(nn.Module):
    """
    Transformer encoder with sinusoidal positional encoding and attention pooling.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 80,
    ):
        """
        Args:
            input_dim: Dimension of input features per slice.
            hidden_dim: Internal feedforward dimension.
            num_layers: Number of transformer encoder layers.
            nhead: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum number of slices expected (default 80).
        """
        super().__init__()

        # Fixed sinusoidal positional embeddings
        self.register_buffer(
            "pos_encoder",
            generate_sinusoidal_position_embeddings(max_seq_len, input_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, slice_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) - Input sequence of slice embeddings.
            slice_mask: (B, S) - 1 for real slices, 0 for padding. Optional.

        Returns:
            (B, D) - Global embedding per sequence.
        """
        B, S, D = x.shape
        x = x + self.pos_encoder[:, :S, :]

        # Build src_key_padding_mask if slice_mask is provided
        if slice_mask is not None:
            src_key_padding_mask = slice_mask == 0  # True for pad
        else:
            src_key_padding_mask = None

        out = self.transformer(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, S, D)

        # Attention pooling
        scores = self.attn_pool(out).squeeze(-1)  # (B, S)
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, S)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * out, dim=1)  # (B, D)

        return self.dropout(pooled)
