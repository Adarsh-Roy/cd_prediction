import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM over the (padded) slice sequence."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        """
        Args:
            input_dim: Dimension of input features per slice.
            hidden_dim: Internal feedforward dimension.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x: (batch_size, num_slices, `self.input_dim`)

        Returns
            (batch_size, 2·hidden_dim) — concatenated [fwd, bwd] last hidden states.
        """
        _, (h_n, _) = self.lstm(x)  # h_n: (2, batch_size, hidden_dim)
        h_fwd, h_bwd = h_n[-2], h_n[-1]  # (batch_size, hidden_dim) each
        return torch.cat([h_fwd, h_bwd], dim=-1)
