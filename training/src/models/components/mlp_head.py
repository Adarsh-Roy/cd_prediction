import torch
import torch.nn as nn


class CdRegressor(nn.Module):
    """
    Regression MLP that maps a global embedding vector to a scalar Cd value.
    """

    def __init__(self, input_dim: int = 256):
        """
        Args:
            input_dim: Dimension of the input embedding
        Output:
            Tensor of shape (B,) – scalar Cd values per input
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) – global embedding vector from encoder

        Returns:
            Cd prediction: (B,) – one scalar per sample
        """
        return self.net(x).squeeze(1)
