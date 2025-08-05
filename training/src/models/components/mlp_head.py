import torch
import torch.nn as nn


class CdRegressor(nn.Module):
    """
    Regression MLP that maps a global embedding vector to a scalar Cd value.
    """

    def __init__(self, input_dim: int = 512):
        """
        Args:
            input_dim: Dimension of the input embedding
        Output:
            Tensor of shape (batch_size,) – scalar Cd values per input
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) – global embedding vector from encoder

        Returns:
            Cd prediction: (batch_size,) – one scalar per sample
        """
        return self.net(x).squeeze(1)
