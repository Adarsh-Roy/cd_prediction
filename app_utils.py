import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import paddle
from pypcd import pypcd
from typing import List

# --- Model and Helper Functions (copied from backend) ---

def load_point_cloud(file_path: Path) -> np.ndarray:
    """
    Load either a Paddle-saved tensor (.paddle_tensor) or a .pcd file.

    Returns
    -------
    np.ndarray  shape (N, 3), dtype float32
    """
    ext = file_path.suffix.lower()

    if ext == ".paddle_tensor":
        return paddle.load(str(file_path)).numpy().astype(np.float32)

    if ext == ".pcd":
        pc = pypcd.PointCloud.from_path(str(file_path))
        return np.stack(
            [pc.pc_data["x"], pc.pc_data["y"], pc.pc_data["z"]], axis=-1
        ).astype(np.float32)

    raise ValueError(f"Unsupported file type: {ext}")


def generate_slices(
    points: np.ndarray, num_slices: int = 80, axis: str = "x"
) -> List[np.ndarray]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    coord = points[:, axis_idx]
    bins = np.linspace(coord.min(), coord.max(), num_slices + 1)

    slices = []
    for i in range(num_slices):
        mask = (coord >= bins[i]) & (coord < bins[i + 1])
        slice_2d = np.delete(points[mask], axis_idx, axis=1)  # drop slicing axis
        slices.append(slice_2d.astype(np.float32))
    return slices


def pad_and_mask_slices(
    slices: List[np.ndarray], target_points: int = 6500
) -> tuple[np.ndarray, np.ndarray]:
    S = len(slices)
    padded = np.zeros((S, target_points, 2), dtype=np.float32)
    mask = np.zeros((S, target_points), dtype=np.float32)

    for i, sl in enumerate(slices):
        n = min(len(sl), target_points)
        padded[i, :n] = sl[:n]
        mask[i, :n] = 1
    return padded, mask


def load_scaler(path: Path) -> StandardScaler:
    if not path.is_file():
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)


class PointNet2D(nn.Module):
    def __init__(self, input_dim: int = 2, emb_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(),
            nn.Conv1d(128, emb_dim, 1),
            nn.LeakyReLU(),
        )
        self.attn = nn.Conv1d(emb_dim, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, 2, N)
        feats = self.mlp(x)  # (B, F, N)
        logits = self.attn(feats)  # (B, 1, N)
        logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(feats * weights, dim=-1)  # (B, F)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 1, batch_first=True, bidirectional=True
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(seq)
        return torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2·hidden)


class CdRegressor(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class Cd_PLM_Model(nn.Module):
    def __init__(
        self,
        slice_input_dim: int = 2,
        slice_emb_dim: int = 256,
        lstm_hidden_dim: int = 256,
        design_emb_dim: int = 512,
    ):
        super().__init__()
        self.slice_encoder = PointNet2D(slice_input_dim, slice_emb_dim)
        self.temporal_encoder = LSTMEncoder(slice_emb_dim, lstm_hidden_dim)
        self.head = CdRegressor(design_emb_dim)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        slices, mask = x  # (B,S,P,2), (B,S,P)
        B, S, P, _ = slices.shape
        flat = slices.view(B * S, P, 2)
        mflat = mask.view(B * S, P)
        emb = self.slice_encoder(flat, mflat)  # (B*S, F)
        emb = emb.view(B, S, -1)
        design = self.temporal_encoder(emb)  # (B, 2·hidden)
        return self.head(design)  # (B,)


@torch.inference_mode()
def predict_cd(
    point_cloud_path: Path,
    model: Cd_PLM_Model,
    scaler: StandardScaler,
    device: torch.device,
    num_slices: int = 80,
    axis: str = "x",
    target_points: int = 6500,
) -> float:

    pts = load_point_cloud(point_cloud_path)
    slices = generate_slices(pts, num_slices, axis)
    padded, mask = pad_and_mask_slices(slices, target_points)

    x = torch.from_numpy(padded).unsqueeze(0).to(device)  # (1,S,P,2)
    m = torch.from_numpy(mask).unsqueeze(0).to(device)  # (1,S,P)

    pred_scaled = model((x, m)).item()
    cd = scaler.inverse_transform([[pred_scaled]])[0, 0]
    return float(cd)
