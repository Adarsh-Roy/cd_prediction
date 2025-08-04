"""
Single-file inference for Cd prediction.

Example
-------
python -m src.main predict \
    --config experiments/baseline.yaml \
    --checkpoint experiments/exp_name/checkpoints/best_model.pt \
    --point-cloud data/car_0001.paddle_tensor
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data

# ── project imports ──────────────────────────────────────────────────── #
from src.config.constants import (  # constants & mapping
    DEFAULT_NUM_SLICES,
    DEFAULT_SLICE_AXIS,
    DEFAULT_TARGET_POINTS,
    SCALER_FILE,
    SUBSET_DIR,
    model_to_padded,
)
from src.data.slices import (
    PointCloudSlicer,
    pad_and_mask_slices,  #
)
from src.models.model import get_model
from src.utils.helpers import prepare_device
from src.utils.io import load_config, load_scaler
from src.utils.logger import logger


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict(
    *,
    cfg_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    point_cloud_path: Union[str, Path],
) -> float:
    """
    Run inference on a **single** point-cloud file and return the un-scaled Cd.

    Parameters
    ----------
    cfg_path
        Path to the YAML / JSON experiment config (used to recreate the model).
    checkpoint_path
        Path to the trained *.pt file.
    point_cloud_path
        Path to the *.paddle_tensor point-cloud to score.
    device
        Optional override for the compute device (e.g. "cuda:0").

    Returns
    -------
    float
        Predicted drag-coefficient **in the original scale**.
    """
    cfg = load_config(cfg_path)
    device = prepare_device(cfg.get("device"))
    model_type: str = cfg["model"]["model_type"]
    model_params = cfg["model"][model_type]

    # ------------------------------------------------------------------ #
    # Model & checkpoint                                                 #
    # ------------------------------------------------------------------ #
    model = get_model(model_type=model_type, **model_params).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint → {checkpoint_path}")

    # ------------------------------------------------------------------ #
    # Point-cloud → 2-D slices                                           #
    # ------------------------------------------------------------------ #
    num_slices = cfg["data"].get("num_slices", DEFAULT_NUM_SLICES)
    axis = cfg["data"].get("slice_axis", DEFAULT_SLICE_AXIS)

    slicer = PointCloudSlicer(
        input_dir=Path("."),  # dummy
        output_dir=Path("."),  # dummy
        num_slices=num_slices,
        axis=axis,
        max_files=None,
        split="predict",
        subset_dir=SUBSET_DIR,
    )

    slices = slicer.process_file(Path(point_cloud_path))

    # ------------------------------------------------------------------ #
    # Build model input                                                  #
    # ------------------------------------------------------------------ #
    padded: bool = model_to_padded[model_type]
    if padded:
        target_pts = cfg["data"].get("target_points", DEFAULT_TARGET_POINTS)
        slices_padded, point_mask = pad_and_mask_slices(slices, target_pts)
        slices_t = torch.from_numpy(slices_padded).unsqueeze(0).float().to(device)
        p_mask_t = torch.from_numpy(point_mask).unsqueeze(0).float().to(device)
        model_input = (slices_t, p_mask_t)
    else:
        batches: List[Batch] = []
        for sl in slices:
            data = Data(x=torch.from_numpy(sl.astype(np.float32)))
            batches.append(Batch.from_data_list([data]).to(device))
        model_input = batches

    # ------------------------------------------------------------------ #
    # Forward pass                                                       #
    # ------------------------------------------------------------------ #
    import time

    # ----------------- Measure inference time -----------------
    start = time.perf_counter()
    pred_scaled: float = float(model(model_input).squeeze().cpu())
    end = time.perf_counter()

    inference_time = end - start
    logger.info(f"Inference time: {inference_time:.6f} seconds")
    logger.info(f"Predicted (scaled) Cd = {pred_scaled:.5f}")

    # ------------------------------------------------------------------ #
    # Inverse-transform to original units                                #
    # ------------------------------------------------------------------ #
    scaler = load_scaler(SCALER_FILE)  # uses the global scaler path
    cd_unscaled = float(scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0])
    logger.info(f"Predicted (un-scaled) Cd = {cd_unscaled:.5f}")
    return cd_unscaled
