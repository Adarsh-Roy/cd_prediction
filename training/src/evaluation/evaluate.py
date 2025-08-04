"""
Run offline evaluation of a saved Cd-Regressor checkpoint.

Example
-------
python -m src.main evaluate \
    --config experiments/baseline.yaml \
    --checkpoint experiments/exp_name/checkpoints/best_model_val_mae=0.0123.pt \
    --split test
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from ignite.engine import create_supervised_evaluator
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
from ignite.metrics.regression.r2_score import R2Score
from torch.utils.data import DataLoader

from src.config.constants import PREPARED_DATASET_DIR, model_to_padded
from src.data.dataset import CdDataset, ragged_collate_fn
from src.models.model import get_model
from src.utils.helpers import make_unscale, prepare_device, prepare_ragged_batch_fn
from src.utils.io import load_config
from src.utils.logger import logger


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_evaluation(
    cfg_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
    preapred_dataset_dir: Path = PREPARED_DATASET_DIR,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint on a dataset split.

    Args:
        cfg_path:        YAML / JSON used during training (to recreate model).
        checkpoint_path: Path to best model (.pt file).
        split:           Dataset split – "val" or "test".
        batch_size:      Optional override.

    Returns:
        Dict with metric names → values.
    """
    cfg = load_config(cfg_path)
    device = prepare_device(cfg.get("device"))
    debugging = cfg.get("debugging", False)
    padded: bool = model_to_padded[cfg["model"]["model_type"]]
    batch_size = cfg["data"].get("batch_size", 4)

    # --------------------------------------------------------------------- #
    # Model, optimiser, criterion                                           #
    # --------------------------------------------------------------------- #
    model_type = cfg["model"]["model_type"]
    model = get_model(model_type=model_type, **cfg["model"][model_type]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    data_set = CdDataset(
        root_dir=preapred_dataset_dir,
        split=split,
        fit_scaler=False,
        padded=padded,
        debugging=debugging,
    )
    scaler = data_set.scaler
    unscale_fn = make_unscale(scaler=scaler)

    if padded:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            pin_memory=(device.type == "cuda"),
            shuffle=False,
            drop_last=False,
        )
        evaluator = create_supervised_evaluator(
            model,
            metrics={
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            },
            output_transform=unscale_fn,
            device=device,
        )
    else:
        data_loader = DataLoader(
            data_set,
            shuffle=True,
            drop_last=True,
            batch_size=batch_size,
            pin_memory=(device.type == "cuda"),
            collate_fn=ragged_collate_fn,
        )
        evaluator = create_supervised_evaluator(
            model,
            metrics={
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            },
            output_transform=unscale_fn,
            prepare_batch=prepare_ragged_batch_fn,
            device=device,
        )

    eval_pbar = ProgressBar(desc=f"Evaluating ({split})", persist=True)
    eval_pbar.attach(evaluator)

    # run once
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics
    metrics["rmse"] = metrics["mse"] ** 0.5
    logger.info(
        f"Split={split} | "
        f"MAE={metrics['mae']:.4f} MSE={metrics['mse']:.4f} "
        f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
    )

    return metrics
