"""
Training loop for Cd prediction.

Run from project root, e.g.:

    python -m src.main train \
        --config experiments/configuration.json \
        --resume experiments/checkpoints/best_model_val_loss=0.0123.pt

The file expects:
*   A CD dataset (`src.data.dataset.CdDataset`)
*   A model
*   Project-wide paths in `src.config.constants`
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    ModelCheckpoint,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
from ignite.metrics.regression.r2_score import R2Score
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config.constants import EXP_DIR, PREPARED_DATASET_DIR, model_to_padded
from src.data.dataset import (
    CdDataset,
    ragged_collate_fn,
)
from src.models.model import get_model
from src.utils.helpers import make_unscale, prepare_device, prepare_ragged_batch_fn
from src.utils.io import load_config
from src.utils.logger import logger


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_training(
    exp_name: str,
    cfg_path: str | Path,
    resume: Path | None = None,
    preapred_dataset_dir: Path = PREPARED_DATASET_DIR,
    fit_scaler: bool = False,
):
    """
    Main entry-point called from CLI to run training.
    The training config is loaded from `cfg_path`.
    The training is resumed if checkpoint path `resume` if provided.

    Args:
        exp_name: name of this experiment (used to create sub-directories for
        checkpoints and tb-logs)
        cfg_path: path to YAML / JSON describing the experiment.
        resume:   optional checkpoint to resume (full state: model, optimiser, trainer).
        preapred_dataset_dir: path to the directory with the prepared dataset.
    """
    cfg = load_config(cfg_path)

    # --------------------------------------------------------------------- #
    # Reproducibility & device                                              #
    # --------------------------------------------------------------------- #
    seed = cfg.get("seed", 42)
    debugging = cfg.get("debugging", False)
    batch_size = cfg["data"].get("batch_size", 4)
    torch.manual_seed(seed)
    device = prepare_device(cfg.get("device"))

    # --------------------------------------------------------------------- #
    # Data                                                                  #
    # --------------------------------------------------------------------- #
    padded: bool = model_to_padded[cfg["model"]["model_type"]]
    train_set = CdDataset(
        root_dir=preapred_dataset_dir,
        split="train",
        fit_scaler=fit_scaler,
        padded=padded,
        debugging=debugging,
    )
    val_set = CdDataset(
        root_dir=preapred_dataset_dir,
        split="val",
        fit_scaler=False,
        padded=padded,
        debugging=debugging,
    )
    scaler = train_set.scaler
    unscale_fn = make_unscale(scaler=scaler)

    # --------------------------------------------------------------------- #
    # Model, optimiser, criterion                                           #
    # --------------------------------------------------------------------- #
    model_type = cfg["model"]["model_type"]
    model = get_model(model_type=model_type, **cfg["model"][model_type]).to(device)
    optim_params = cfg.get("optim", {}).get("params", {"lr": 1e-3})
    optimizer = torch.optim.Adam(model.parameters(), **optim_params)
    criterion = nn.MSELoss()

    if padded:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            pin_memory=(device.type == "cuda"),
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            pin_memory=(device.type == "cuda"),
            shuffle=False,
            drop_last=False,
        )
        trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
        train_evaluator = create_supervised_evaluator(
            model,
            metrics={
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            },
            output_transform=unscale_fn,
            device=device,
        )
        val_evaluator = create_supervised_evaluator(
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
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            drop_last=True,
            batch_size=batch_size,
            pin_memory=(device.type == "cuda"),
            collate_fn=ragged_collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            shuffle=False,
            drop_last=False,
            batch_size=batch_size,
            pin_memory=(device.type == "cuda"),
            collate_fn=ragged_collate_fn,
        )
        trainer = create_supervised_trainer(
            model,
            optimizer,
            criterion,
            device=device,
            prepare_batch=prepare_ragged_batch_fn,
        )
        train_evaluator = create_supervised_evaluator(
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
        val_evaluator = create_supervised_evaluator(
            model,
            metrics={
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            },
            output_transform=unscale_fn,
            device=device,
            prepare_batch=prepare_ragged_batch_fn,
        )

    # --------------------------------------------------------------------- #
    # Logging and Monitoring                                                #
    # --------------------------------------------------------------------- #
    train_pbar = ProgressBar(desc="Training", persist=True)
    train_pbar.attach(trainer, output_transform=lambda loss: {"batch_loss": loss})

    train_eval_pbar = ProgressBar(desc="Training Set Evaluation", persist=True)
    train_eval_pbar.attach(train_evaluator)
    val_eval_pbar = ProgressBar(desc="Validation Set Evaluation", persist=True)
    val_eval_pbar.attach(val_evaluator)

    # Log running loss every N iterations
    log_interval = cfg["training"].get("log_interval", 100)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def _log_iter(engine):
        train_pbar.log_message(
            f"\n\nEpoch[{engine.state.epoch}] "
            f"Iter[{engine.state.iteration}] "
            f"bach_loss={engine.state.output:.4f}\n\n"  # scaled batch loss for this iteration
        )

    tb_logger = TensorboardLogger(log_dir=EXP_DIR / exp_name / "tb-logs")

    # Log the scaled loss that we are using for criterion calculation during training.
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval_and_log(engine):
        train_evaluator.run(train_loader)
        train_metrics = train_evaluator.state.metrics
        val_evaluator.run(val_loader)
        val_metrics = val_evaluator.state.metrics
        train_pbar.log_message(
            f"\n\nEpoch {engine.state.epoch}: "
            f"train MAE={train_metrics['mae']:.4f} | "
            f"train R2={train_metrics['r2']:.4f} | "
            f"val MAE={val_metrics['mae']:.4f} | "
            f"val R2={val_metrics['r2']:.4f}\n\n"
        )

        step = engine.state.epoch

        tb_logger.writer.add_scalars(
            "MAE",
            {"train": train_metrics["mae"], "validation": val_metrics["mae"]},
            step,
        )

        tb_logger.writer.add_scalars(
            "MSE",
            {"train": train_metrics["mse"], "validation": val_metrics["mse"]},
            step,
        )

        tb_logger.writer.add_scalars(
            "R2", {"train": train_metrics["r2"], "validation": val_metrics["r2"]}, step
        )

    def score_fn(eng):
        """
        Score function needed for early stopping and checkpointing.
        Should be used with Val Evaluator's engine.
        Returns the negative of mae.
        """
        return -eng.state.metrics["mae"]  # minimise MAE

    # --------------------------------------------------------------------- #
    # Early stopping                                                        #
    # --------------------------------------------------------------------- #
    early_stop_handler = EarlyStopping(
        patience=cfg["training"].get("early_stop_patience", 10),
        score_function=score_fn,
        trainer=trainer,
        min_delta=cfg["training"].get("early_stop_min_delta", 0.0),
        cumulative_delta=False,
    )
    val_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    # --------------------------------------------------------------------- #
    # Checkpointing                                                         #
    # --------------------------------------------------------------------- #

    # save the best models based on val_evaluator's mae
    # (meant for inference, not resuming training)
    best_model_saver = ModelCheckpoint(
        dirname=EXP_DIR / exp_name / "model_checkpoints",
        filename_prefix="best_",
        n_saved=3,
        global_step_transform=global_step_from_engine(trainer),
        score_function=score_fn,
        score_name="val_mae",
        require_empty=False,
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED, best_model_saver, {"model": model}
    )

    # save the entire training state after every epoch of trainer
    # (meant for resuming)
    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer}
    training_state_saver = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(
            EXP_DIR / exp_name / "training_state_checkpoints",
            create_dir=True,
            atomic=True,
            require_empty=False,
        ),
        filename_prefix="training_state_",
        global_step_transform=global_step_from_engine(trainer),
        n_saved=5,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, training_state_saver)

    # Resume full state if requested
    if resume:
        ckpt_fp = Path(resume)
        if not ckpt_fp.exists():
            logger.error(f"Resume checkpoint not found: {ckpt_fp}")
            sys.exit(1)
        logger.info(f"Resuming from checkpoint: {ckpt_fp}")
        to_load = {"model": model, "optimizer": optimizer, "trainer": trainer}
        Checkpoint.load_objects(
            to_load=to_load, checkpoint=torch.load(ckpt_fp, map_location=device)
        )
        # Override the learning-rate(s) if a value is provided in the config
        cfg_lr = cfg.get("optim", {}).get("params", {}).get("lr")
        if cfg_lr is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = param_group.get("lr", None)
                if old_lr is not None and old_lr != cfg_lr:
                    logger.info(
                        f"Param-group {i}: overriding LR {old_lr:.3e} → {cfg_lr:.3e}"
                    )
                param_group["lr"] = cfg_lr

    # --------------------------------------------------------------------- #
    # Start training                                                        #
    # --------------------------------------------------------------------- #
    with logging_redirect_tqdm():
        trainer.run(train_loader, max_epochs=cfg["training"]["epochs"])
    tb_logger.close()
