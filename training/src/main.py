"""
Main command-line entry-point for the Cd-prediction project.

Run from the project root, e.g.

    python -m src.main train \
        --config experiments/exp_name/config.json \
        --resume experiments/exp_name/checkpoints/best_model_val_mae=-0.012.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.config.constants import (
    DEFAULT_NUM_SLICES,
    DEFAULT_SLICE_AXIS,
    DEFAULT_TARGET_POINTS,
    POINT_CLOUDS_DIR,
    PREPARED_DATASET_DIR,
    PREPARED_DATASET_PADDED_DIR,
    SLICE_DIR,
    SUBSET_DIR,
    model_to_padded,
)
from src.data.dataset import CdDataset
from src.data.slices import PointCloudSlicer, display_slices, prepare_dataset
from src.evaluation.evaluate import run_evaluation
from src.inference.predict import predict
from src.training.ignite_loops import run_training
from src.utils.io import load_config
from src.utils.logger import logger


def build_parser() -> argparse.ArgumentParser:
    """Configure argparse sub-commands."""
    parser = argparse.ArgumentParser(
        description="Cd-prediction: slicing · training · evaluation · inference"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --------------------------------------------------------------------- #
    # slice                                                                 #
    # --------------------------------------------------------------------- #
    slice_p = subparsers.add_parser("slice", help="Slice 3-D point clouds")
    slice_p.add_argument("--input-dir", type=Path, default=POINT_CLOUDS_DIR)
    slice_p.add_argument("--output-dir", type=Path, default=SLICE_DIR)
    slice_p.add_argument("--num-slices", type=int, default=DEFAULT_NUM_SLICES)
    slice_p.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_SLICE_AXIS)
    slice_p.add_argument("--max-files", type=int)
    slice_p.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all"
    )
    slice_p.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)

    # --------------------------------------------------------------------- #
    # visualize                                                             #
    # --------------------------------------------------------------------- #
    viz_p = subparsers.add_parser("visualize", help="Show saved 2-D slices")
    viz_p.add_argument(
        "-i", "--input", type=Path, required=True, help="Path to .npy slice file"
    )
    viz_p.add_argument("--cols", type=int, default=5)
    viz_p.add_argument("--limit", type=int)
    viz_p.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_SLICE_AXIS)
    viz_p.add_argument("--save-path")

    # --------------------------------------------------------------------- #
    # prep                                                                  #
    # --------------------------------------------------------------------- #
    prep_p = subparsers.add_parser("prep", help="Prepare Dataset")
    prep_p.add_argument(
        "-s", "--split", choices=["train", "val", "test", "all"], default="all"
    )
    prep_p.add_argument("--slice-dir", type=Path, default=SLICE_DIR)
    prep_p.add_argument("--output-dir", type=Path, default=PREPARED_DATASET_DIR)
    prep_p.add_argument(
        "--pad",
        action="store_true",
        help="If set, the slices are padded and masked. Use the flag --target-points to specify the number of points per slice.",
    )
    prep_p.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS)
    prep_p.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)

    # --------------------------------------------------------------------- #
    # fit scaler                                                            #
    # --------------------------------------------------------------------- #
    fit_scaler_p = subparsers.add_parser("fit_scaler", help="Fit Scaler")
    fit_scaler_p.add_argument("--prepared-dataset-dir", type=Path)
    fit_scaler_p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML / JSON experiment config",
    )
    fit_scaler_p.add_argument(
        "--padded",
        action="store_true",
        help="Only used when --prepared-dataset-dir is not provided."
        "If set, use the default padded dataset directory, otherwise, use the default non-padded dataset",
    )

    # --------------------------------------------------------------------- #
    # train                                                                 #
    # --------------------------------------------------------------------- #
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML / JSON experiment config",
    )
    train_p.add_argument(
        "--resume",
        type=Path,
        help="Full path to checkpoint to resume from.  "
        "If omitted, you will be prompted interactively.",
    )
    train_p.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Name for this experiment run (e.g. lr1e-3_bs32).",
    )
    train_p.add_argument("--prepared-dataset-dir", type=Path)
    train_p.add_argument(
        "--fit-scaler",
        action="store_true",
        help="If set, fit a scaler to the training data. Otherwise, use the already"
        "available scaler.",
    )

    # --------------------------------------------------------------------- #
    # Evaluate                                                              #
    # --------------------------------------------------------------------- #
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_p.add_argument(
        "--config", type=Path, required=True, help="Experiment config YAML / JSON"
    )
    eval_p.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model *.pt file"
    )
    eval_p.add_argument("--split", choices=["val", "test"], default="test")
    eval_p.add_argument(
        "--prepared-dataset-dir", type=Path, default=PREPARED_DATASET_DIR
    )

    # --------------------------------------------------------------------- #
    # Predict                                                               #
    # --------------------------------------------------------------------- #
    # --------------------------------------------------------------------- #
    # Predict                                                               #
    # --------------------------------------------------------------------- #
    pred_p = subparsers.add_parser("predict", help="Run single-file inference")
    pred_p.add_argument(
        "--config", type=Path, required=True, help="Experiment config YAML / JSON"
    )
    pred_p.add_argument(
        "--checkpoint", type=Path, required=True, help="Model *.pt checkpoint"
    )
    pred_p.add_argument(
        "--point-cloud",
        type=Path,
        required=True,
        help="Path to the *.paddle_tensor point-cloud file",
    )

    return parser


# --------------------------------------------------------------------------- #
# Entrypoint dispatcher                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------ slice --------------------------------- #
    if args.command == "slice":
        slicer = PointCloudSlicer(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_slices=args.num_slices,
            axis=args.axis,
            max_files=args.max_files,
            split=args.split,
            subset_dir=args.subset_dir,
        )
        slicer.run()

    # ---------------------------- visualize ------------------------------- #
    elif args.command == "visualize":
        if not args.input.is_file():
            parser.error(f"File not found: {args.input}")
        slices = np.load(args.input, allow_pickle=True)
        car_id = args.input.stem
        display_slices(
            slices,
            design_id=car_id,
            n_cols=args.cols,
            limit=args.limit,
            axis=args.axis,
            save_path=args.save_path,
        )

    # ------------------------------- pad ---------------------------------- #
    elif args.command == "prep":
        if args.pad:
            target_points = args.target_points
        else:
            target_points = None

        prepare_dataset(
            slice_dir=args.slice_dir,
            output_dir=args.output_dir,
            split=args.split,
            target_points=target_points,
            subset_dir=args.subset_dir,
        )

    elif args.command == "fit_scaler":
        dataset_path: Path | None = args.prepared_dataset_dir
        cfg = load_config(args.config)
        padded: bool = cfg["data"]["padded"]
        debugging: bool = cfg["debugging"]
        if dataset_path is None:
            if padded:
                dataset_path = PREPARED_DATASET_PADDED_DIR
            else:
                dataset_path = PREPARED_DATASET_DIR
        _ = CdDataset(
            root_dir=dataset_path,
            split="train",
            padded=padded,
            debugging=debugging,
            fit_scaler=True,
        )

    # ------------------------------ train --------------------------------- #
    elif args.command == "train":
        resume: Path | None = args.resume
        if resume:
            if not Path(resume).is_file():
                logger.error(f"Checkpoint not found: {resume}")
                sys.exit(1)
        dataset_path: Path | None = args.prepared_dataset_dir
        if dataset_path is None:
            cfg = load_config(args.config)
            padded: bool = model_to_padded[cfg["model"]["model_type"]]
            if padded:
                dataset_path = PREPARED_DATASET_PADDED_DIR
            else:
                dataset_path = PREPARED_DATASET_DIR

        run_training(
            exp_name=args.exp_name,
            cfg_path=args.config,
            resume=resume,
            preapred_dataset_dir=dataset_path,
            fit_scaler=args.fit_scaler,
        )

    # ---------------------------- evaluate -------------------------------- #
    elif args.command == "evaluate":
        dataset_path: Path | None = args.prepared_dataset_dir
        if dataset_path is None:
            cfg = load_config(args.config)
            padded: bool = model_to_padded[cfg["model"]["model_type"]]
            if padded:
                dataset_path = PREPARED_DATASET_PADDED_DIR
            else:
                dataset_path = PREPARED_DATASET_DIR
        run_evaluation(
            cfg_path=args.config,
            checkpoint_path=args.checkpoint,
            split=args.split,
            preapred_dataset_dir=dataset_path,
        )

    # ----------------------------- predict -------------------------------- #
    elif args.command == "predict":
        cd_val = predict(
            cfg_path=args.config,
            checkpoint_path=args.checkpoint,
            point_cloud_path=args.point_cloud,
        )
        print(f"Predicted Cd  →  {cd_val:.5f}")


if __name__ == "__main__":
    main()
