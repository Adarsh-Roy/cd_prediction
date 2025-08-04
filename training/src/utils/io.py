"""
Utility method for input/output from the disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from src.config.constants import DRAG_CSV, SCALER_FILE, SUBSET_DIR
from src.utils.logger import logger

if TYPE_CHECKING:
    import numpy as np
    from sklearn.preprocessing import StandardScaler


def load_point_cloud(file_path: Path) -> np.ndarray:
    """
    Args:
        file_path: Path to the .paddle_tensor file.
    Returns:
        A numpy array of shape (N, 3) where N is the number of points in the point cloud.
    """
    import paddle
    import numpy as np

    obj = paddle.load(str(file_path))
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, 'numpy'):
        return obj.numpy()
    # If it's a paddle tensor but doesn't have .numpy(), convert it manually
    return np.array(obj)


def load_design_ids(split: str, subset_dir: Path = SUBSET_DIR) -> set[str]:
    """
    Load the design IDs for a given split from the subset directory.

    Args:
        split: The data split to load the design IDs for.
        subset_dir: Path to the directory with the design IDs for the split. (.txt files)
    Returns:
        A set of design ID strings.
    """
    split_file = split_file = Path(subset_dir) / f"{split}_design_ids.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        ids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(ids)} design IDs for split: {split}")
    return ids


def save_scaler(scaler, path: Path = SCALER_FILE) -> Path:
    """
    Persist a fitted sklearn-style scaler (e.g. StandardScaler) to disk.

    Args:
        scaler: any object with sklearn's fit/transform attributes (has mean_, scale_, etc.).
        path: path to the scaler file

    Returns:
        The full path where the scaler was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler to {path}")
    return path


def load_scaler(path: Path = SCALER_FILE) -> StandardScaler:
    """
    Load a previously saved scaler from disk.

    Args:
        path: path to the scaler file

    Returns:
        The deserialized scaler object (e.g. a StandardScaler with mean_/scale_ populated).

    Raises:
        FileNotFoundError if the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No scaler found at: {path}")
    scaler = joblib.load(path)
    logger.info(f"Loaded scaler from {path}")
    return scaler


def load_cd_map(csv_path: Path = DRAG_CSV) -> dict[str, float]:
    """
    Load the drag-coefficient CSV and return a dict that maps design IDs to C_d.

    Args:
        csv_path: Path to the CSV file.
    Returns:
        A dict of the form `{design_id: Cd}`.
    """
    import pandas as pd

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, usecols=["Design", "Average Cd"])
    cd_map = dict(zip(df["Design"], df["Average Cd"]))
    logger.info(f"Loaded Cd table with {len(cd_map)} entries from {csv_path}")
    return cd_map


def load_config(cfg_path: str | Path) -> dict:
    """Load a YAML or JSON experiment-config file."""
    import yaml
    import json

    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if cfg_path.suffix in {".yml", ".yaml"}:
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
    elif cfg_path.suffix == ".json":
        with cfg_path.open() as f:
            cfg = json.load(f)
    else:
        raise ValueError(f"Unsupported config type: {cfg_path.suffix}")
    logger.info(f"Loaded config from {cfg_path}")
    return cfg
