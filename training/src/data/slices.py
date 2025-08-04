"""
Loads .paddle_tensor point clouds, bins them into num_slices along axis.
Pads the slices to target number of points per slice.
"""

from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config.constants import (
    DEFAULT_SLICE_AXIS,
    DEFAULT_TARGET_POINTS,
    SUBSET_DIR,
)
from src.utils.io import load_cd_map, load_design_ids, load_point_cloud
from src.utils.logger import logger


class PointCloudSlicer:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        num_slices: int,
        axis: str,
        max_files: Optional[int],
        split: Literal["train", "val", "test", "all", "predict"],
        subset_dir: Path,
    ) -> None:
        """
        Args:
            input_dir: Path to the directory with the Point Clouds (.paddle_tensor files)
            output_dir: Path to the directory where the slices (.npy files) will be saved.
            num_slices: Number of slices to divide the point clouds into.
            axis: Axis along which to slice the point clouds. Must be one of 'x', 'y', or 'z'.
            max_files: Maximum number of files to process. If None, process all files.
            split: The data split to slice.
            subset_dir: Path to the directory with the design IDs for the split. (.txt files)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_slices = num_slices
        self.axis = axis
        self.max_files = max_files
        self.split = split
        self.subset_dir = Path(subset_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.axis_map = {"x": 0, "y": 1, "z": 2}
        if self.axis not in self.axis_map:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

        if self.split == "all":
            self.valid_ids = (
                load_design_ids("train", self.subset_dir)
                | load_design_ids("val", self.subset_dir)
                | load_design_ids("test", self.subset_dir)
            )
        elif self.split != "predict":
            self.valid_ids = load_design_ids(self.split, self.subset_dir)

    def generate_slices(self, points: np.ndarray) -> list[np.ndarray]:
        """
        Divides the point clouds along the axis `self.axis` into `self.num_slices` bins
        and returns the slices.

        Args:
            points: A numpy ndarray of shape (N,3) where N is the number of points
            in the point cloud.
        Returns:
            A list of length `self.num_slices` where each element is a 2-column array
            of shape (N_i, 2), where N_i is the number of points in the i_th slice.
        """
        axis = self.axis_map[self.axis]
        coords_along_axis = points[:, axis]
        min_val, max_val = coords_along_axis.min(), coords_along_axis.max()
        bin_edges = np.linspace(min_val, max_val, self.num_slices + 1)

        slices = []
        for i in range(self.num_slices):
            low, high = bin_edges[i], bin_edges[i + 1]
            # create a boolean mask to identify the pionts that will fall in this bin
            mask = (coords_along_axis >= low) & (coords_along_axis < high)
            # get all the points that fall in this bin
            slice_points = points[mask]

            # drop the coordinates along `self.axis` to project all the points in this bin
            # to the plane perpendicular to `self.axis`. This is one slice.
            slice = np.delete(slice_points, axis, axis=1)
            slices.append(slice)

        return slices

    def process_file(self, file_path: Path) -> list[np.ndarray]:
        """
        Slice the point cloud and return the slices.

        Args:
            file_path: Path to the .paddle_tensor file.
        Returns:
            The list of slices for this point cloud.
        """
        points = load_point_cloud(file_path)
        slices = self.generate_slices(points)
        total_points = sum(len(sl) for sl in slices)
        logger.info(
            f"{file_path.name} -> {total_points} total points across {len(slices)} slices."
        )
        return slices

    def run(self):
        """
        Slice all the point clouds in the `self.input_dir` correspondings to the
        design IDs for `self.split` and save the slices (.npy file) to `self.output_dir` for each.
        """
        all_files = sorted(self.input_dir.glob("*.paddle_tensor"))
        filtered_files = [f for f in all_files if f.stem in self.valid_ids]

        if self.max_files is not None:
            filtered_files = filtered_files[: self.max_files]

        logger.info(
            f"Processing {len(filtered_files)} point clouds from split: {self.split}"
        )

        count_success, count_error = 0, 0

        with logging_redirect_tqdm():
            for file_path in tqdm(filtered_files, desc=f"Slicing {self.split} set"):
                try:
                    design_id = file_path.stem
                    output_path = self.output_dir / f"{design_id}_axis-{self.axis}.npy"
                    slices = self.process_file(file_path)
                    np.save(
                        output_path, np.array(slices, dtype=object), allow_pickle=True
                    )
                    count_success += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    count_error += 1

        logger.info(
            f"Finished slicing split '{self.split}': {count_success} succeeded, {count_error} failed."
        )


def pad_and_mask_slices(
    slices: Sequence[np.ndarray],
    target_points=DEFAULT_TARGET_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the slices to have a fixed (`target_points`) number of points per slice.

    Args:
        slices: A sequence of slices, each a 2-column array of shape (N, 2).
        target_points: The target number of points per slice.

    Returns:
        A tuple of two elements:
        - padded: float32 ndarray of shape (num_slices, target_points, 2)
        - point_mask: float32 ndarray of shape (num_slices, target_points).
    """
    num_slices = len(slices)
    padded = np.zeros((num_slices, target_points, 2), dtype=np.float32)
    point_mask = np.zeros((num_slices, target_points), dtype=np.float32)

    for i, sl in enumerate(slices):
        # expected shape of a slice: (N_i, 2)
        if sl.shape[0] == 0:
            continue
        if target_points < sl.shape[0]:
            raise ValueError(
                f"Number of points in a slice is greater than {target_points}"
            )
        n_pts = sl.shape[0]
        padded[i, :n_pts] = sl[:n_pts]
        point_mask[i, :n_pts] = 1

    return padded, point_mask


def prepare_dataset(
    slice_dir: Path,
    output_dir: Path,
    split: Literal["train", "val", "test", "all"],
    target_points: Optional[int] = None,
    subset_dir=SUBSET_DIR,
) -> None:
    """
    Prepare the dataset (.npz files) and save them to output_dir or
    output_dir/padded if target_points is provided.
    The .npz files saved have the Cd zipped together with the slices
    (and optionally point mask if `target_points` is provided).

    Args:
        slice_dir: Path to the directory with the slices (.npy files)
        output_dir: Path to the directory where the prepared dataset (.npz files) will be saved.
        split: The data split to prepare.
        target_points: The target number of points per slice if padding is required.
        subset_dir: Path to the directory with the design IDs for the split. (.txt files)

    """
    slice_dir = Path(slice_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if split == "all":
        design_ids = (
            load_design_ids("train", subset_dir)
            | load_design_ids("val", subset_dir)
            | load_design_ids("test", subset_dir)
        )
    else:
        design_ids = load_design_ids(split, subset_dir)
    cd_map = load_cd_map()
    logger.info(f"Preparing split: {split} -> {len(design_ids)} design IDs")

    all_files = [f for f in slice_dir.glob("*.npy")]
    matched = [f for f in all_files if f.stem.split("_axis")[0] in design_ids]
    logger.info(f"Matched {len(matched)} files in {slice_dir} for split '{split}'")

    with logging_redirect_tqdm():
        for fname in tqdm(
            matched,
            desc=f"Prepare Dataset for split {split}, padded={True if target_points else False}",
        ):
            in_path = fname
            try:
                slices = np.load(in_path, allow_pickle=True)

                processed_slices, point_mask = None, None

                if target_points:
                    pad_dir = output_dir / "padded"
                    pad_dir.mkdir(parents=True, exist_ok=True)

                    # pad the slices if `target_points` is provided.
                    # This is needed for backwards compatibility with LSTM model.
                    processed_slices, point_mask = pad_and_mask_slices(
                        slices, target_points
                    )
                    save_dir = pad_dir
                else:
                    processed_slices = slices
                    save_dir = output_dir

                design_id = fname.stem
                design_id = design_id.split("_axis")[0]
                cd_val = cd_map.get(design_id)
                if cd_val is None:
                    logger.warning(f"Cd not found for {design_id} – file skipped")
                    continue

                out_path = save_dir / f"{design_id}.npz"
                if target_points:
                    np.savez_compressed(
                        out_path,
                        slices=processed_slices,
                        point_mask=point_mask,
                        Cd=cd_val,
                    )
                else:
                    np.savez_compressed(
                        out_path,
                        slices=processed_slices,
                        Cd=cd_val,
                    )
            except Exception as e:
                logger.warning(
                    f"Preparing dataset for design_id: {fname.name} failed: {e}"
                )

    logger.info(f"Done: Saved {len(matched)} data points to {output_dir}")


def display_slices(
    slices,
    design_id=None,
    n_cols=5,
    limit=None,
    figsize=(15, 3),
    axis=DEFAULT_SLICE_AXIS,
    save_path=None,
):
    """
    Display or save 2D slices from a point cloud.

    slices: list of (num_of_points, 2) np arrays
    """
    if limit:
        slices = slices[:limit]

    all_points = np.vstack([sl for sl in slices if sl.size])
    xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
    ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()

    pad_x = 0.02 * (xmax - xmin)
    pad_y = 0.02 * (ymax - ymin)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    n = len(slices)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    for idx, sl in enumerate(slices):
        ax = axes.flat[idx]
        if sl.size:
            ax.scatter(sl[:, 0], sl[:, 1], s=2, c="k")
        else:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(f"{axis} ∈ slice {idx}", fontsize=8)

    for j in range(n, n_rows * n_cols):
        axes.flat[j].axis("off")

    fig.suptitle(
        f"Slices for {design_id}" if design_id else "Point Cloud Slices", fontsize=14
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[✓] Saved to {save_path}")
        plt.close()
    else:
        plt.show()
