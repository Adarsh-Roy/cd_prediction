"""
CdDataset
~~~~~~~~~
Loads padded & masked 2-D slices from *.npz files, (optionally) normalises
the C_d StandardScaler, and returns tensors ready for
PyTorch-Ignite loops.

Normalisation rules
-------------------
* Fit the scaler **once** on the *train* split (all points where point_mask==1).
* Persist it to `constants.SCALER_FILE`.
* All other splits **load** the persisted scaler – never refit.
* Cd (target) is never scaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.config.constants import SCALER_FILE
from src.utils.io import load_design_ids, load_scaler, save_scaler
from src.utils.logger import logger


def ragged_collate_fn(batch: list[tuple[np.ndarray, torch.Tensor]]):
    """
    Collate function for *non-padded* CdDataset samples.

    Each item in `batch` is:
        x = (slices_obj_array,)   # shape (num_slices,), slices_obj_array[i] -> (N_i, 2)
        y = Cd  (0-D tensor)

    Returns
    -------
    batched_slices : List[Batch]
        Length S list.  batched_slices[j] is a PyG Batch that contains
        all designs’ j-th slice stacked together (variable N_i handled by PyG).
        Each Data has attribute .x  →  (N_i, 2) float tensor.
    cd_values : torch.FloatTensor
        Shape (batch_size,)  target Cd for each car in the mini-batch.
    """
    # ----------------── unzip ----------------── #
    design_slices_list, cd_list = zip(*batch)
    # design_slices_list is a tuple of length batch_size. Each element in this tuple
    # is an object array of length num_slices.
    # cd_list is a tuple of length batch_size

    # ---------------- zip across slice index ---------------- #
    slices_by_index = zip(*design_slices_list)
    # generator of length num_slices
    # yields the list of slices at index i for all designs in the batch.

    batched_slices: list[Batch] = []
    for slice_list in slices_by_index:
        # len(slice_list) = batch_size.
        # Each element is (N_i, 2) ndarray
        data_list = [
            Data(x=torch.from_numpy(pts.astype(np.float32)))
            for pts in slice_list  # pts is a (N_i, 2) array
        ]
        batched_slices.append(Batch.from_data_list(data_list))

    cd_values = torch.stack(cd_list)  # shape -> (batch_size,)

    return batched_slices, cd_values


class CdDataset(Dataset):
    """
    Dataset yielding
    - if padded is True: (
                            x = (
                                    slices: shape -> (num_slices, target_points, 2),
                                    point_mask: shape -> (num_slices, target_points),
                                ),
                            y = Cd
                        )
    - if padded is False: (
                            x = slices: object array of
                                        shape -> (num_slices,), where each element
                                                 of the num_slices length array
                                                 is a 2D array of shape (N_i, 2)
                                                 where N_i is number of points
                                                 in the i_th slice.
                            y = Cd
                        )
    """

    def __init__(
        self,
        root_dir: Path,
        split: str,
        fit_scaler: bool = False,
        padded: bool = False,
        debugging: bool = False,
    ) -> None:
        """
        Initialize the dataset builder.

        Args:
            root_dir:   directory containing the *.npz files.
            split:      the data split to build dataset for.
            fit_scaler: if True, fit a new StandardScaler and persist it,
                        otherwise load the persisted scaler.
                        We fit on the training split, and use that for validation
                        and testing split.
            padded:     if True, the .npz files have point_mask and padded slices.
                        Otherwise the .npz files have raw slices and Cd only.
        """
        self.padded = padded
        self.root_dir = Path(root_dir)
        design_ids = load_design_ids(split)
        all_npz_file_paths = self.root_dir.glob("*.npz")
        self.file_paths = sorted(f for f in all_npz_file_paths if f.stem in design_ids)
        if debugging:
            self.file_paths = self.file_paths[:200]
        if not self.file_paths:
            raise RuntimeError(f"No data found in {self.root_dir}")

        # ----------------------------------------------------------------- #
        # Build / load Cd-scaler                                            #
        # ----------------------------------------------------------------- #
        self.scaler: StandardScaler
        try:
            self.scaler = (
                self._fit_and_save_scaler() if fit_scaler else load_scaler(SCALER_FILE)
            )
            if self.scaler.mean_ is None or self.scaler.scale_ is None:
                raise ValueError(
                    "Transformation Scaler not loaded correctly. Can't get the"
                    " mean and scale."
                )
        except Exception as e:
            logger.error(f"Error in getting transformation scaler. Error msg: {e}")
            raise e

        logger.info(
            f"{split:<5} dataset with {len(self.file_paths)} samples | "
            f"Cd Scaler mean={self.scaler.mean_[0]:.3f} | "
            f"Cd Scaler scale={self.scaler.scale_[0]:.3f}"
        )

    # ------------------------------ scaler ------------------------------ #
    def _fit_and_save_scaler(self) -> StandardScaler:
        """
        Fit a StandardScaler on Cd values from the split.
        """
        logger.info("Fitting StadnardScaler on Cd targets")
        cds = np.array(
            [self._load_npz(fp, only_cd=True, scale=False) for fp in self.file_paths],
            dtype=np.float32,
        ).reshape(-1, 1)
        scaler = StandardScaler().fit(cds)
        save_scaler(scaler, SCALER_FILE)
        logger.info(f"Scaler saved -> {SCALER_FILE}")
        return scaler

    # ---------------------------- I/O utils ----------------------------- #
    def _load_npz(self, fp: Path, *, only_cd: bool = False, scale: bool = True):
        """
        Load the required data from the .npz file stored at `fp`.

        Args:
            fp: Path to the npz file.
            only_cd: if True, return only the Cd value.
            scale: if True, scale the Cd before returning.

        Returns:
        - if only_cd is False: Returns a tuple
            - if padded is True: (slices, point_mask, cd)
                - slices: shape -> (num_slices, target_points, 2)
                - point_mask: shape -> (num_slices, target_points)
            - if padded is False: (slices, cd)
                - slices: shape -> (numb_slices,)
        - if only_cd is True: Cd value (float)
        """
        data = np.load(fp, allow_pickle=True)
        cd = data["Cd"]
        if scale:
            cd = float(self.scaler.transform([[cd]])[0, 0])
        if only_cd:
            return cd
        slices = data["slices"]
        if self.padded:
            point_mask = data["point_mask"]
            return slices.astype(np.float32), point_mask.astype(np.float32), cd
        else:
            return slices, cd

    # ------------------------- standard Dataset ------------------------- #
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        """
        Returns:
            x:
                • if padded=True: a 2-tuple
                    (slices: FloatTensor of shape (num_slices, target_points, 2),
                    point_mask: FloatTensor of shape (num_slices, target_points))
                • if padded=False:
                    slices_obj: a NumPy object-dtype array of length num_slices,
                                where slices_obj[i] has shape (Ni, 2)

            y:
                A 0-D FloatTensor containing the Cd value.
        """
        if self.padded:
            slices, point_mask, cd = self._load_npz(self.file_paths[idx])
            x = (
                torch.from_numpy(slices),
                torch.from_numpy(point_mask),
            )
        else:
            slices, cd = self._load_npz(self.file_paths[idx])
            x = slices
        y = torch.tensor(cd, dtype=torch.float32)
        return x, y
