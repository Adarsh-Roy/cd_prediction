from pathlib import Path

# ------------------------------------------------------------------ #
#  Project roots                                                     #
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/Cd_prediction/C_d
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
EXP_DIR = PROJECT_ROOT / "experiments"

# ------------------------------------------------------------------ #
#  Raw / processed paths                                             #
# ------------------------------------------------------------------ #
POINT_CLOUDS_DIR = RAW_DIR / "PointClouds"
SUBSET_DIR = RAW_DIR / "subset_dir"

SLICE_DIR = PROC_DIR / "slices"
PREPARED_DATASET_DIR = PROC_DIR / "prepared_dataset"
PREPARED_DATASET_PADDED_DIR = PROC_DIR / "prepared_dataset" / "padded"
SCALER_FILE = PROC_DIR / "scalers" / "scaler.pkl"  # <— single file
# TODO: [LATER] support exp-name based scaler files when we have different datasets


# ------------------------------------------------------------------ #
#  Cd table                                                          #
# ------------------------------------------------------------------ #
# Master CSV produced by the CFD post-processing workflow
DRAG_CSV = RAW_DIR / "cleaned_drag_coefficients.csv"

# ------------------------------------------------------------------ #
#  Pre-processing defaults                                           #
# ------------------------------------------------------------------ #
DEFAULT_NUM_SLICES = 80
DEFAULT_SLICE_AXIS = "x"
DEFAULT_TARGET_POINTS = 6_500


# ------------------------------------------------------------------ #
#  Model to Padding mapping                                          #
# ------------------------------------------------------------------ #
model_to_padded: dict[str, bool] = {"plm": True, "dlm": False}
