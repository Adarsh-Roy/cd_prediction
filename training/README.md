# Drag Coefficient Prediction - Training

This directory contains the code for training and evaluating drag coefficient prediction models.

## Project Overview

The goal of this project is to predict the drag coefficient (Cd) of a vehicle based on its 3D point cloud representation. The process involves:

1.  **Slicing:** Taking 3D point clouds and slicing them into a sequence of 2D representations.
2.  **Preprocessing:** Preparing the sliced data for the models, including padding and masking.
3.  **Training:** Training a model to predict the Cd value from the prepared data.
4.  **Evaluation:** Evaluating the trained model on a test set.
5.  **Inference:** Using a trained model to predict the Cd for a new, unseen point cloud.

### Data Requirements for Training

For training the models, the system expects raw 3D point cloud files (e.g., `.pcd` or `.paddle_tensor` format) to be placed in the `training/data/raw/PointClouds/` directory. The `slice` command (detailed below) will process these files.

Additionally, the `clean_drag_coefficients.csv` file is required to provide the labels (truth values) for the drag coefficients (Cd) corresponding to the point cloud data. A sample `clean_drag_coefficients.csv` is provided in the `training/data/raw/` directory. This sample file contains the drag coefficients for the DrivAerNet++ dataset and serves as an example of the expected format.

## Environment Setup

1.  **Navigate to the training directory:**

    ```bash
    cd training
    ```

2.  **Create and activate a virtual environment (with any virtual environment manager of your choice, I recommend `uv`):**

    ```bash
    uv venv .venv --python==3.11
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

The main entry point for all operations is `src/main.py`. The available commands are:

*   `slice`: Slice 3D point clouds into 2D representations.
*   `visualize`: Visualize the 2D slices.
*   `prep`: Prepare the dataset for training.
*   `fit_scaler`: Fit a scaler to the training data.
*   `train`: Train a new model.
*   `evaluate`: Evaluate a trained model.
*   `predict`: Run inference on a single point cloud.

### Data Preprocessing

**1. Slice the Point Clouds**

This command takes the raw 3D point clouds and creates 2D slices.

```bash
python -m src.main slice --input-dir <path/to/point_clouds> --output-dir <path/to/slices>
```

**2. Prepare the Dataset**

This command prepares the sliced data for training. Use the `--pad` flag if your model requires padded inputs.

```bash
python -m src.main prep --slice-dir <path/to/slices> --output-dir <path/to/prepared_data> --pad
```

**3. Fit the Scaler**

This command fits a `StandardScaler` to the training data, which is used to normalize the Cd values.

```bash
python -m src.main fit_scaler --config <path/to/config.yaml>
```

### Training

To train a model, you need to provide an experiment name and a configuration file.

```bash
python -m src.main train --exp-name <experiment_name> --config <path/to/config.yaml>
```

**TensorBoard**

During training, you can monitor the progress using TensorBoard. The logs are saved in the `experiments/<experiment_name>` directory.

To launch TensorBoard, run the following command from the `training` directory:

```bash
tensorboard --logdir experiments
```

### Evaluation

To evaluate a trained model, you need to provide the configuration file and the path to the saved checkpoint.

```bash
python -m src.main evaluate --config <path/to/config.yaml> --checkpoint <path/to/checkpoint.pt>
```

### Inference

To run inference on a single point cloud, you need the configuration file, a checkpoint, and the path to the point cloud file.

```bash
python -m src.main predict --config <path/to/config.yaml> --checkpoint <path/to/checkpoint.pt> --point-cloud <path/to/point_cloud.[paddle_tensor,pcd]>
```
