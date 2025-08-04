# Experiments

This directory is used to store all the artifacts related to a specific experiment. When you run the `train` command, a new directory is created here with the name you provide in the `--exp-name` argument.

Each experiment directory will contain:

*   `checkpoints/`: Saved model checkpoints.
*   `tensorboard/`: TensorBoard logs for monitoring training progress.
