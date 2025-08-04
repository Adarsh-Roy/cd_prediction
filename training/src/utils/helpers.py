import torch
from sklearn.preprocessing import StandardScaler


def prepare_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_ragged_batch_fn(batch, device, non_blocking):
    slice_batches, cd_values = batch
    # move each PyG Batch
    slice_batches = [sb.to(device) for sb in slice_batches]
    # move targets
    cd_values = cd_values.to(device, non_blocking=non_blocking)
    return slice_batches, cd_values


def make_unscale(scaler: StandardScaler):
    def _unscale(x, y, y_pred):
        """
        This is used to unscale the output from the model before passing it on for
        metrics calculation. Ignite passes `(y_pred, y)` in the `output` argument.

        Args:
            output: A tuple of (y_pred, y)

        Returns:
            A tuple of unscaled prediction and real values. That is, (y_pred_u, y_u)
        """

        y_pred_u = (
            torch.from_numpy(
                scaler.inverse_transform(y_pred.detach().cpu().reshape(-1, 1))
            )
            .to(y_pred.device)
            .view_as(y_pred)
        )
        y_u = (
            torch.from_numpy(scaler.inverse_transform(y.detach().cpu().reshape(-1, 1)))
            .to(y.device)
            .view_as(y)
        )
        return y_pred_u, y_u

    return _unscale
