from src.config.constants import DEFAULT_NUM_SLICES
from src.models.experiment_models.model_PLM import Cd_PLM_Model


def get_model(
    model_type: str = "plm",
    slice_input_dim: int = 2,
    slice_emb_dim: int = 256,
    design_emb_dim: int = 512,
    lstm_hidden_dim: int = 256,
    k_neighbors: int = 16,
    num_slices: int = DEFAULT_NUM_SLICES,
):
    """
    Return an instantiated Cd model based on model_type.

    - "plm" → PointNet + LSTM + MLP
    - "dlm" → DGCNN + LSTM + MLP
    - "ptm" → PointNet + Transformer + MLP [LATER]
    - "dtm" → DGCNN + Transformer + MLP (uses dynamic slice batching) [LATER]

    Returns:
        nn.Module
    """
    if model_type == "plm":
        return Cd_PLM_Model(
            slice_input_dim=slice_input_dim,
            slice_emb_dim=slice_emb_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            design_emb_dim=design_emb_dim,
        )
    elif model_type == "dlm":
        raise NotImplementedError("DLM is not implemented yet.")

    elif model_type == "ptm":
        raise NotImplementedError("PTM is not implemented yet.")

    elif model_type == "dtm":
        raise NotImplementedError("DTM is not implement yet.")

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. use 'plm'.")
