from src.models.experiment_models.model_PLM import Cd_PLM_Model


def get_model(
    model_type: str = "plm",
    slice_input_dim: int = 2,
    slice_emb_dim: int = 256,
    design_emb_dim: int = 512,
    lstm_hidden_dim: int = 256,
):
    """
    Return an instantiated Cd model based on model_type.

    - "plm" → PointNet + LSTM + MLP

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
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. use 'plm'.")
