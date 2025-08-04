import streamlit as st
import torch
from pathlib import Path
import tempfile
import shutil

from app_utils import (
    Cd_PLM_Model,
    predict_cd,
    load_scaler,
)

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("Drag Coefficient Prediction")


# --- Model Loading ---
@st.cache_resource
def load_model_and_scaler():
    SCALER_PATH = Path("scalers/scaler.pkl")
    MODEL_PATH = Path("models/cd_plm_model.pt")
    device = torch.device("cpu")
    scaler = load_scaler(SCALER_PATH)
    model = Cd_PLM_Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, scaler, device


model, scaler, device = load_model_and_scaler()

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a .pcd or .paddle_tensor file", type=["pcd", "paddle_tensor"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = Path(tmp.name)

    try:
        st.write("File uploaded successfully. Processing...")
        cd_value = predict_cd(
            point_cloud_path=tmp_path,
            model=model,
            scaler=scaler,
            device=device,
        )
        st.metric(label="Predicted Drag Coefficient (Cd)", value=f"{cd_value:.5f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        tmp_path.unlink()