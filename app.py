import streamlit as st
import torch
from pathlib import Path
import tempfile
import shutil
import sys

# Add the training src directory to the python path
sys.path.append("training")

from src.models.model import get_model
from src.inference.predict import predict_for_app
from src.utils.io import load_scaler

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("Drag Coefficient Prediction")
st.markdown("🐙 [View on GitHub](https://github.com/your-username/your-repo)")

st.info("Currently using the **PLM** model.")

# --- Model Loading ---
@st.cache_resource
def load_model_and_scaler():
    SCALER_PATH = Path("scalers/scaler.pkl")
    MODEL_PATH = Path("models/cd_plm_model.pt")
    device = torch.device("cpu")
    scaler = load_scaler(SCALER_PATH)
    model = get_model(model_type="plm")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, scaler, device

model, scaler, device = load_model_and_scaler()

# --- Prediction Logic ---
def run_prediction(file_path):
    try:
        st.write("Processing...")
        cd_value = predict_for_app(
            model=model, scaler=scaler, device=device, point_cloud_path=file_path
        )
        st.metric(label="Predicted Drag Coefficient (Cd)", value=f"{cd_value:.5f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- UI ---

# --- Demo Files ---
st.subheader("Load a Demo File")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Demo 1 (.pcd)"):
        run_prediction(Path("demo_files/DrivAer_F_D_WM_WW_0001.pcd"))
with col2:
    if st.button("Demo 2 (.pcd)"):
        run_prediction(Path("demo_files/DrivAer_F_D_WM_WW_0002.pcd"))
with col3:
    if st.button("Demo 1 (.paddle_tensor)"):
        run_prediction(Path("demo_files/DrivAer_F_D_WM_WW_0001.paddle_tensor"))
with col4:
    if st.button("Demo 2 (.paddle_tensor)"):
        run_prediction(Path("demo_files/DrivAer_F_D_WM_WW_0002.paddle_tensor"))

# --- File Uploader ---
st.subheader("Upload Your Own File")
uploaded_file = st.file_uploader(
    "Choose a .pcd or .paddle_tensor file", type=["pcd", "paddle_tensor"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = Path(tmp.name)
    
    run_prediction(tmp_path)
    tmp_path.unlink()