import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components.sidebar import model_selector
from frontend.pages.image_page import render_image_page
from frontend.pages.timeseries_page import render_timeseries_page
from backend.registry import MODELS

st.set_page_config(
    page_title="Deep Learning Model Zoo",
    layout="wide"
)

st.title("🧠 Deep Learning Model Zoo")

selected_model = model_selector()

if selected_model is None:
    st.info("Select a model from the sidebar.")
else:
    model_cfg = MODELS[selected_model]

    if model_cfg["type"] == "image":
        render_image_page(selected_model)

    elif model_cfg["type"] == "timeseries":
        render_timeseries_page(selected_model)

    else:
        st.warning("Model type not supported yet.")
