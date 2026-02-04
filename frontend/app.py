import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import streamlit as st

from frontend.components.sidebar import model_selector
from frontend.pages.image_page import render_image_page

st.set_page_config(
    page_title="Deep Learning Model Zoo",
    layout="wide"
)

st.title("🧠 Deep Learning Model Zoo")

selected_model = model_selector()

if selected_model == "LeNet5 (MNIST)":
    render_image_page(selected_model)
else:
    st.info("Select a model from the sidebar.")
