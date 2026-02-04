import streamlit as st
from backend.registry import MODELS

def model_selector():
    st.sidebar.title("Models")
    return st.sidebar.selectbox(
        "Select a model",
        list(MODELS.keys())
    )
