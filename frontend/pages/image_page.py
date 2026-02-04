import streamlit as st
from PIL import Image
import torch

from backend.registry import MODELS
from backend.loader import load_torchscript_model
from backend.preprocessing.image import preprocess_lenet_mnist
from backend.inference import infer_image_classifier
from frontend.components.markdown_viewer import render_markdown


@st.cache_resource
def get_model(model_path):
    return load_torchscript_model(model_path)


def render_image_page(model_name: str):
    cfg = MODELS[model_name]

    st.header(model_name)

    # --- Docs ---
    render_markdown(cfg["docs"]["info"], "📘 About this model")
    render_markdown(cfg["docs"]["learning"], "🧪 Learning experience")

    # --- Image input ---
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is None:
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", width=200)

    # --- Inference ---
    if st.button("Run Inference"):
        model = get_model(cfg["model_path"])

        x = preprocess_lenet_mnist(image)

        result = infer_image_classifier(
            model=model,
            image_tensor=x,
            class_names=cfg["classes"]
        )

        st.success(f"Prediction: **{result['prediction']}**")

        st.subheader("Class Probabilities")
        st.bar_chart(result["probabilities"])
