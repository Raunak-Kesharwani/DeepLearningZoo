import streamlit as st
from tokenizers import Tokenizer

from backend.registry import MODELS
from backend.loader import load_torchscript_model
from backend.inference import generate_text_lm
from frontend.components.markdown_viewer import render_markdown


@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = load_torchscript_model(model_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer


def render_language_model_page(model_name: str):
    cfg = MODELS[model_name]

    st.header(model_name)

    # ---- Docs ----
    render_markdown(cfg["docs"]["info"], "📘 About this model")
    render_markdown(cfg["docs"]["learning"], "🧪 Learning experience")

    st.subheader("Text Generation")

    prompt = st.text_input(
        "Prompt",
        value="The meaning of life is"
    )

    col1, col2 = st.columns(2)
    with col1:
        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=10,
            max_value=200,
            value=100,
            step=10
        )
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.2,
            max_value=1.5,
            value=0.8,
            step=0.1
        )

    if st.button("Generate"):
        with st.spinner("Generating text..."):
            model, tokenizer = load_model_and_tokenizer(
                cfg["model_path"],
                cfg["tokenizer"]
            )

            text = generate_text_lm(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

        st.subheader("Generated Text")
        st.write(text)
