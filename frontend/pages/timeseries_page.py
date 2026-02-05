import streamlit as st
import numpy as np

from backend.registry import MODELS
from backend.loader import load_torchscript_model
from backend.preprocessing.timeseries import preprocess_sine_window
from backend.inference import infer_lstm_timeseries
from frontend.components.markdown_viewer import render_markdown


@st.cache_resource
def get_model(model_path):
    return load_torchscript_model(model_path)


def render_timeseries_page(model_name: str):
    cfg = MODELS[model_name]

    st.header(model_name)

    # --- Docs ---
    render_markdown(cfg["docs"]["info"], "📘 About this model")
    render_markdown(cfg["docs"]["learning"], "🧪 Learning experience")

    st.subheader("Input Sequence (20 values)")

    st.caption(
        "Enter 20 numeric values (e.g. a sine wave window). "
        "The model predicts the next value."
    )

    # Default example (sine wave)
    default_values = np.sin(0.02 * np.arange(20)).tolist()

    values = []
    cols = st.columns(4)
    for i in range(20):
        with cols[i % 4]:
            val = st.number_input(
                f"x[{i}]",
                value=float(default_values[i]),
                key=f"ts_{i}"
            )
            values.append(val)

    if st.button("Run Inference"):
        try:
            x = preprocess_sine_window(values)

            model = get_model(cfg["model_path"])
            pred = infer_lstm_timeseries(model, x)

            st.success(f"Predicted next value: **{pred:.5f}**")

            # --- Visualization ---
            st.subheader("Sequence Visualization")
            seq = values + [pred]
            st.line_chart(seq)

        except Exception as e:
            st.error(str(e))
