import streamlit as st
from tokenizers import Tokenizer

from backend.registry import MODELS
from backend.loader import load_torchscript_model
from backend.inference import generate_answer_seq2seq
from frontend.components.markdown_viewer import render_markdown


@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = load_torchscript_model(model_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer


def render_qa_page(model_name: str):
    cfg = MODELS[model_name]

    st.header(model_name)

    # ---- Docs ----
    render_markdown(cfg["docs"]["info"], "📘 About this model")
    render_markdown(cfg["docs"]["learning"], "🧪 Learning experience")

    st.subheader("Question Answering")

    question = st.text_input(
        "Question",
        value="What is artificial intelligence?"
    )

    context = st.text_area(
        "Context",
        value=(
            "Artificial intelligence is a field of computer science "
            "that focuses on creating machines capable of performing "
            "tasks that typically require human intelligence."
        ),
        height=150
    )

    max_len = st.slider(
        "Max answer length",
        min_value=10,
        max_value=80,
        value=40,
        step=5
    )

    if st.button("Generate Answer"):
        with st.spinner("Generating answer..."):
            model, tokenizer = load_model_and_tokenizer(
                cfg["model_path"],
                cfg["tokenizer"]
            )

            answer = generate_answer_seq2seq(
                model=model,
                tokenizer=tokenizer,
                question=question,
                context=context,
                max_len=max_len
            )

        st.subheader("Answer")
        st.write(answer)
