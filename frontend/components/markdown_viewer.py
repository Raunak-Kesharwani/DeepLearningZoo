import streamlit as st
from pathlib import Path

def render_markdown(path, title=None):
    if title:
        st.subheader(title)

    content = Path(path).read_text(encoding="utf-8")
    st.markdown(content)
