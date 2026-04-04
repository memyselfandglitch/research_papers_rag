import os

import requests
import streamlit as st

def get_backend_url() -> str:
    backend_url = None
    try:
        # `st.secrets` will raise if no local/cloud secrets are configured.
        backend_url = st.secrets.get("BACKEND_URL")
    except Exception:
        backend_url = None

    if not backend_url:
        backend_url = os.environ.get("BACKEND_URL")

    if backend_url:
        return backend_url.rstrip("/")

    return "http://localhost:8000"


API_BASE_URL = get_backend_url()
IS_STREAMLIT_CLOUD = bool(os.environ.get("STREAMLIT_SERVER_PORT"))

st.title("Research Paper RAG")

if API_BASE_URL == "http://localhost:8000" and IS_STREAMLIT_CLOUD:
    st.error(
        "Backend URL is not configured. Add `BACKEND_URL` in Streamlit app secrets "
        "and point it to your deployed Render backend."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload"):
        try:
            with st.spinner("Extracting text, chunking, embedding, and indexing..."):
                pdf_bytes = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, pdf_bytes, "application/pdf")}
                res = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)

            if res.ok:
                st.success(res.json().get("message", "uploaded"))
            else:
                st.error(res.text)
        except requests.RequestException as exc:
            st.error(f"Could not reach backend at `{API_BASE_URL}`. Details: {exc}")

query = st.text_input("Ask a question")

if query:
    try:
        with st.spinner("Retrieving and generating answer..."):
            res = requests.get(f"{API_BASE_URL}/query", params={"q": query}, timeout=120)

        if res.ok:
            data = res.json()
            st.write(data.get("answer", ""))

            sources = data.get("sources", [])
            if sources:
                st.subheader("Sources")
                for s in sources:
                    st.write(f"Page {s.get('page')} - {s.get('text')[:200]}...")
        else:
            st.error(res.text)
    except requests.RequestException as exc:
        st.error(f"Could not reach backend at `{API_BASE_URL}`. Details: {exc}")
