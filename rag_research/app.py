import os

import requests
import streamlit as st

backend_url = None
# try:
#     # `st.secrets` will raise `StreamlitSecretNotFoundError` if no local
#     # secrets are configured. So keep this optional.
#     backend_url = st.secrets.get("BACKEND_URL")
# except Exception:
#     backend_url = None

# if not backend_url:
backend_url = os.environ.get("BACKEND_URL")

API_BASE_URL = backend_url or "http://localhost:8000"

st.title("Research Paper RAG")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload"):
        with st.spinner("Extracting text, chunking, embedding, and indexing..."):
            pdf_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, pdf_bytes, "application/pdf")}
            res = requests.post(f"{API_BASE_URL}/upload", files=files)

        if res.ok:
            st.success(res.json().get("message", "uploaded"))
        else:
            st.error(res.text)

query = st.text_input("Ask a question")

if query:
    with st.spinner("Retrieving and generating answer..."):
        res = requests.get(f"{API_BASE_URL}/query", params={"q": query})

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

