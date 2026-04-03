import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded globals
embedding_model = None
reranker_model = None
openai_client = None
faiss_index = None

index = None
chunks_store: List[str] = []
metadata_store: List[Dict] = []


# ------------------ HEALTH ------------------
@app.get("/")
def health():
    return {"status": "ok"}


# ------------------ MODELS ------------------
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("BAAI/bge-small-en")
    return embedding_model


def get_reranker():
    global reranker_model
    if reranker_model is None:
        from sentence_transformers import CrossEncoder
        reranker_model = CrossEncoder("BAAI/bge-reranker-base")
        # Optional lighter:
        # reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker_model


def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AI_PROXY_BASE_URL")

        if api_key and base_url:
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
        elif api_key:
            openai_client = OpenAI(api_key=api_key)
        elif base_url:
            openai_client = OpenAI(base_url=base_url)
        else:
            openai_client = OpenAI()
    return openai_client


# ------------------ PDF ------------------
def extract_pages(pdf_bytes: bytes) -> List[str]:
    import fitz  # lazy load

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pages.append(page.get_text() or "")
    return pages


# ------------------ CHUNKING ------------------
def chunk_text_with_metadata(pages, chunk_size=500, overlap=100):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    step = chunk_size - overlap
    chunks, metas = [], []

    for page_idx, page_text in enumerate(pages):
        text = (page_text or "").strip()
        if not text:
            continue

        words = text.split()
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))
            metas.append({"page": page_idx + 1})

    return chunks, metas


# ------------------ EMBEDDINGS ------------------
def embed_chunks(chunks):
    model = get_embedding_model()
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def create_index(embeddings):
    import faiss  # lazy load

    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    return idx


# ------------------ RETRIEVAL ------------------
def retrieve(query, k=5):
    global index

    if index is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    model = get_embedding_model()
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    _, indices = index.search(q_emb, k)

    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        results.append({
            "page": metadata_store[idx]["page"],
            "text": chunks_store[idx]
        })
    return results


def multi_retrieve(queries, k=5):
    seen = set()
    results = []

    for q in queries:
        for r in retrieve(q, k):
            key = (r["page"], r["text"])
            if key not in seen:
                seen.add(key)
                results.append(r)
    return results


def rerank_chunks(query, chunks, top_n=8):
    if not chunks:
        return []
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]


# ------------------ LLM ------------------
def llm_chat(messages):
    client = get_openai_client()
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    return res.choices[0].message.content or ""


def expand_query(query, n_queries=5):
    prompt = f"""Rewrite into {n_queries} search queries:\n{query}"""
    text = llm_chat([{"role": "user", "content": prompt}]).strip()
    lines = [re.sub(r"^[-*\\d.\\s]+", "", l.strip()) for l in text.splitlines() if l.strip()]
    return (lines + [query] * n_queries)[:n_queries]


# ------------------ API ------------------
@app.post("/upload")
async def upload(file: UploadFile):
    global index, chunks_store, metadata_store

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    pages = extract_pages(pdf_bytes)
    chunks, metas = chunk_text_with_metadata(pages)

    embeddings = embed_chunks(chunks)
    index = create_index(embeddings)

    chunks_store = chunks
    metadata_store = metas

    return {"message": "uploaded", "chunks": len(chunks)}


@app.get("/query")
def query(q: str, k: int = 5):
    if index is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    expanded = expand_query(q)
    candidates = multi_retrieve(expanded, k)
    top_chunks = rerank_chunks(q, candidates)

    answer = llm_chat([
        {"role": "user", "content": f"Answer using:\n{top_chunks}\n\nQ:{q}"}
    ])

    return {
        "answer": answer,
        "sources": top_chunks,
        "expanded_queries": expanded
    }