import os
from typing import Dict, List, Optional, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sentence_transformers import SentenceTransformer

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI()

# Streamlit runs in the browser; CORS keeps local dev simple.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded to avoid slow startup while developing.
embedding_model: Optional[SentenceTransformer] = None
openai_client: Optional[OpenAI] = None

index: Optional[faiss.Index] = None
chunks_store: List[str] = []
metadata_store: List[Dict] = []


def get_embedding_model() -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("BAAI/bge-small-en")
    return embedding_model


def get_openai_client() -> OpenAI:
    """
    Create the OpenAI client lazily so the server can start without API keys.
    """
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AI_PROXY_BASE_URL")

        # If OPENAI_BASE_URL is set, we route through the proxy; otherwise we
        # default to the official OpenAI endpoint.
        if api_key and base_url:
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
        elif api_key:
            openai_client = OpenAI(api_key=api_key)
        elif base_url:
            # Will raise a clear error when we actually call the API.
            openai_client = OpenAI(base_url=base_url)
        else:
            openai_client = OpenAI()
    return openai_client


def extract_pages(pdf_bytes: bytes) -> List[str]:
    """Extract raw text per page from a PDF file (in-memory bytes)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[str] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pages.append(page.get_text() or "")
    return pages


def chunk_text_with_metadata(
    pages: List[str], chunk_size: int = 500, overlap: int = 100
) -> Tuple[List[str], List[Dict]]:
    """
    Word-based chunking with overlap, keeping page metadata for citations.

    Each chunk inherits the page it was cut from (so citations are meaningful).
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    step = chunk_size - overlap
    chunks: List[str] = []
    metas: List[Dict] = []

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


def embed_chunks(chunks: List[str]) -> np.ndarray:
    model = get_embedding_model()
    # normalize_embeddings=True enables cosine similarity with IndexFlatIP.
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def create_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (embeddings are normalized)
    index.add(np.asarray(embeddings, dtype="float32"))
    return index


def retrieve(query: str, k: int = 5) -> List[Dict]:
    if index is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    model = get_embedding_model()
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    _scores, indices = index.search(q_emb, k)

    results: List[Dict] = []
    for idx in indices[0]:
        if idx < 0:
            continue
        page = metadata_store[idx]["page"]
        results.append({"page": page, "text": chunks_store[idx]})
    return results


def generate_answer(query: str, context: str) -> str:
    client = get_openai_client()
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know.
Cite page numbers if possible.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


@app.post("/upload")
async def upload(file: UploadFile):
    global index, chunks_store, metadata_store

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pages = extract_pages(pdf_bytes)
    chunks, metas = chunk_text_with_metadata(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF.")

    embeddings = embed_chunks(chunks)
    index = create_index(embeddings)
    chunks_store = chunks
    metadata_store = metas

    return {"message": "uploaded", "chunks": len(chunks)}


@app.get("/query")
def query(q: str, k: int = 5):
    retrieved = retrieve(q, k=k)
    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in retrieved])

    answer = generate_answer(q, context)
    return {"answer": answer, "sources": retrieved}

