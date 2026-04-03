import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

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
reranker_model: Optional[CrossEncoder] = None

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


def get_reranker() -> CrossEncoder:
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    return reranker_model


def format_chunks(chunks: List[Dict[str, str]]) -> str:
    # Keep page tags so the LLM can cite.
    return "\n\n".join([f"[Page {c['page']}]\n{c['text']}" for c in chunks])


def llm_chat(messages: List[Dict[str, str]]) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def expand_query(query: str, n_queries: int = 5) -> List[str]:
    prompt = f"""
Rewrite the user's query into {n_queries} diverse, specific search queries
that would retrieve relevant sections from a research paper.

User query:
{query}

Return ONLY the {n_queries} queries, one per line, without numbering or bullets.
"""
    text = llm_chat([{"role": "user", "content": prompt}]).strip()
    # Support both plain lines and accidental bullet/number formats.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned: List[str] = []
    for l in lines:
        l = re.sub(r"^[-*]\s*", "", l)
        l = re.sub(r"^\d+\.\s*", "", l)
        if l:
            cleaned.append(l)
    # Ensure we return exactly n_queries (truncate or pad with original query).
    if len(cleaned) >= n_queries:
        return cleaned[:n_queries]
    while len(cleaned) < n_queries:
        cleaned.append(query)
    return cleaned


def multi_retrieve(queries: List[str], k: int = 5) -> List[Dict[str, str]]:
    all_results: List[Dict[str, str]] = []
    seen: set[tuple[int, str]] = set()

    for q in queries:
        retrieved = retrieve(q, k=k)
        for r in retrieved:
            key = (int(r["page"]), r["text"])
            if key in seen:
                continue
            seen.add(key)
            all_results.append(r)
    return all_results


def rerank_chunks(query: str, chunks: List[Dict[str, str]], top_n: int = 8) -> List[Dict[str, str]]:
    if not chunks:
        return []
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _score in ranked[:top_n]]


def extract_info(context_chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    context = format_chunks(context_chunks)
    prompt = f"""
You are extracting structured reproduction-critical information from research paper excerpts.

Context excerpts (with page tags):
{context}

Extract the following as accurately as possible:
- dataset_used (string)
- model_architecture (string)
- training_procedure (string)
- evaluation_method (string)
- reproduction_steps (string; step-by-step if present, otherwise a short summary)
- missing_information (array of strings for anything the context does not contain)

Return STRICT JSON with exactly those keys.
If something is not present, use an empty string (for string fields) and [] (for missing_information).
"""
    text = llm_chat([{"role": "user", "content": prompt}]).strip()
    default_extracted: Dict[str, Any] = {
        "dataset_used": "",
        "model_architecture": "",
        "training_procedure": "",
        "evaluation_method": "",
        "reproduction_steps": "",
        "missing_information": [],
    }

    # Try to parse JSON even if the model adds surrounding text.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            default_extracted["missing_information"] = ["extraction_json_parse_failed"]
            return default_extracted
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            default_extracted["missing_information"] = ["extraction_json_parse_failed"]
            return default_extracted


def synthesize_answer(query: str, extracted: Dict[str, Any], context_chunks: List[Dict[str, str]]) -> str:
    context = format_chunks(context_chunks)
    extracted_json = json.dumps(extracted, ensure_ascii=False)
    prompt = f"""
Using ONLY the extracted information and the context excerpts below, answer the user's question.

Rules:
- Provide a reproduction-focused response with clear hierarchy.
- If the question asks about reproduction, training, dataset construction, validation, or evaluation: produce step-by-step instructions.
- If required details are missing: explicitly say what's missing (and list it in "What is NOT specified").
- When stating a fact, cite the source page in the form [Page X] (use the page tags from the context excerpts).
- Do not use outside knowledge.

Extracted JSON:
{extracted_json}

Context excerpts:
{context}

User question:
{query}

Output format (Markdown):
1. Prerequisites (data, tools, platforms, access requirements)
2. Step-by-step procedure (ordered; include inputs/tools/outputs where relevant)
3. Key parameters / constraints (hyperparameters, filtering rules, split definitions, validation protocol)
4. What is NOT specified in the paper (merge missing_information + any empty extracted fields)

Hard constraints:
- If a field is empty/[] in the extracted JSON, do not guess it; put it in section (4).
- Keep statements grounded in the provided context; no generic filler.
"""
    return llm_chat([{"role": "user", "content": prompt}]).strip()


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
    if index is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    expanded = expand_query(q, n_queries=5)
    candidates = multi_retrieve(expanded, k=k)  # ~25 chunks
    top_chunks = rerank_chunks(q, candidates, top_n=8)

    # Structured extraction from the top context.
    extracted = extract_info(top_chunks)
    answer = synthesize_answer(q, extracted, top_chunks)

    return {"answer": answer, "sources": top_chunks, "expanded_queries": expanded}

