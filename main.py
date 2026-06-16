import os
os.environ["PYTHONUTF8"] = "1"

import re
import uuid
import traceback
from pathlib import Path
from typing import List, Optional
from collections import deque

import asyncio
import threading
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import pypdf
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue
)
from groq import Groq

# ─── Config ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

COLLECTION_NAME   = "pdf_chunks"
EMBED_MODEL       = "multi-qa-MiniLM-L6-cos-v1"
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_DIM        = 384

MAX_PDFS          = 5
MAX_HISTORY_PAIRS = 10
UPSERT_BATCH_SIZE = 100
EMBED_BATCH_SIZE  = 64

# ── Semantic chunking config ──────────────────────────────────────────────────
SEMANTIC_THRESHOLD   = 0.30  # lower = larger chunks = more complete context per chunk
MIN_CHUNK_CHARS      = 250   # higher = forces more sentences per chunk
MAX_CHUNK_CHARS      = 1500

# ── Retrieval config ──────────────────────────────────────────────────────────
RETRIEVAL_LIMIT     = 100   # wide net — ensures reranker always has enough candidates
SCORE_THRESHOLD     = 0.0   # zero — reranker is the only quality filter
MAX_CHUNKS_PER_FILE = 10    # high cap — reranker decides quality, not this
FINAL_CHUNK_LIMIT   = 6     # chunks sent to LLM after reranking
HISTORY_PAIRS_TO_LLM = 10

# Reranker score threshold — chunks below this after reranking are considered
# not relevant enough to use as context. LLM falls back to general knowledge.
RERANK_SCORE_THRESHOLD = -9.5  # only fall back to general knowledge if truly nothing relevant

# ─── Greeting detection ───────────────────────────────────────────────────────
GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|day)|"
    r"what'?s\s*up|sup|hiya|yo)\W*\s*$",
    re.IGNORECASE,
)

GREETING_RESPONSE = """👋 Hello! I'm your **PDF Research Assistant**, powered by Groq (LLaMA 3.3 70B) and Qdrant.

Here's what I can do for you:

📄 **Upload & Index PDFs** — Upload up to 5 PDF documents. I'll extract, chunk, and index them semantically so every idea is searchable.

🔍 **Answer Questions** — Ask me anything about your uploaded documents. I'll retrieve the most relevant passages and give you a precise, cited answer.

📚 **Cross-PDF Reasoning** — Ask questions that span multiple documents. I'll compare, contrast, and synthesize information across all your PDFs.

🗂 **Source Citations** — Every answer includes the exact PDF filename and page number so you can verify the source.

💬 **Conversation Memory** — I remember the last 10 exchanges so you can ask follow-up questions naturally.

🌐 **General Knowledge** — If your question isn't covered by the uploaded PDFs, I'll answer from general knowledge and let you know.

**To get started:** Upload your PDFs using the panel on the left, then ask me anything!"""

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="PDF RAG Agent — Qdrant", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Global error handlers ────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[UNHANDLED ERROR] {request.url.path}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# ─── Embedding model ──────────────────────────────────────────────────────────
print(f"[startup] Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
_test_dim = len(embedder.encode("test").tolist())
print(f"[startup] Embedding model ready. Output dim: {_test_dim}")
assert _test_dim == VECTOR_DIM, (
    f"FATAL: model outputs {_test_dim} dims but VECTOR_DIM={VECTOR_DIM}."
)

# ─── Cross-encoder reranker ───────────────────────────────────────────────────
# The reranker reads the question AND each chunk together and scores how well
# the chunk actually answers the question — much more accurate than cosine
# similarity alone which only compares vectors independently.
print(f"[startup] Loading reranker model: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)
print(f"[startup] Reranker ready.")

# ─── Qdrant setup ─────────────────────────────────────────────────────────────
def make_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        return QdrantClient(**kwargs)
    return QdrantClient(":memory:")

qdrant = make_qdrant_client()

existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[startup] Created collection '{COLLECTION_NAME}'.")
else:
    print(f"[startup] Using existing collection '{COLLECTION_NAME}'.")

# ─── Job store (tracks background indexing jobs) ─────────────────────────────
# Each upload job gets a unique job_id. The frontend polls /status/{job_id}
# until status changes from "indexing" to "done" or "error".
job_store: dict = {}
# job_store[job_id] = {
#   "status": "indexing" | "done" | "error",
#   "session_id": str,
#   "files": list of result dicts,
#   "error": str (only on error)
# }

# ─── Conversation memory ──────────────────────────────────────────────────────
conversation_store: dict = {}

def get_history(sid: str) -> list:
    return list(conversation_store.get(sid, []))

def get_trimmed_history(sid: str) -> list:
    full = list(conversation_store.get(sid, []))
    return full[-(HISTORY_PAIRS_TO_LLM * 2):]

def add_to_history(sid: str, role: str, content: str):
    if sid not in conversation_store:
        conversation_store[sid] = deque(maxlen=MAX_HISTORY_PAIRS * 2)
    conversation_store[sid].append({"role": role, "content": content})

# ─── PDF text extraction ──────────────────────────────────────────────────────
def extract_text_by_page(path: Path) -> list:
    try:
        reader = pypdf.PdfReader(str(path))
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            if text:
                pages.append({"page_number": page_num, "text": text})
            print(f"  PAGE {page_num} LENGTH: {len(text)}")
        print(f"TOTAL PAGES WITH TEXT: {len(pages)}")
        return pages
    except Exception as e:
        print(f"[PDF extraction error] {e}")
        return []

# ─── Semantic chunking ────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    result = []
    for s in sentences:
        parts = s.split("\n")
        result.extend([p.strip() for p in parts if p.strip()])
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk_text(text: str, page_number: int, filename: str) -> list:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        chunk_text = sentences[0]
        return [{
            "text": chunk_text,
            "embed_text": f"[Source: {filename}, Page {page_number}]\n{chunk_text}",
            "page_number": page_number,
            "num_sentences": 1,
        }]

    print(f"    Embedding {len(sentences)} sentences for semantic chunking...")
    vecs = embedder.encode(sentences, show_progress_bar=False, batch_size=64)

    chunks = []
    current_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(vecs[i - 1], vecs[i])
        current_text = " ".join(current_sentences)

        if len(current_text) >= MAX_CHUNK_CHARS:
            chunk_text = current_text.strip()
            chunks.append({
                "text": chunk_text,
                "embed_text": f"[Source: {filename}, Page {page_number}]\n{chunk_text}",
                "page_number": page_number,
                "num_sentences": len(current_sentences),
            })
            current_sentences = [sentences[i]]
            continue

        if sim < SEMANTIC_THRESHOLD:
            if len(current_text) >= MIN_CHUNK_CHARS:
                chunk_text = current_text.strip()
                chunks.append({
                    "text": chunk_text,
                    "embed_text": f"[Source: {filename}, Page {page_number}]\n{chunk_text}",
                    "page_number": page_number,
                    "num_sentences": len(current_sentences),
                })
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "embed_text": f"[Source: {filename}, Page {page_number}]\n{chunk_text}",
                "page_number": page_number,
                "num_sentences": len(current_sentences),
            })

    return chunks


def chunk_pages(pages: list, filename: str = "") -> list:
    print("CHUNKING STARTED (semantic — topic-boundary detection)")
    all_chunks = []
    try:
        for page in pages:
            text = str(page.get("text", ""))
            page_number = page.get("page_number", -1)
            if not text.strip():
                continue
            page_chunks = semantic_chunk_text(text, page_number, filename)
            all_chunks.extend(page_chunks)
            print(f"  Page {page_number}: {len(page_chunks)} semantic chunks")
        print(f"TOTAL CHUNKS: {len(all_chunks)} (dynamic sizes, topic-bounded)")
        return all_chunks
    except Exception as e:
        traceback.print_exc()
        print(f"CHUNKING ERROR: {e}")
        return []

# ─── PDF ingestion ────────────────────────────────────────────────────────────
def ingest_pdf(path: Path, filename: str, session_id: str) -> dict:
    pages = extract_text_by_page(path)
    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'")

    chunks = chunk_pages(pages, filename=filename)
    if not chunks:
        raise ValueError(f"Chunking produced 0 chunks for '{filename}'")
    print(f"CHUNKS CREATED: {len(chunks)}")

    pdf_id = str(uuid.uuid4())

    all_embeddings = []
    try:
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch_texts = [c["embed_text"] for c in chunks[i:i + EMBED_BATCH_SIZE]]
            batch_vecs = embedder.encode(batch_texts, show_progress_bar=False).tolist()
            all_embeddings.extend(batch_vecs)
            print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
        print(f"TOTAL EMBEDDINGS: {len(all_embeddings)}")
    except Exception as e:
        raise ValueError(f"Embedding failed: {e}")

    if len(all_embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count mismatch: {len(all_embeddings)} embeddings for {len(chunks)} chunks"
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=all_embeddings[i],
            payload={
                "session_id": session_id,
                "pdf_id": pdf_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": chunks[i]["page_number"],
                "num_sentences": chunks[i].get("num_sentences", 1),
                "text": chunks[i]["text"],
            },
        )
        for i in range(len(chunks))
    ]

    total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    for batch_num, start in enumerate(range(0, len(points), UPSERT_BATCH_SIZE), 1):
        batch = points[start:start + UPSERT_BATCH_SIZE]
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"  Upserted batch {batch_num}/{total_batches} ({len(batch)} points)")
        except Exception as e:
            raise ValueError(f"Qdrant upsert failed at batch {batch_num}: {e}")

    print(f"UPSERT SUCCESS — {len(points)} total points")
    return {"pdf_id": pdf_id, "chunks": len(chunks)}

# ─── Reranking ────────────────────────────────────────────────────────────────
def rerank_chunks(query: str, chunks: list) -> list:
    """
    Cross-encoder reranking.

    The bi-encoder (Qdrant search) embeds the question and chunks separately
    and compares them as vectors — fast but shallow.

    The cross-encoder reads the question AND each chunk together in one pass
    and scores how well the chunk actually answers the question — slower but
    much more accurate.

    Pipeline:
      Qdrant retrieves 80 candidates (fast, broad)
          ↓
      Cross-encoder rescores all 80 (accurate, targeted)
          ↓
      Top 6 by rerank score sent to LLM
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    print(f"  [rerank] Scoring {len(pairs)} chunks with cross-encoder...")

    scores = reranker.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort by rerank score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Log rerank results
    print(f"  [rerank] Top scores: "
          f"{[round(c['rerank_score'], 2) for c in reranked[:FINAL_CHUNK_LIMIT]]}")
    print(f"  [rerank] Top sources: "
          f"{[c['filename'] for c in reranked[:FINAL_CHUNK_LIMIT]]}")

    return reranked


# ─── Query expansion ─────────────────────────────────────────────────────────
def expand_query(question: str) -> list:
    """
    Two-step query expansion:
    1. Split compound questions ("who is X and what is Y") into sub-questions.
       Each sub-question is a focused search on one concept.
    2. Generate rephrasings for each sub-question to catch vocabulary mismatches.

    Example:
      Input:  "who is anil ananthaswamy and what is a lazy controller"
      Split:  ["who is anil ananthaswamy", "what is a lazy controller"]
      Expand: ["Anil Ananthaswamy science author biography",
               "Anil Ananthaswamy Why Machines Learn writer",
               "lazy controller definition concept",
               "lazy controller system behaviour explanation"]
    """
    if not GROQ_API_KEY:
        return [question]

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query generator for a document retrieval system. "
                        "Given a question, do the following:\n"
                        "1. If the question contains multiple sub-questions (joined by 'and', 'also', ','), "
                        "split them into separate focused queries.\n"
                        "2. For each concept, generate 1-2 rephrasings using different vocabulary "
                        "that might match how a textbook or article would phrase it.\n"
                        "Return ONLY the queries, one per line, no numbering, no explanation, max 6 lines."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}",
                },
            ],
            max_tokens=200,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        variants = [q.strip() for q in raw.split("\n") if q.strip()][:6]
        # Always include the original question
        all_queries = [question] + variants
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                deduped.append(q)
        print(f"  [query expansion] {len(deduped)} queries: {deduped}")
        return deduped
    except Exception as e:
        print(f"  [query expansion failed] {e} — using original query only")
        return [question]


# ─── Semantic search ──────────────────────────────────────────────────────────
def search_qdrant(
    query: str,
    session_id: str,
    pdf_id: Optional[str] = None,
) -> List[dict]:
    # Step 1: expand the query into multiple variants
    # This fixes vocabulary mismatch — user phrasing vs PDF phrasing
    queries = expand_query(query)

    conditions = [FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    if pdf_id:
        conditions.append(FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id)))
    search_filter = Filter(must=conditions)

    try:
        # Step 2: run Qdrant search for EACH query variant, merge results
        # deduplicating by chunk_index+filename so the same chunk isn't
        # scored twice by the reranker
        seen_ids: set = set()
        all_hits = []

        for q_variant in queries:
            try:
                enhanced = "Represent this sentence for searching relevant passages: " + q_variant
                q_vec = embedder.encode(enhanced, show_progress_bar=False).tolist()
                hits = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=q_vec,
                    query_filter=search_filter,
                    limit=RETRIEVAL_LIMIT,
                    with_payload=True,
                )
                for h in hits:
                    dedup_key = f"{h.payload['filename']}:{h.payload['chunk_index']}"
                    if dedup_key not in seen_ids:
                        seen_ids.add(dedup_key)
                        all_hits.append(h)
            except Exception as e:
                print(f"  [search variant error] {e}")
                continue

        if not all_hits:
            print("  [search] No chunks retrieved across all queries")
            return []

        print(f"  [search] {len(all_hits)} total unique candidates across all query variants")

        # Step 3: sort merged hits by cosine score, apply per-file diversity cap
        all_hits.sort(key=lambda h: h.score, reverse=True)

        file_counts: dict = {}
        diverse = []
        for h in all_hits:
            fname = h.payload["filename"]
            if file_counts.get(fname, 0) < MAX_CHUNKS_PER_FILE:
                file_counts[fname] = file_counts.get(fname, 0) + 1
                diverse.append(h)

        print(f"  [search] {len(diverse)} candidates from "
              f"{len(file_counts)} PDF(s) going to reranker")

        # Convert to dicts for reranking
        candidates = [
            {
                "text": h.payload["text"],
                "filename": h.payload["filename"],
                "pdf_id": h.payload["pdf_id"],
                "chunk_index": h.payload["chunk_index"],
                "page_number": h.payload["page_number"],
                "num_sentences": h.payload.get("num_sentences", "?"),
                "score": round(h.score, 4),
                "rerank_score": 0.0,
            }
            for h in diverse
        ]

        # Step 4: rerank all candidates with cross-encoder
        reranked = rerank_chunks(query, candidates)

        # Step 5: take top FINAL_CHUNK_LIMIT
        # Only fall back to general knowledge if the BEST chunk scores below threshold
        # (if even the top result is irrelevant, the rest will be worse)
        top = reranked[:FINAL_CHUNK_LIMIT]
        best_score = top[0]["rerank_score"] if top else -99

        if best_score < RERANK_SCORE_THRESHOLD:
            print(f"  [rerank] Best chunk score {best_score:.2f} below threshold — no relevant context")
            final = []
        else:
            final = top
            print(f"  [rerank] Best chunk score: {best_score:.2f} — using PDF context")

        sources = {}
        for c in final:
            sources[c["filename"]] = sources.get(c["filename"], 0) + 1
        print(f"\n[Retrieval] {len(final)} chunks after reranking from "
              f"{len(sources)} PDF(s):")
        for fname, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {count} chunk(s) ← {fname}")
        if len(final) < FINAL_CHUNK_LIMIT:
            print(f"  [warn] Only {len(final)}/{FINAL_CHUNK_LIMIT} chunks retrieved — "
                  f"reranker had {len(candidates)} candidates")

        return final

    except Exception as e:
        print(f"[Qdrant search error] {e}")
        return []

# ─── Groq LLM ─────────────────────────────────────────────────────────────────
def call_groq(
    question: str,
    context_chunks: List[dict],
    history: list,
    use_general_knowledge: bool = False,
) -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY is not set. Please configure your API key."

    client = Groq(api_key=GROQ_API_KEY)

    by_file: dict = {}
    for c in context_chunks:
        by_file.setdefault(c["filename"], []).append(c)

    num_sources = len(by_file)

    if num_sources > 1:
        comparison_rules = (
            "3. Always cite which PDF(s) your answer draws from, "
            "e.g. (Source: filename.pdf, Page X).\n"
            "4. Explicitly compare and contrast how different PDFs treat the same concept.\n"
            "5. If a concept appears in multiple PDFs, explain each source's perspective."
        )
    else:
        comparison_rules = (
            "3. Always cite the PDF and page number your answer draws from.\n"
            "4. Focus on answering precisely and completely from the available context."
        )

    # When no relevant chunks found, instruct LLM to answer from general knowledge
    if use_general_knowledge:
        system = """You are an expert research assistant.

The user's question was not found in any of the uploaded PDF documents.
Answer the question using your general knowledge.

RULES:
1. Start your answer with: "This information was not found in the uploaded PDFs. Here is what I know:"
2. Give a clear, accurate, helpful answer from general knowledge.
3. Be honest if you are uncertain about something.
4. Do not fabricate sources or citations."""

        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

    else:
        system = f"""You are an expert research assistant with access to {num_sources} PDF document(s).

Your job is to give accurate, complete answers grounded strictly in the provided context.

RULES:
1. Read ALL source sections carefully before forming your answer.
2. Base your answer ONLY on the retrieved passages — do not add outside knowledge unless the answer is completely absent from the PDFs.
{comparison_rules}
6. If the answer is not in any PDF, say: "This information was not found in the uploaded PDFs. Based on general knowledge: ..." then answer from general knowledge.
7. Never fabricate page numbers, quotes, or facts.
8. Be specific and direct — answer the question that was actually asked."""

        context_str = ""
        for filename, chunks in by_file.items():
            context_str += f"\n{'='*50}\n"
            context_str += f"DOCUMENT: {filename}\n"
            context_str += f"{'='*50}\n"
            for c in chunks:
                context_str += (
                    f"\n[Page {c['page_number']} | "
                    f"Vector score: {c['score']} | "
                    f"Rerank score: {round(c.get('rerank_score', 0), 2)}]\n"
                    f"{c['text']}\n"
                    f"{'-'*30}\n"
                )

        source_note = f"{num_sources} PDF(s): {', '.join(by_file.keys())}"
        user_content = (
            f"Retrieved passages from {source_note}:\n"
            f"{context_str}\n"
            f"Question: {question}\n\n"
            f"Answer based strictly on the passages above."
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=2048,
        temperature=0.2,
    )
    return response.choices[0].message.content

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path("static/index.html")
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else "<h1>PDF RAG Agent</h1>"


def _run_ingest_job(job_id: str, session_id: str, file_data: list):
    """
    Runs in a background thread. Ingests all PDFs and updates job_store
    when done. The frontend polls /status/{job_id} to check progress.
    """
    results = []
    try:
        for filename, save_path in file_data:
            try:
                info = ingest_pdf(save_path, filename, session_id)
                results.append({
                    "filename": filename,
                    "pdf_id": info["pdf_id"],
                    "chunks": info["chunks"],
                    "status": "ok",
                })
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ingest error] {filename}\n{tb}")
                results.append({
                    "filename": filename,
                    "status": "error",
                    "detail": str(e),
                })
        job_store[job_id] = {
            "status": "done",
            "session_id": session_id,
            "files": results,
        }
        print(f"[job {job_id}] Indexing complete — {len(results)} file(s)")
    except Exception as e:
        job_store[job_id] = {
            "status": "error",
            "session_id": session_id,
            "files": results,
            "error": str(e),
        }
        print(f"[job {job_id}] Indexing failed: {e}")


@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    session_id: str = Form(default=""),
):
    if not session_id:
        session_id = str(uuid.uuid4())
    if len(files) > MAX_PDFS:
        raise HTTPException(400, f"Max {MAX_PDFS} PDFs allowed.")

    # Read file bytes immediately (UploadFile is not thread-safe)
    file_data = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"'{f.filename}' is not a PDF.")
        raw = await f.read()
        save_path = UPLOAD_DIR / f"{session_id}_{f.filename}"
        save_path.write_bytes(raw)
        file_data.append((f.filename, save_path))

    # Create a job and return immediately — no waiting for indexing
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "indexing",
        "session_id": session_id,
        "files": [],
    }

    # Kick off indexing in a background thread so the HTTP response
    # is sent instantly and the browser doesn't cancel the request
    thread = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, session_id, file_data),
        daemon=True,
    )
    thread.start()

    print(f"[job {job_id}] Indexing started in background for {len(file_data)} file(s)")

    # Return immediately with job_id — frontend polls /status/{job_id}
    return {
        "job_id": job_id,
        "session_id": session_id,
        "status": "indexing",
        "message": f"Indexing {len(file_data)} PDF(s) in background. Poll /status/{job_id} for progress.",
    }


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Frontend polls this endpoint every 3 seconds after uploading.
    Returns status: indexing | done | error
    When status is done, returns the same files array as the old /upload did.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return job


@app.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form(...),
    pdf_id: Optional[str] = Form(default=None),
):
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if not session_id.strip():
        raise HTTPException(400, "session_id is required.")

    # ── Greeting detection ────────────────────────────────────────────────────
    if GREETING_PATTERNS.match(question.strip()):
        add_to_history(session_id, "user", question)
        add_to_history(session_id, "assistant", GREETING_RESPONSE)
        return {
            "answer": GREETING_RESPONSE,
            "source": "greeting",
            "sources_used": [],
            "references": [],
            "context_used": False,
            "session_id": session_id,
        }

    history = get_trimmed_history(session_id)
    chunks  = search_qdrant(question, session_id, pdf_id=pdf_id)

    # ── Determine if we have useful context ──────────────────────────────────
    # If reranker returned no chunks above threshold, fall back to general knowledge
    use_general_knowledge = len(chunks) == 0
    source = "pdf" if chunks else "general_knowledge"
    sources_used = list({c["filename"] for c in chunks})

    try:
        answer = call_groq(
            question,
            chunks,
            history,
            use_general_knowledge=use_general_knowledge,
        )

        if chunks:
            refs = []
            for c in chunks[:6]:
                refs.append(
                    f"[{c['filename']} | Page {c['page_number']} | "
                    f"Vector: {c['score']} | Rerank: {round(c.get('rerank_score', 0), 2)}]"
                )
            answer += "\n\n─── Sources Retrieved ───\n" + "\n".join(refs)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Groq error]\n{tb}")
        return JSONResponse(status_code=200, content={
            "answer": f"❌ Groq API error: {str(e)}",
            "source": "error",
            "sources_used": [],
            "context_used": False,
            "session_id": session_id,
        })

    add_to_history(session_id, "user", question)
    add_to_history(session_id, "assistant", answer)

    return {
        "answer": answer,
        "source": source,
        "sources_used": sources_used,
        "references": [
            {
                "filename": c["filename"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
                "rerank_score": round(c.get("rerank_score", 0), 2),
                "text": c["text"],
            }
            for c in chunks[:6]
        ],
        "context_used": bool(chunks),
        "session_id": session_id,
    } 


@app.get("/history/{session_id}")
async def get_conv_history(session_id: str):
    return {"session_id": session_id, "history": get_history(session_id)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "vector_dim": VECTOR_DIM,
        "chunking": "semantic",
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "score_threshold": SCORE_THRESHOLD,
        "rerank_score_threshold": RERANK_SCORE_THRESHOLD,
        "max_chunks_per_file": MAX_CHUNKS_PER_FILE,
        "final_chunk_limit": FINAL_CHUNK_LIMIT,
        "groq_key_set": bool(GROQ_API_KEY),
        "groq_reachable": True,
        "qdrant_ok": True,
        "qdrant_mode": "cloud" if QDRANT_URL else "in-memory",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)