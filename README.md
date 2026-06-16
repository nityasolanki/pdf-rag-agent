# 🤖 PDF RAG Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-In--Memory-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3_70B-F55036?style=for-the-badge&logo=groq&logoColor=white)

A production-grade RAG agent that lets you chat with your PDF documents using semantic chunking, cross-encoder reranking, and query expansion — powered by Groq LLaMA 3.3 70B.

</div>

---

## How it works

**Upload** → PDFs are extracted page-by-page, split into semantic chunks (topic-boundary detection), embedded with `multi-qa-MiniLM-L6-cos-v1`, and stored in Qdrant with a unique `pdf_id` per file.

**Ask** → The question is expanded into multiple query variants via Groq, each variant searches Qdrant independently, results are merged and deduplicated, then a cross-encoder reranker rescores all candidates. The top 6 chunks by rerank score are sent to Groq LLaMA 3.3 70B as context.

**Fallback** → If the best reranked chunk scores below the threshold, Groq answers from general knowledge and says so explicitly.

```
Question
  └─► Query expansion (Groq — split + rephrase)
        └─► Qdrant search per variant (bi-encoder, cosine, 100 candidates each)
              └─► Merge + deduplicate candidates
                    └─► Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
                          └─► Top 6 chunks → Groq llama-3.3-70b-versatile → Answer
```

---

## Stack

| | |
|---|---|
| Framework | FastAPI + Uvicorn |
| Vector DB | Qdrant (in-memory) |
| Embedding model | `multi-qa-MiniLM-L6-cos-v1` — 384-dim, optimised for retrieval |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq — `llama-3.3-70b-versatile` |
| PDF parsing | PyPDF |

---

## Key features

- **Semantic chunking** — chunks are split at topic boundaries (cosine similarity between sentences), not at fixed character counts. Produces coherent, self-contained chunks.
- **Query expansion** — compound questions are split into sub-questions; each is rephrased to catch vocabulary mismatches between user language and PDF language.
- **Cross-encoder reranking** — the bi-encoder (Qdrant) retrieves 100 broad candidates fast; the cross-encoder re-reads each candidate alongside the question and scores relevance accurately.
- **Per-file diversity cap** — up to 10 chunks per PDF in the candidate pool, so a large PDF doesn't crowd out smaller ones.
- **Background indexing** — upload returns instantly with a `job_id`; indexing runs in a background thread; frontend polls `/status/{job_id}`.
- **Conversation memory** — last 10 turns are passed to Groq on every request.
- **Greeting detection** — regex-matched greetings return a canned help message without touching Qdrant or Groq.
- **Source citations** — every answer appends the filename, page number, vector score, and rerank score for each retrieved chunk.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Groq API key** — free at [console.groq.com](https://console.groq.com)

```bash
# Mac / Linux
export GROQ_API_KEY=gsk_your_key_here

# Windows CMD
set GROQ_API_KEY=gsk_your_key_here

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
```

**3. Run**
```bash
python main.py
```

Open **http://localhost:8000**

> ⚠️ Models download on first run: embedding model (~90 MB) and reranker (~80 MB). Both are cached locally after that.

> **Note:** Qdrant runs in-memory — all indexed data is lost when the server stops. Re-upload PDFs each session.

---

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/upload` | Upload PDFs — returns `job_id` immediately |
| `GET` | `/status/{job_id}` | Poll indexing progress |
| `POST` | `/ask` | Ask a question |
| `GET` | `/history/{session_id}` | Conversation history |
| `GET` | `/health` | Config and status dump |

### `POST /upload`

Returns immediately. Indexing runs in the background.

```json
{
  "job_id": "abc-123",
  "session_id": "xyz-456",
  "status": "indexing"
}
```

### `GET /status/{job_id}`

Poll every few seconds until `status` is `"done"` or `"error"`.

```json
{
  "status": "done",
  "session_id": "xyz-456",
  "files": [
    { "filename": "report.pdf", "pdf_id": "...", "chunks": 38, "status": "ok" }
  ]
}
```

### `POST /ask`

| Field | Required | Description |
|---|---|---|
| `question` | ✅ | The question |
| `session_id` | ✅ | From `/upload` response |
| `pdf_id` | ❌ | Restrict search to one specific PDF |

```json
{
  "answer": "The findings show...\n\n─── Sources Retrieved ───\n[report.pdf | Page 4 | ...]",
  "source": "pdf",
  "sources_used": ["report.pdf"],
  "references": [
    { "filename": "report.pdf", "page_number": 4, "score": 0.82, "rerank_score": 3.41 }
  ],
  "context_used": true
}
```

`source` is `"pdf"`, `"general_knowledge"`, `"greeting"`, or `"error"`.

---

## Configuration

Key constants at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `SEMANTIC_THRESHOLD` | `0.30` | Cosine similarity below which a new chunk starts. Lower = larger chunks. |
| `MIN_CHUNK_CHARS` | `250` | Minimum characters before a chunk boundary is allowed |
| `MAX_CHUNK_CHARS` | `1500` | Hard cap — chunk is split here regardless of similarity |
| `RETRIEVAL_LIMIT` | `100` | Candidates fetched from Qdrant per query variant |
| `MAX_CHUNKS_PER_FILE` | `10` | Per-PDF diversity cap before reranking |
| `FINAL_CHUNK_LIMIT` | `6` | Chunks sent to LLM after reranking |
| `RERANK_SCORE_THRESHOLD` | `-9.5` | Below this rerank score, fall back to general knowledge |
| `MAX_HISTORY_PAIRS` | `10` | Conversation turns remembered per session |

---

## Project structure

```
pdf-rag-agent/
├── main.py           # All app logic — routes, RAG pipeline, chunking, reranking
├── requirements.txt
├── static/
│   └── index.html    # Chat UI
└── uploads/          # PDFs saved here (auto-created)
```

---

## Troubleshooting

**`No extractable text`** → PDF is scanned/image-based. PyPDF only reads PDFs with a text layer.

**Reranker returns no chunks / falls back to general knowledge** → Lower `RERANK_SCORE_THRESHOLD` (e.g. `-11.0`) to accept more results, or check that the PDF content is relevant to the question.

**Slow first startup** → Embedding and reranker models download once on first run and are cached. Subsequent startups are fast.

**Data gone after restart** → Expected — Qdrant is in-memory. Re-upload your PDFs each session.
