# QueryMind 🔍

> Upload your own PDFs. Ask questions across all of them. Get answers with pinpoint citations — not just "page 4", but the exact passage, highlighted.

---

## What Is This?

QueryMind is a **private document Q&A system** built on a RAG (Retrieval-Augmented Generation) pipeline with an agentic self-evaluation loop.

You upload PDFs. You ask questions. The system finds the most relevant passages across all your documents, reasons about whether it found enough, and returns a cited answer — traceable back to the exact source file, page, and text region.

**Real use cases:**
- Research student synthesizing 20 papers for a literature review
- Lawyer querying 50 case documents simultaneously
- Doctor querying patient history PDFs
- Company querying an internal knowledge base

**What makes it more than basic RAG:** If the first retrieval pass isn't confident enough, the system reformulates the query and tries again. That feedback loop is the "agentic" part.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                     CLIENT                          │
│              React + TypeScript                     │
│         Upload PDFs | Ask Questions                 │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP / REST
                   ▼
┌─────────────────────────────────────────────────────┐
│                   API LAYER                         │
│                FastAPI (async)                      │
│         /upload  /query  /documents                 │
└──────────────────┬──────────────────────────────────┘
                   │
       ┌───────────┴──────────┐
       ▼                      ▼
┌─────────────┐     ┌──────────────────────────────────┐
│  INGESTION  │     │          QUERY PIPELINE           │
│  PIPELINE   │     │                                  │
│             │     │  Query → Decomposer               │
│ PDFExtractor│     │            ↓                     │
│     ↓       │     │        Retriever (Hybrid)         │
│  Chunker    │     │            ↓                     │
│     ↓       │     │         Reranker                 │
│  Embedder   │     │            ↓                     │
│     ↓       │     │         Generator                │
│ IndexStore  │     │            ↓                     │
│             │     │      Self-Evaluator              │
│ (runs once  │     │            ↓                     │
│  on upload) │     │  Good enough? → Return answer    │
└─────────────┘     │  Not good enough? → Re-retrieve  │
                    └──────────────────────────────────┘
```

Two completely separate pipelines:
- **Ingestion** — runs once when you upload a document. Heavy, can be slow, runs in the background.
- **Query** — runs every time you ask a question. Must be fast.

---

## Storage Layers

```
Hot storage   → FAISS index       (in memory — fast vector search)
Warm storage  → BM25 index        (in memory — fast keyword search)
Cold storage  → Raw chunks + metadata  (JSON/SQLite on disk)
```

Different data lives at different speeds of access. Same pattern as Redis + PostgreSQL in traditional web systems.

---

## Pipeline Deep Dive

### 1. PDF Extraction (`extractor.py`)

Uses **PyMuPDF (fitz)** — the industry default for production PDF parsing. Fastest, most accurate, gives page-level control, handles complex layouts and multi-column text reasonably well.

**Why not the alternatives?**

| Library | Best for | Weakness |
|---|---|---|
| PyMuPDF (fitz) ✅ | Text-heavy docs, research papers | Struggles with scanned PDFs |
| pdfplumber | PDFs with structured tables | Slower, worse on complex layouts |
| PyPDF2 | Nothing in particular | Older and strictly worse than fitz |
| Amazon Textract / Google Document AI | Scanned PDFs (images of text) | Expensive, requires internet, overkill |

QueryMind targets research papers and text-heavy documents — not scanned, not table-heavy. Fitz is the right tool. For a financial document analyzer with lots of tables, you'd use pdfplumber or combine both.

**Known limitation:** Math in PDFs is stored as symbols, sometimes as images, sometimes as LaTeX. Fitz extracts it as broken unicode. For math-heavy documents you'd want [Nougat (Meta)](https://github.com/facebookresearch/nougat) or MathPix.

---

### 2. Chunking (`chunker.py`)

Splits extracted text into overlapping fixed-size character windows.

**Why chunks have `start_char` and `end_char` metadata — two reasons people always ask about:**

**Reason 1 — Precise source highlighting in the UI.**
You already have the page number. But that only tells you *which page*. To highlight the exact sentence or passage (like Google Books does), you need to know where on that page the text starts and stops. `start_char` and `end_char` give you that.

**Reason 2 — Deduplication during retrieval.**
Because chunks overlap (chunk_overlap is intentional — preserves context across boundaries), two retrieved chunks may cover the same region of text. `start_char` / `end_char` let you detect overlapping chunks during retrieval and deduplicate them — so you don't feed the LLM the same content twice wrapped in slightly different words.

**Chunking strategies — what exists:**

| Strategy | Description | Quality |
|---|---|---|
| Character splitting ✅ (current) | Dumb but fast. Can split mid-sentence. | Baseline |
| Sentence splitting | Uses nltk/spacy, never splits mid-sentence | Better context |
| Recursive character splitting | Tries paragraph → sentence → word → char | Smarter fallback |
| Semantic chunking | Embeds sentences, groups by meaning | Best, most expensive |
| Structure-aware | Uses headers/sections as natural boundaries | Great for formal docs |

We start simple intentionally. Upgrading the chunker strategy and demonstrating RAGAS score improvement is a concrete eval-driven improvement — a good interview talking point.

**Why `all_chunks.extend(chunks)` not `.append()`:**

```python
# append → adds the list as ONE nested item
all_chunks.append([c1, c2, c3])
# result: [[c1, c2, c3]]  ← broken — embedder receives a list, not a chunk dict

# extend → flattens into the parent list
all_chunks.extend([c1, c2, c3])
# result: [c1, c2, c3]  ← correct — embedder can loop over individual chunks
```

After processing 3 PDFs, `all_chunks` is one flat list of all chunks from all documents. The next step (`embedder.encode(all_chunks)`) expects to loop over individual chunk dicts — nested lists would crash it.

---

### 3. Embedding (`embedder.py`)

Uses **sentence-transformers / MiniLM-L6-v2** — outputs 384-dimensional vectors.

**What an embedding is:** A list of 384 numbers. Each number captures a different aspect of the text's meaning. Similar text → similar numbers → close together in 384-dimensional space.

**Why normalize vectors before adding to FAISS:**

FAISS `IndexFlatIP` computes raw inner product (dot product), NOT cosine similarity:

```
FAISS IndexFlatIP  =  A · B              (just this)
Cosine similarity  =  A · B / (|A| × |B|) (this)
```

Without normalization, a vector 10x larger than another would score 10x higher in FAISS — not because it's more relevant, but just because of its magnitude. After L2 normalization every vector has magnitude 1, so inner product equals cosine similarity automatically. You normalize so that only *direction* matters, not magnitude.

**Why batch_size=32:**
The model processes a matrix, not one sentence at a time. Larger batches = fewer forward passes = faster. But larger batches consume more RAM. 32 is the safe default that works on most hardware. It's a speed/memory tradeoff — batch size doesn't affect the quality of the embeddings themselves.

**Embedding model options:**

| Model | Type | Quality | Cost |
|---|---|---|---|
| sentence-transformers/MiniLM ✅ | Local | Good | Free |
| OpenAI text-embedding-3-small | API | Better | Per token |
| Cohere Embed | API | Competitive | Per token |
| BAAI/bge-large-en-v1.5 | Local | Beats OpenAI on some benchmarks | Free |
| Instructor embeddings | Local | Task-conditioned | Free, experimental |

The architecture is model-agnostic — `model_name` is a parameter. Swapping models doesn't change anything else.

---

### 4. Hybrid Retrieval (`retriever.py`)

**Why neither vector search nor keyword search alone is enough:**

Dense retrieval (FAISS) understands meaning but misses exact keywords. Sparse retrieval (BM25) finds keywords but misses meaning. They fail on opposite cases.

Example: query = *"what did Einstein say about photoelectric effect"*
- Vector search: high score on semantically similar chunks, might miss exact proper nouns
- BM25: finds "Einstein" and "photoelectric" instantly, misses semantic paraphrases

Hybrid search combines both:

```
final_score = α × dense_score + (1 - α) × sparse_score
```

`alpha` is tunable. Default 0.5 — equal weight. Technical documents with lots of jargon → increase sparse weight. Conceptual documents → increase dense weight.

**Why scores are normalized before combining:**
Dense scores (FAISS) live around 0.7–0.99. BM25 scores live around 0–15. You can't add them directly — BM25 would dominate just because its numbers are bigger. Normalizing both to [0,1] is what makes `alpha` actually meaningful.

**Why retrieve top 20, not top 5:**
Because the reranker comes next. Retriever casts a wide net (fast, rough), reranker sharpens focus (slow, precise). If you only retrieved 5, you might miss the best answer entirely.

**`IndexFlatIP` vs `IndexFlatL2`:**
Sentence-transformers were trained to optimize cosine similarity, not Euclidean distance. Using `IndexFlatL2` on those vectors gives worse results. Use `IndexFlatIP` + normalize.

---

### 5. Reranking (`reranker.py`)

**Why a two-stage approach:**

FAISS and BM25 look at query and chunk *separately*. Neither looks at the query and chunk *together as a pair*.

Reranker reads both simultaneously:

```
Query:  "what are side effects of aspirin?"
Chunk:  "aspirin is widely used for pain relief since 1899"
```

FAISS: high score (both about aspirin). BM25: high score (keyword match). But does this chunk actually answer "what are the side effects?" — **No.** The reranker catches this. Scores it low. Drops it.

Retriever casts wide net → fast, gets 20 candidates.  
Reranker reads carefully → slow, picks best 5.

**On `rerank_score` vs `retrieval_score` for citations:**
Use rerank_score. It's more accurate. Retrieval score was rough — reranker corrected it. Always trust the more careful measurement.

**On `enumerate` in the scoring loop:**
```python
for i, chunk in enumerate(chunks):
    chunk["rerank_score"] = float(scores[i])
```
`enumerate` gives you both the index and the item at each iteration. This attaches each reranker score back to its chunk so they travel together through the rest of the pipeline.

**On `sorted` with `lambda`:**
```python
reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
```
`key=lambda x: x["rerank_score"]` tells `sorted` what value to compare — without it, Python wouldn't know how to compare two dicts. `reverse=True` sorts highest to lowest. Returns a new list, doesn't modify the original.

---

### 6. Generation (`generator.py`)

Calls **Groq** (LLM inference) with the reranked chunks formatted as labeled context:

```
[SOURCE 1 - report.pdf, page 4]
AI is transforming industries through...

[SOURCE 2 - paper.pdf, page 12]
The study found that attention mechanisms...
```

`temperature=0.1` keeps output factual and focused — close to 0 means minimal randomness, which is correct for research Q&A.

Returns answer + sources together:

```python
{
  "answer": "According to SOURCE 1, AI is transforming...",
  "sources": [
    { "source": "report.pdf", "page": 4, "text": "...", "rerank_score": 0.97 },
    ...
  ]
}
```

Answer and evidence travel together. This is what enables citations in the UI and lets you verify the LLM's response.

---

### 7. API Layer (`routes.py` + `main.py`)

**FastAPI with `APIRouter`:**
If all endpoints lived in `main.py` it would become one giant messy file. `APIRouter` lets each domain (documents, queries) live in its own file while still registering with the main FastAPI app.

**Why async + background tasks for `/upload`:**
PDF ingestion is slow — could take 30 seconds for a large document. You don't want the browser hanging. FastAPI handles this with background tasks — upload returns immediately, ingestion runs in the background, frontend polls for status.

**The three endpoints:**
- `POST /upload` — accepts PDF, starts background ingestion, returns immediately
- `GET /documents` — returns current status of all documents (used for polling)
- `POST /query` — accepts question, runs full query pipeline, returns answer + sources

**`body.question` — why this syntax:**
```python
class QueryRequest(BaseModel):
    question: str
    conversation_history: list = []
```
FastAPI automatically parses the incoming JSON into a `QueryRequest` object named `body`. So `body.question` is just accessing the `question` field from that parsed JSON.

**Polling — why and how:**
Frontend calls `GET /documents` repeatedly at fixed intervals to check when processing is done:

```js
const interval = setInterval(async () => {
  const data = await fetch("/documents").then(r => r.json())
  if (data.status === "ready") {
    clearInterval(interval)   // stop polling
    setDocumentReady(true)    // update UI
  }
}, 3000)  // every 3 seconds
```

Three alternatives exist: polling (frontend keeps asking), webhooks (backend calls frontend), WebSockets (persistent connection, backend pushes). Polling is simplest and good enough here. WebSockets would be overkill.

---

## Full Pipeline Flow

```
PDF Upload
   ↓
PDFExtractor     → text + page metadata per page
   ↓
Chunker          → overlapping fixed-size chunks with start_char/end_char
   ↓
Embedder         → 384-dim normalized vectors per chunk
   ↓
IndexStore       → FAISS index (dense) + BM25 index (sparse) built
   
--- document ready ---

User Query
   ↓
Embedder         → embed the query
   ↓
HybridRetriever  → top 20 chunks (α × dense + (1-α) × sparse)
   ↓
Reranker         → top 5 chunks (cross-encoder, reads query+chunk as pair)
   ↓
Generator        → LLM call with formatted context, returns answer + sources
   ↓
Self-Evaluator   → is context sufficient?
   ├── Yes → return answer
   └── No  → reformulate query → retrieve again → try once more
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript |
| API | FastAPI (async) |
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers / MiniLM-L6-v2 |
| Vector Search | FAISS (IndexFlatIP) |
| Keyword Search | BM25 (rank_bm25) |
| Reranking | Cross-encoder model |
| LLM Inference | Groq |

---

## Known Limitations

- **Scanned PDFs** — fitz returns empty string on image-based PDFs. Needs Textract or Nougat for those.
- **Math-heavy documents** — LaTeX/equations extract as broken unicode. Use Nougat (Meta) for academic papers with heavy math.
- **Chunking quality** — current character-based splitting can cut mid-sentence. Upgrade path: recursive → sentence → semantic chunking.
- **In-memory indexes** — FAISS and BM25 indexes are rebuilt on each server restart. Production would persist them to disk.

---

## Design Decisions Worth Knowing

**Stateless API** — each request carries everything the server needs. No session state. Means you can run multiple FastAPI instances behind a load balancer later.

**Pipeline pattern** — each component has one job, one input type, one output type. Chunker doesn't know embedder exists. Embedder doesn't know retriever exists. Swap any component without breaking others.

**Why top_k=20 in retriever, not 5** — retriever is fast and rough; reranker is slow and precise. Cast wide net first, sharpen focus second. Retrieving only 5 risks missing the best answer entirely.

**alpha=0.5 default** — equal weight to dense and sparse retrieval. Tune toward sparse for jargon-heavy technical docs, toward dense for conceptual/abstract content.


def run_all_evals():
    print(f"Total chunks indexed: {len(indexer.chunks)}")
    print(f"Sample chunk: {indexer.chunks[0]['text'][:200]}")