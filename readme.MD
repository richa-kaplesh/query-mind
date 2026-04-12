There are four main libraries:
PyMuPDF (fitz) — what we're using. Fastest, most accurate, gives you page-level control, handles complex layouts, tables, multi-column text reasonably well. Industry default for production.
pdfplumber — better than fitz for tables specifically. If your PDF has structured tables with rows and columns, pdfplumber extracts them cleanly. Fitz would just give you jumbled text. Weakness — slower, worse on complex layouts.
PyPDF2 — older, simpler, worse at everything. No real reason to use it today.
Amazon Textract / Google Document AI — cloud OCR services. You send the PDF to their API, they return structured text. Handles scanned PDFs (images of documents) which none of the above can handle. Fitz returns empty string on scanned pages — Textract reads them. Expensive, requires internet.
Why fitz for QueryMind:
Our PDFs are research papers and text-heavy documents. Not scanned, not mostly tables. Fitz is the right tool. If we were building a financial document analyzer with lots of tables, we'd use pdfplumber or combine both.
Great questions. All of them. This is exactly the right time to ask.

Let me go one by one.

---

## PDF Extraction — why fitz, what else exists

There are four main libraries:

**PyMuPDF (fitz)** — what we're using. Fastest, most accurate, gives you page-level control, handles complex layouts, tables, multi-column text reasonably well. Industry default for production.

**pdfplumber** — better than fitz for tables specifically. If your PDF has structured tables with rows and columns, pdfplumber extracts them cleanly. Fitz would just give you jumbled text. Weakness — slower, worse on complex layouts.

**PyPDF2** — older, simpler, worse at everything. No real reason to use it today.

**Amazon Textract / Google Document AI** — cloud OCR services. You send the PDF to their API, they return structured text. Handles scanned PDFs (images of documents) which none of the above can handle. Fitz returns empty string on scanned pages — Textract reads them. Expensive, requires internet.

**Why fitz for QueryMind:**
Our PDFs are research papers and text-heavy documents. Not scanned, not mostly tables. Fitz is the right tool. If we were building a financial document analyzer with lots of tables, we'd use pdfplumber or combine both.

---

## Chunking — why character splitting, what else exists

You're right that character splitting is the most basic approach. There are several others:

**Character/token splitting** — what we're doing. Dumb but fast. Doesn't understand sentence boundaries at all. Can split mid-sentence.

**Sentence splitting** — use nltk or spacy to detect sentence boundaries, never split mid-sentence. Cleaner chunks, better context preservation. Slightly slower.

**Recursive character splitting** — tries to split on paragraph breaks first, then sentences, then words, then characters as a last resort. This is what LangChain's RecursiveCharacterTextSplitter does. Smarter than pure character splitting.

**Semantic chunking** — embed every sentence, then group sentences that are semantically similar into one chunk. Chunks represent complete thoughts not fixed sizes. Most expensive but best quality.

**Document structure aware** — use headers, sections, bullet points in the document itself as natural boundaries. Works well for structured documents like reports and papers.

**Why character splitting for now:**
We're starting simple intentionally. We will upgrade this later — it's one of the improvements that directly affects retrieval quality and is easy to demonstrate in an eval. Swapping chunker strategy and showing RAGAS score improvement is a great interview talking point.

---

## Mathematical data — this is a genuinely hard problem

You've identified one of the real unsolved challenges in RAG.

Math in PDFs is stored as symbols, sometimes as images, sometimes as LaTeX. Fitz extracts it as broken unicode — `∫f(x)dx` might come out as garbage characters or nothing at all.

For math-heavy documents there are specialized approaches:

**MathPix** — converts math images and PDFs into LaTeX. Paid API. Accurate.

**Nougat (Meta)** — open source model specifically trained to read academic papers with math and output clean markdown with proper LaTeX. This is what you'd actually use for a research paper RAG that handles equations.

**LaTeX chunking** — if you have LaTeX source files, parse them structurally — theorems, proofs, definitions as natural chunk boundaries.

For math Q&A specifically — character chunking is terrible because a question and its answer might be split across chunks. You'd want to detect Q&A pairs as atomic units and never split them. This is a document-structure-aware chunking problem.

QueryMind right now would struggle with math-heavy PDFs. That's an honest limitation worth knowing.

---

## Why sentence-transformers, what else exists

**sentence-transformers (local)** — what we're using. Free, runs on your machine, good quality, 384 dimensions for MiniLM. No API calls, no cost, works offline.

**OpenAI text-embedding-3-small** — better quality than MiniLM, 1536 dimensions, costs money per token, requires internet. Standard in production systems.

**Cohere Embed** — competitive with OpenAI, slightly cheaper, also has a reranking API which pairs well.

**BGE embeddings (Beijing Academy of AI)** — open source, runs locally, actually beats OpenAI on some benchmarks. `BAAI/bge-large-en-v1.5` is what serious open source RAG systems use.

**Instructor embeddings** — you can give the model instructions like "represent this scientific paper for retrieval" — changes the embedding based on task. Interesting but experimental.

**Why sentence-transformers for QueryMind:**
Free, local, good enough quality for our purposes. In a production system you'd likely use OpenAI or BGE. The architecture doesn't change — you swap the model name, everything else stays identical. That's why we designed Embedder with `model_name` as a parameter.

---

## Why batch size 32

The model doesn't process one sentence at a time internally — it processes a matrix. Batch size is how many sentences you put in that matrix at once.

Batch size 1 — you make 1000 separate forward passes for 1000 chunks. Slow.

Batch size 32 — you make 32 forward passes. Each pass handles 32 chunks simultaneously on the GPU/CPU. Much faster.

Why not batch size 1000 then? Memory. Larger batches consume more RAM. 32 is a safe default that works on most machines without running out of memory. On a GPU with lots of VRAM you'd increase it. On a weak machine you'd decrease it.

It's a hardware tradeoff, not a model quality tradeoff. Batch size doesn't affect the embeddings themselves, only the speed of generating them.

---

Now you actually understand every decision in those three files.

Ready for `retriever.py`?


Let me think about this properly before we write a single line.

---

## QueryMind — What is it actually for?

Imagine you're a researcher. You have 15 PDF papers on transformer architectures. You want to ask:

*"What do these papers collectively say about attention mechanisms in low-resource settings?"*

A normal search engine finds keywords. ChatGPT doesn't have your papers. Google Scholar finds papers but doesn't answer questions.

QueryMind does this — you upload your own private documents, ask questions across all of them simultaneously, and get answers that tell you exactly which paper, which page, which section the answer came from.

**Real usecases:**
- Research student analyzing 20 papers for literature review
- Lawyer querying 50 case documents simultaneously
- Doctor querying patient history PDFs
- Company querying internal knowledge base

**The key differentiator over basic RAG:** QueryMind doesn't just retrieve — it reasons. If the first retrieval isn't good enough, it tries again differently. That's the agentic layer.

---

## System Design First — Before Any Code

This is what separates senior engineers from juniors. Before writing retriever.py, understand where it sits in the full system.

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
                   ▼
┌─────────────────────────────────────────────────────┐
│               INGESTION PIPELINE                    │
│   PDFExtractor → Chunker → Embedder → IndexStore    │
│                                                     │
│   Runs once per document upload                     │
│   Stores chunks + vectors + BM25 index              │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│               QUERY PIPELINE                        │
│                                                     │
│   Query → Decomposer → Retriever → Reranker         │
│                             ↓                       │
│                         Generator                   │
│                             ↓                       │
│                    Self-Evaluator                   │
│                             ↓                       │
│              Good enough? → Return answer           │
│              Not good enough? → Re-retrieve         │
└─────────────────────────────────────────────────────┘
```

Two completely separate pipelines. This is important.

**Ingestion** runs when you upload a document. Heavy, slow, runs once.

**Query** runs when you ask a question. Must be fast, runs every time.

---

## System Design Concepts Baked Into QueryMind

**1. Separation of concerns**
Ingestion and query are completely decoupled. You can re-index documents without touching query logic. You can upgrade the retriever without re-ingesting documents.

**2. Async architecture**
PDF ingestion is slow — could take 30 seconds for a large document. You don't want the user's browser hanging. FastAPI handles this with background tasks — upload returns immediately, ingestion runs in background, frontend polls for status.

**3. Storage layers — this is classic system design**
```
Hot storage  → FAISS index (in memory, fast vector search)
Warm storage → BM25 index (in memory, fast keyword search)
Cold storage → Raw chunks + metadata (JSON/SQLite on disk)
```
Different data lives at different speeds of access. This is the same pattern as Redis + PostgreSQL in web systems.

**4. Stateless API**
Each query request carries everything the server needs. No session state. This means you can run multiple instances of the FastAPI server behind a load balancer later — horizontal scaling.

**5. Pipeline pattern**
Each component has one job, one input type, one output type. Chunker doesn't know embedder exists. Embedder doesn't know retriever exists. This is why we can swap any component without breaking others.

**6. Agentic loop — this is new**
```
Query comes in
    ↓
Retrieve top K chunks
    ↓
Is context sufficient? (LLM decides)
    ├── Yes → Generate answer
    └── No  → Reformulate query → Retrieve again → Try once more
```
This is a feedback loop inside your system. Not a straight pipeline anymore — a graph. This is what makes QueryMind an agent, not just a retriever.

---

## Now retriever.py — concept first

Here's the core problem.

You have a user query. You have 10,000 chunks indexed. You need the 5 most relevant chunks. How?

**Approach 1 — Dense retrieval (vector search)**

Embed the query → get a 384-dim vector → find chunks whose vectors are closest in cosine similarity → return top K.

Good at: semantic similarity, synonyms, paraphrasing
Bad at: exact keyword matches, rare technical terms, proper nouns

If user asks *"what did Einstein say about photoelectric effect"* — vector search might miss chunks that literally contain "Einstein" and "photoelectric" if the embedding space doesn't cluster them perfectly.

**Approach 2 — Sparse retrieval (BM25)**

Classic keyword search. Score each chunk based on term frequency and document frequency. The same algorithm Google was built on.

Good at: exact keyword matches, technical terms, names, numbers
Bad at: synonyms, semantic understanding

*"photoelectric effect 1905"* — BM25 finds it instantly. But *"Einstein's work on light quanta"* — might miss it if "light quanta" and "photoelectric" aren't in the same chunk.

**Why neither alone is enough — this is the interview answer:**

Dense search understands meaning but misses keywords.
Sparse search finds keywords but misses meaning.
They fail on opposite cases.

**Hybrid search combines both scores.** You get semantic understanding AND keyword precision. This is why every serious production RAG system uses hybrid search.

The formula:
```
final_score = α × dense_score + (1-α) × sparse_score
```

Alpha is a tunable parameter. Default 0.5 — equal weight. You can tune it based on your document type. Technical documents with lots of jargon → increase sparse weight. Conceptual documents → increase dense weight.

---

## Now the code

```python
# backend/core/retriever.py

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class HybridRetriever:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
        self.dimension = 384
```

`alpha` controls the dense/sparse balance. We store it at init time so you can tune it without rewriting logic.

`chunks` — we keep the full chunk list in memory. This is our warm storage. When FAISS returns index positions, we look up the actual chunk text here.

`dimension = 384` — matches MiniLM output. If you swap the embedding model, change this one number.

```python
    def index_chunks(self, chunks: List[dict]) -> None:
        self.chunks = chunks
```

Store chunks for lookup later.

```python
        # Build BM25 index
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
```

BM25 needs tokenized text — list of words, not raw string. We lowercase to make matching case-insensitive. `BM25Okapi` is the most common BM25 variant — Okapi refers to the specific probabilistic formula used.

```python
        # Build FAISS index
        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
        
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
```

`IndexFlatIP` — IP means Inner Product. When vectors are L2 normalized, inner product equals cosine similarity. So we normalize first, then use IP index.

Why not `IndexFlatL2` (euclidean distance)? Because cosine similarity is what sentence-transformers were trained to optimize. Using L2 distance on those vectors gives worse results.

`faiss.normalize_L2` modifies the array in place — normalizes every vector to unit length.

```python
    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 20) -> List[dict]:
```

Why top_k=20 and not 5? Because reranker comes after this. We retrieve 20 candidates generously, then reranker picks the best 5. Retriever casts wide net, reranker sharpens focus.

```python
        # Dense retrieval
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, top_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
```

FAISS returns scores and indices as 2D arrays (batch dimension first). We take `[0]` because we're querying one thing at a time.

```python
        # Sparse retrieval
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
```

BM25 returns a score for every single chunk in your corpus. Not just top K — all of them. We'll combine with dense scores and then take top K after.

```python
        # Normalize both score arrays to [0,1]
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s == 0:
                return scores
            return (scores - min_s) / (max_s - min_s)

        dense_all = np.zeros(len(self.chunks))
        dense_all[dense_indices] = normalize(dense_scores)
        
        sparse_all = normalize(sparse_scores)
```

Why normalize? Dense scores and sparse scores live on completely different scales. FAISS returns values around 0.7–0.99. BM25 returns values like 0–15. You cannot add them directly — BM25 would dominate just because its numbers are bigger. Normalization brings both to the same 0–1 range so alpha actually means something.

```python
        # Combine
        hybrid_scores = self.alpha * dense_all + (1 - self.alpha) * sparse_all
```

This is the entire hybrid search formula. One line. Everything before it was preparation for this.

```python
        # Get top_k indices sorted by hybrid score
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["retrieval_score"] = float(hybrid_scores[idx])
            results.append(chunk)

        return results
```

`argsort` returns indices that would sort the array. `[::-1]` reverses to descending. `[:top_k]` takes the best K.

We attach `retrieval_score` to each chunk — reranker and evaluator use this later.

---

## Full file

```python
# backend/core/retriever.py

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List

class HybridRetriever:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
        self.dimension = 384

    def index_chunks(self, chunks: List[dict]) -> None:
        self.chunks = chunks

        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 20) -> List[dict]:
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, top_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)

        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s == 0:
                return scores
            return (scores - min_s) / (max_s - min_s)

        dense_all = np.zeros(len(self.chunks))
        dense_all[dense_indices] = normalize(dense_scores)
        sparse_all = normalize(sparse_scores)

        hybrid_scores = self.alpha * dense_all + (1 - self.alpha) * sparse_all

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["retrieval_score"] = float(hybrid_scores[idx])
            results.append(chunk)

        return results
```

---

## Pipeline so far

```
PDF → extractor → chunker → embedder → retriever → reranker → generator
  ✅       ✅          ✅         ✅           ✅            ← next
```

---

**Interview question before reranker:**

We retrieve top 20, not top 5. The reranker then picks the best 5.

Why is a two-stage approach better than just retrieving top 5 directly with higher quality? What does the reranker know that FAISS doesn't?

Think about it. Then we do reranker.py.

Yes. You're on the right track. Let me sharpen it.

---

FAISS and BM25 look at query and chunk **separately.**

FAISS says — is this chunk's vector close to query vector?
BM25 says — does this chunk contain query keywords?

Neither of them looks at **query and chunk together as a pair.**

---

Reranker does exactly that.

It takes query + chunk as one combined input and asks — **does this chunk actually answer this specific question?**

---

Concrete example:

Query: **"what are side effects of aspirin?"**

Chunk: **"aspirin is widely used for pain relief and has been studied extensively since 1899"**

FAISS — high score. Vector is close, both about aspirin.
BM25 — high score. Contains keyword "aspirin".

But does this chunk answer "what are the side effects?" — **No.**

Reranker reads both together and catches this. Scores it low. Drops it.

---

This is why two stages exist.

Retriever casts a wide net — fast, gets 20 candidates.
Reranker reads carefully — slow, picks best 5.

Fast rough search first. Slow precise scoring second.

---

Now ready for reranker.py?