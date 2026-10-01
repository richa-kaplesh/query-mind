# QueryMind

An agentic RAG system for document Q&A. Upload a PDF or a CSV and ask questions in plain language. Both file types go through one API, with a different retrieval/computation strategy for each, and every answer is checked by a reviewer before it reaches the user.

**Frontend:** https://query-mind-cyan.vercel.app/#/
**Backend:** https://query-mind.onrender.com/
**Debug dashboard:** `/#/logs` on the frontend

---

## What it does

- **PDF path:** heading-aware extraction (OCR fallback for scanned pages) → line-boundary chunking → hybrid retrieval (BM25 + FAISS) → Jina reranking → an agent that decides *when* and *with what query* to search the document, then answers with inline citations (`[source, p.N — Heading]`).
- **CSV path:** parses the file into a structured schema → an agent writes pandas code → the code runs in a validated, resource-limited sandbox → the answer is synthesised from the real computed result, not a guess.
- **Review before answering (both paths):** a second LLM pass independently checks the draft. If it fails, the draft is revised with the reviewer's feedback, up to a fixed limit.

## Architecture

**PDF pipeline**
`PDFExtractor → TextChunker → Embedder (Jina v3) → Indexer (FAISS + BM25)`
At query time: `Agent loop ⇄ RAGTool (search_documents) → HybridRetriever → Reranker (Jina) → draft → RAGReviewer → answer`

**CSV pipeline**
`CSVExtractor → CSVSchema`
At query time: `Agent loop ⇄ PandasSandboxTool → CodeValidator → persistent pandas worker → draft → Reviewer (DataInspectTool) → answer`

Both extractors return a discriminated union (`PDFExtractionResult` / `CSVExtractionResult`) so the API can branch on document type without forcing CSV data into a shape built for PDFs.

All LLM calls go through a separate **LLM Gateway** service (see below), not directly to a provider.

### Agent loop

`Generator` runs a think → act → observe loop (`agent_max_iterations = 6`). The model may call its tool several times per question: refining a search query for PDFs, or retrying after a sandbox error / empty result for CSVs. Tool output is truncated to `agent_tool_result_max_chars` (4000). If the iteration budget runs out, a forced final-answer call produces a response from what was gathered.

### PDF retrieval: `search_documents`

Retrieval is exposed to the model as a tool rather than forced before generation, so a multi-part question can trigger several differently-worded searches.

1. Hybrid retrieval: top 30 candidates, dense (Jina `jina-embeddings-v3`, 1024-d, FAISS inner product) and sparse (BM25) scores min-max blended with `alpha = 0.5`.
2. Jina reranker (`jina-reranker-v2-base-multilingual`): top 15, **rank-based, no absolute score cutoff**.
3. Passages are added in rank order until a ~3000-token budget is reached, each prefixed with its source, page and section heading.
4. If the returned passages are the *entire* document, the tool says so, so the model stops searching and answers (or reports the information is absent).

### CSV sandbox

The model writes arbitrary pandas code against a pre-loaded DataFrame `df` and assigns its answer to `result`. Three layers contain it:

| Layer | What it enforces |
|---|---|
| **Static validation** (`CodeValidator`, before execution) | Attribute **allowlist** (only listed DataFrame/Series/`pd` methods); no imports, lambdas or function definitions; blocks `eval`/`exec`/`open`/`getattr`/`query` etc.; no dunder access; max 3000 characters and 400 AST nodes |
| **Runtime proxies** (`SafeModuleProxy`, `SafeDataFrameProxy`) | A second, independent allowlist check at attribute-access time for `pd` and `df`; restricted builtins; each query runs on a copy of the DataFrame |
| **Process isolation** | Dedicated worker process (`spawn`); 15 s per-query timeout (worker is killed and respawned on timeout); on Linux, `RLIMIT_AS` 500 MB, `RLIMIT_CPU` 10 s, `RLIMIT_NPROC`; result truncated to 5000 characters |

**Persistent worker:** pandas is imported once when the FastAPI app boots (lifespan hook), and the dataset is loaded once per upload instead of per query. The worker is self-healing: if it dies, it is respawned and the current file is reloaded.

### Reviewer (generate → review → revise)

- **CSV `Reviewer`:** an independent fact-check using `DataInspectTool`, a fixed menu of read-only lookups (`shape`, `columns`, `dtypes`, `describe`, `value_counts`, `null_count`, `unique_count`, `head`). It deliberately cannot re-run the generator's code, so the check takes a different route to the same fact. It spot-checks at most two high-risk claims.
- **`RAGReviewer`:** a faithfulness check of the draft against the passages that were actually retrieved. Catches invented claims, wrong citations and prior-knowledge leakage.
- On a `fail` verdict, the reviewer's feedback is appended to the same conversation and the agent revises (`reviewer_max_revisions = 2`). If a PDF answer still fails after revisions, it is returned with an "could not be fully verified" note.
- The reviewer **fails open**: on an LLM error or when no valid verdict is produced, the draft passes.

### LLM Gateway

LLM calls go through `gateway_client` to a separate, independently deployed routing service (`gateway_url`). Per the recorded eval configuration, the gateway provides sticky per-conversation provider routing, token-bucket rate limiting (Groq / Gemini), retry with backoff before provider fallback, and a semantic response cache that is disabled for any tool-calling request.

### Observability

- **Traces:** every request produces a trace of its steps (LLM calls, tool calls, review calls and verdicts, final answer, timings). Held in memory, last 200 retained.
- **Token tracking:** every call is logged by purpose (agent, review, revision, forced final answer) to `backend/logs/token_log.json`. Gateway-routed calls use estimated counts, flagged as such.
- **Dashboard:** the frontend `/logs` page shows live traces (SSE), eval run history, and token usage.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload file(s); clears prior session state, then ingests in the background |
| `GET` | `/documents/status` | Ingestion status per file (`processing` / `ready` / `failed` + error code) |
| `POST` | `/query/stream` | Ask a question about the active document (Server-Sent Events: `tool` and `token` events, then `[DONE]`) |
| `DELETE` | `/documents/{filename}` | Remove a document from the session |
| `POST` | `/reset` | Clear the session and unload the CSV from the worker |
| `GET` | `/dashboard/evals` | Eval run history, newest first |
| `GET` | `/dashboard/traces`, `/dashboard/traces/{id}` | Request traces |
| `GET` | `/dashboard/stream` | Live trace updates (SSE) |
| `GET` | `/dashboard/tokens` | Token usage log |

Ingestion failures return typed error codes: `password_protected`, `corrupt_file`, `no_extractable_text`, `unknown`.

## Evaluation

Both pipelines are checked against golden datasets with known-correct answers, with a labeled, append-only run history for before/after comparison (`backend/eval/eval_history.json`).

**CSV path:** custom harness that traces each answer back to the model's generated code and raw tool output, not just its final text. Scoring is deterministic (numeric / range / word-boundary string match against a value computed from golden pandas code) with an LLM-judge fallback. Each run samples 15 of 45 golden questions across schema, count, stat, filter and groupby categories.

| Run | Tool appropriateness | Answer correctness |
|---|---|---|
| Baseline (post-fixes) | 53% | 20% |
| Post scoring fixes | 73% | 60% |
| Final (direct provider) | 80% | 93% |
| Gateway production pass | 87% → 100%* | 87% |

\* Same answers re-scored after correcting three golden entries that wrongly required computation for schema-derivable questions; no new LLM calls were made.

Early "wrong answer" cases frequently turned out to be correct model output being mis-scored by the eval itself, which is why the harness traces to the raw code and tool output.

**PDF path:** faithfulness, answer relevancy, context precision and context recall on a 15-question set. Best recorded run: faithfulness 100%, answer relevancy 100%, context recall 100%, context precision ~87%.
> These PDF numbers were recorded on 2026-08-29 against the earlier forced-retrieval pipeline. They have not yet been re-run against the agentic `search_documents` path, so treat them as a baseline rather than a measurement of the current system.

Real bugs found through evals: a fixed-size chunker merging a data line into an unrelated chunk (fixed with heading-aware chunking), and a hard reranker score threshold dropping a correct chunk ranked #4 by a 0.0009 margin (fixed with rank-based reranking).

## Design decisions

- **Retrieval as a tool, with a reviewer as the safety net.** An earlier design forced retrieval before generation; the current design lets the model search as many times as needed and uses `RAGReviewer` to catch answers not grounded in what was retrieved.
- **Rank-based reranking, not absolute thresholds.** The same Jina score meant "relevant" for one query and "noise" for another, and a fixed cutoff silently dropped a correct answer.
- **Sandboxed code execution for CSV, not templated queries.** The model writes arbitrary pandas, so it is not limited to a fixed set of operations; the attribute allowlist, runtime proxies and process limits contain it.
- **Independent review, not self-review.** The CSV reviewer cannot execute code and uses a fixed lookup menu, so it cannot repeat the generator's mistake the same way.
- **Persistent worker instead of per-query subprocess.** Removes repeated pandas import and CSV-load overhead.
- **Single active document, by design.** Each `/upload` clears previous state and resets the index. Queries resolve to the first ready file; multi-document Q&A is a deferred extension.
- **Eval-driven development.** Every fix traces to a specific, reproduced failure.

## Known limitations

- **Sandbox is hardened, not hardened-for-hostile-tenants.** The allowlist is name-based, not type-aware (`sum` is allowed on any object), and objects returned from allowed calls (a Series, a GroupBy) are not re-wrapped by the runtime proxies. Resource limits only apply on Linux. Suitable for a demo / self-hosted deployment, not for untrusted multi-tenant use.
- **Reviewer fails open and checks narrowly.** The CSV reviewer spot-checks at most two claims; any reviewer error lets the draft through unverified.
- **Streaming is cosmetic.** The agent loop and review complete first, then the finished answer is emitted word by word. Time-to-first-token includes the full generate → review → revise cycle.
- **In-memory, single-process state.** The document registry, FAISS/BM25 index, traces and conversation ID live in process memory (single worker by design; use a real vector store to scale), so state is lost on restart and not safe to share across concurrent users.
- **Wide CSVs:** only the first 20 columns get full detail in the schema prompt; the rest are summarised by dtype. The model is instructed to verify other columns via the sandbox, but this relies on it doing so.
- **CSV pipeline tested on a small set of datasets.** Quote-stripping and numeric coercion are written generically, but have not been stress-tested across many differently-messy CSVs (mixed types, unusual delimiters).
- **Retrieval score blend is not principled.** Dense scores are normalised across the full corpus (mostly zeros), sparse scores with true min-max.
- **Eval gaps:** compound-answer questions are scored against a single expected value; PDF eval not yet re-run on the agentic path (above); CSV runs sample 15 of 45 questions.

## Future scope

- Re-run the PDF eval on the agentic path and add agentic-specific metrics (searches per question, revision rate).
- Measure the reviewer's effect: eval runs with and without it, tracking false-fail and missed-error rates.
- Real token streaming.
- Multi-document support and a persistent vector store.
- Stress-testing against more CSV shapes and PDF layouts.
- Compound-answer-aware scoring in the eval harness.

## Tech stack

- **Backend:** FastAPI, Docker (multi-stage), deployed on Render
- **LLM:** `openai/gpt-oss-20b` via a separate LLM Gateway (Groq / Gemini routing)
- **Embeddings / reranking:** Jina (`jina-embeddings-v3`, `jina-reranker-v2-base-multilingual`)
- **Retrieval:** FAISS (dense), BM25 (sparse)
- **PDF extraction:** PyMuPDF, Tesseract OCR fallback
- **CSV execution:** pandas, AST-allowlist validator, runtime proxies, `multiprocessing` worker with `resource` limits
- **Frontend:** React, TypeScript, Vite, deployed on Vercel
- **Eval:** custom harness per pipeline with per-question tracing

## Running locally

**Backend** (Python 3.11; Tesseract installed for OCR fallback)
```bash
cd backend
pip install -r requirements.txt
# .env
# GROQ_API_KEY=...
# JINA_API_KEY=...
# GATEWAY_URL=...   (optional; defaults to the hosted gateway)
uvicorn main:app --reload
```
Or with Docker: `docker build -t querymind backend && docker run -p 8000:8000 -e GROQ_API_KEY=... -e JINA_API_KEY=... querymind`

**Frontend**
```bash
cd frontend
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```
