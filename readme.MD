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
| **Process isolation** | Dedicated worker process (`spawn`); 15 s per-query timeout (worker is killed and respawned on timeout); on Linux, `RLIMIT_AS` (the address space the worker inherits from the main app, plus 600 MB of headroom) and `RLIMIT_CPU` 10 s; result truncated to 5000 characters |

**Persistent worker:** pandas is imported once when the FastAPI app boots (lifespan hook), and the dataset is loaded once per upload instead of per query. The worker is self-healing: if it dies, it is respawned and the current file is reloaded.

### Reviewer (generate → review → revise)

- **CSV `Reviewer`:** an independent fact-check using `DataInspectTool`, a fixed menu of read-only lookups (`shape`, `columns`, `dtypes`, `describe`, `value_counts`, `null_count`, `unique_count`, `head`). It deliberately cannot re-run the generator's code, so the check takes a different route to the same fact. It spot-checks at most two high-risk claims.
- **`RAGReviewer`:** a faithfulness check of the draft against the passages that were actually retrieved. Catches invented claims, wrong citations and prior-knowledge leakage.
- On a `fail` verdict, the reviewer's feedback is appended to the same conversation and the agent revises (`reviewer_max_revisions = 2`). If a PDF answer still fails after revisions, it is returned with an "could not be fully verified" note.
- The reviewer **fails open**: on an LLM error or when no valid verdict is produced, the draft passes.

### LLM Gateway

LLM calls go through `gateway_client` to a separate, independently deployed routing service (`gateway_url`). Per the recorded eval configuration, the gateway provides sticky per-conversation provider routing, token-bucket rate limiting (Groq / Gemini), retry with backoff before provider fallback, and a semantic response cache that is disabled for any tool-calling request.

### Observability

- **Request correlation:** a middleware gives every request an ID (or reuses an incoming `X-Request-ID`) and stores it in a `ContextVar`, so every log line written while handling that request carries it without passing it through function signatures. The ID is sent to the LLM Gateway in the same header and returned in the response headers, so one ID finds a question's logs in both services.
- **Structured logs:** all logs are JSON (timestamp, level, logger, message, `req_id`, plus event fields). Key events: `llm_call` (latency, model, tool calls requested), `tool_call` (duration, result size), `agent_loop_done` (iterations, stop reason), `review_verdict`, and one `request_summary` per question (pipeline, total time, tools used, revisions).
- **Log shipping:** logs go to stdout (Render's log viewer) and, when `LOKI_URL` is set, to Grafana Cloud Loki through a queue-and-batch handler running in a background thread, so a slow or unreachable log backend cannot slow down or break a request. Logs can be filtered by `req_id`, event or level. Grafana Cloud's free plan keeps them for 14 days.
- **Traces:** every request also produces a trace of its steps (LLM calls, tool calls, review calls and verdicts, final answer, timings). Held in memory, last 200 retained.
- **Token tracking:** every call is logged by purpose (agent, review, revision, forced final answer) to `backend/logs/token_log.json`. Gateway-routed calls use estimated counts, flagged as such.
- **Dashboard:** the frontend `/logs` page shows live traces (SSE), eval run history, and token usage.

**Problems the logs surfaced:**
- A production `AttributeError` for `_review_and_revise`: after a manual merge the method had ended up outside the `Generator` class because of its indentation.
- CSV uploads failing on Render with `C error: out of memory` for a 31 MB file. The worker's fixed 500 MB address-space cap was applied after the process had inherited the main app's address space, leaving almost no headroom (about 500 MB already reserved in a local reproduction). The cap is now the inherited size plus 600 MB.
- A 53-second gateway call: the provider SDK's own retry waited 51 seconds on a rate limit, so the gateway's fallback never had a chance to run.

### CI/CD

- **CI (GitHub Actions)** on every push and pull request to `main`: the backend job installs dependencies on Python 3.11 and runs the pytest suite in `backend/tests` (extractors, chunker, sandbox security, safe namespace, token tracker and others); the frontend job runs `npm ci` and the type-check and build on Node 22.
- **Deployment:** the backend deploys to Render (Docker) only after the CI checks pass on `main`; the frontend deploys to Vercel from `main`.

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

Both pipelines are scored against golden datasets, not spot-checked. Every run is appended to `backend/eval/eval_history.json` with a label, the full pipeline config and per-question results, and the history is viewable on the dashboard. The CSV harness records the model's generated pandas code and the raw sandbox output for each question, not just the final text. This is how most of the bugs below were found.

### CSV pipeline

**Setup:** 15 questions sampled from a 45-question golden set (schema, count, statistics, filter, groupby, correlation), all against one dataset: 617 numeric feature columns plus a `class` label. Runs go through the live production path (`generate_stream()` → LLM Gateway → persistent pandas worker), with the reviewer not enabled.

**Metrics:**
- *Tool appropriateness:* whether the model used the sandbox when computation was needed and answered from the schema when it wasn't.
- *Answer correctness:* a deterministic numeric/range/string match against a value computed from golden pandas code (tolerance 0.001), with an LLM-judge fallback (`gpt-oss-120b`) for fuzzy answers.

| Date | Run | Tool appropriateness | Answer correctness |
|---|---|---|---|
| 09-01 | Baseline (post-fixes) | 53% | 20% |
| 09-08 | Post scoring fixes | 73% | 60% |
| 09-10 | Final (standalone script) | 80% | 93% |
| 09-24 | Gateway production pass | 87% | 87% (13/15) |
| 09-24 | Same answers, golden labels corrected* | 100% | 87% (13/15) |

\* Three schema-derivable questions had been labeled as requiring computation; the model answered them correctly from the schema and was penalised. The labels were fixed after seeing the results and the same answers were re-scored, so the 100% is a corrected metric, not an improvement.

The first four rows used a standalone `generate_with_tools()` script that was later found not to be wired into any live route. The gateway pass is the first run against the real serving path, so its 87% is the current number. It is not a regression from 93%.

**What the 20% → 93% climb was:** almost entirely harness and data bugs, not model improvements.
- *Scoring:* the number extractor matched digits inside column names (`f1`, `f617`) before the real answer, missed Unicode dashes as minus signs, and picked up question parameters instead of results. It now anchors on linking words ("is", "equals"). Tolerance had also been set to 0.0001, which implicitly demanded 4+ decimals in prose answers.
- *Data:* the `class` column contained literal quote characters, so `df['class'] == 1` silently matched zero rows. The fix was applied in two independent CSV loaders (extractor and sandbox worker), which had drifted apart, so cleaning moved to one shared utility.
- *Prompt:* schema truncation computed a summary for columns past the first 20 but never appended it, so 597 columns were invisible to the model. "Column doesn't exist" answers were correct inferences from an incomplete prompt.
- *Synthesis:* `gpt-oss-20b` sometimes attempts an unprompted tool call during the no-tools synthesis step, which crashed it. Fixed with a try/except fallback and a tightened instruction.

**Bugs found by the gateway pass:**
- The gateway's semantic cache was caching synthesis calls (they carry no tools, so they looked cacheable). One question's answer leaked into others (33% hit rate on requests that should never be cached). Fixed with an explicit `is_tool_related` flag.
- The pandas worker's queues had no lock, so concurrent callers could consume each other's results. Fixed with a lock around the send/receive exchange.
- A duplicate `route()` definition meant the new retry-with-backoff code never ran across several redeploys. Found by inspecting the pushed repo.

**The two remaining failures** are scoring-design gaps per the run notes, not execution bugs: `schema_06` (correct prose answer, golden value is a raw `.dtypes` repr) and `count_05` (a two-part question, "which class *and* how many", scored against one value).

### PDF pipeline

**Setup:** 15 questions over one document (an HDFC International Funds GIFT Outbound leaflet). An LLM judge (`gpt-oss-120b`) scores each question 0–1 on faithfulness, answer relevancy, context precision and context recall.

| Run | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Baseline, 7 reruns of identical config (08-26 to 08-29) | 0.93 – 1.00 | 1.00 | 0.48 – 0.79 | 0.87 – 1.00 |
| After chunker + reranker fix (08-29) | 1.00 | 1.00 | 0.87 | 1.00 |

**Bug found:** Q13 (a registration-number lookup) failed recall. Two stacked causes: the fixed-size chunker had merged the registration line into an unrelated marketing chunk, and a hardcoded reranker cutoff (`min_score = 0.3`) dropped the right chunk at 0.2991, which ranked #4 of 20. Fixes: heading-aware, line-boundary chunking, and rank-based reranking with no absolute threshold. A duplicate `httpx.post()` call that was double-billing every rerank request was also removed.

### Live end-to-end runner

`backend/eval/run_live_eval.py` sends the golden questions through the deployed API (`/upload`, then `/query/stream`) exactly as the frontend does, so the whole production path is exercised: middleware, tools, reviewer and gateway. Each request carries `X-Request-ID: eval-<run_id>-<question_id>`, so a whole run, or a single question, can be found in the logs of both services. Each run is saved to `backend/eval/live_runs/<run_id>.json` (answer, scores, total time, time to first token), and the runner prints correctness, tool appropriateness, latency percentiles and per-category scores. Scoring is shared with the earlier harness (`scoring.py`).

The CSV set is the 45 hand-written questions plus 72 generated by `make_csv_questions.py`, whose golden values are computed directly from the data with pandas (117 in total). The PDF set is the same 15 questions as above, scored here by an LLM judge on the final answer only, with no retrieval metrics.

**First run (2026-10-05, 10 CSV questions, deployed backend):** 7 of the 10 answers were scored and all 7 were correct. 3 were not scored: one because the scorer did not handle thousands separators (a correct "7,797" did not match 7797), and two because the LLM-judge call failed. Tool appropriateness was 7/10. The run also showed latency: median about 35 s, mean about 68 s and slowest 242 s per question, with time to first token almost equal to total time.

### Limitations of this evaluation

- **The RAGAS PDF scores do not describe the current PDF pipeline.** `run_pdf_eval.py` uses forced retrieval (top-20 → rerank top-5 → single-shot generation), while production now uses the agentic `search_documents` tool and `RAGReviewer`. The live runner does go through the current path, but it has not been run on the PDF set yet and gives no retrieval metrics.
- **The reviewer's effect has not been measured.** The earlier CSV harness ran without it; the live runner includes it because it is part of the production path, but there is no with/without comparison yet, so no measurement of how often it catches real errors or fails good answers.
- **Small and narrow:** 15 questions per pipeline, one PDF, one CSV. The CSV score is sampled from 45 questions, and the PDF result is a single run after the fix, so there is no variance estimate.
- **Judge noise is visible in the data.** Seven runs of an identical pipeline gave context precision from 0.48 to 0.79. The post-fix 0.87 is above that range, but a single run on a changed pipeline is weak evidence.
- **The judge and generator are from the same model family** (`gpt-oss-120b` judging `gpt-oss-20b`), which can bias scoring.
- **Golden labels were adjusted after seeing results** (the three relabeled questions, the 0.0001 → 0.001 tolerance change). Each change was justified in the run notes, but it means the scores are not purely pre-registered.

## Known limitations

- **Sandbox is hardened, not hardened-for-hostile-tenants.** The allowlist is name-based, not type-aware (`sum` is allowed on any object), and objects returned from allowed calls (a Series, a GroupBy) are not re-wrapped by the runtime proxies. Resource limits only apply on Linux. Suitable for a demo / self-hosted deployment, not for untrusted multi-tenant use.
- **Reviewer fails open and checks narrowly.** The CSV reviewer spot-checks at most two claims; any reviewer error lets the draft through unverified.
- **Streaming is cosmetic, and latency is high.** The agent loop and review complete first, then the finished answer is emitted word by word, so time to first token includes the whole generate → review → revise cycle. In the first live run, answers took a median of about 35 s (slowest 242 s) on the free tier, even for questions that used no tool. Per-call and per-tool timings are now logged, and finding where the time goes is the next step.
- **In-memory, single-process state.** The document registry, FAISS/BM25 index, traces and conversation ID live in process memory (single worker by design; use a real vector store to scale), so state is lost on restart and not safe to share across concurrent users.
- **Wide CSVs:** only the first 20 columns get full detail in the schema prompt; the rest are summarised by dtype. The model is instructed to verify other columns via the sandbox, but this relies on it doing so.
- **CSV pipeline tested on a small set of datasets.** Quote-stripping and numeric coercion are written generically, but have not been stress-tested across many differently-messy CSVs (mixed types, unusual delimiters).
- **Retrieval score blend is not principled.** Dense scores are normalised across the full corpus (mostly zeros), sparse scores with true min-max.
- **Eval gaps:** compound-answer questions are scored against a single expected value; PDF eval not yet re-run on the agentic path (above); CSV runs sample 15 of 45 questions.

## Future scope

- Find and reduce the latency, using the per-call and per-tool timings now in the logs.
- Keep a per-question summary (time, tokens, provider, revisions) in a database so history outlives the 14-day log window.
- Support several concurrent users within free-tier limits: per-session state instead of one active document, and a persistent vector store.
- Re-run the PDF eval on the agentic path and add agentic-specific metrics (searches per question, revision rate).
- Measure the reviewer's effect: eval runs with and without it, tracking false-fail and missed-error rates.
- Real token streaming.
- Stress-testing against more CSV shapes and PDF layouts; compound-answer-aware scoring in the eval harness.
- Later: LangGraph orchestration and MCP tool integration.

## Tech stack

- **Backend:** FastAPI, Docker (multi-stage), deployed on Render
- **LLM:** `openai/gpt-oss-20b` via a separate LLM Gateway (Groq / Gemini routing)
- **Embeddings / reranking:** Jina (`jina-embeddings-v3`, `jina-reranker-v2-base-multilingual`)
- **Retrieval:** FAISS (dense), BM25 (sparse)
- **PDF extraction:** PyMuPDF, Tesseract OCR fallback
- **CSV execution:** pandas, AST-allowlist validator, runtime proxies, `multiprocessing` worker with `resource` limits
- **Frontend:** React, TypeScript, Vite, deployed on Vercel
- **Eval:** custom harness per pipeline with per-question tracing, plus a live end-to-end runner
- **Observability:** JSON logging with `contextvars` request IDs, Grafana Cloud (Loki)
- **CI:** GitHub Actions (pytest, frontend build)

## Running locally

**Backend** (Python 3.11; Tesseract installed for OCR fallback)
```bash
cd backend
pip install -r requirements.txt
# .env
# GROQ_API_KEY=...
# JINA_API_KEY=...
# GATEWAY_URL=...   (optional; defaults to the hosted gateway)
# LOKI_URL=... LOKI_USER=... LOKI_TOKEN=...   (optional; ship logs to Grafana Cloud)
uvicorn main:app --reload
```
Or with Docker: `docker build -t querymind backend && docker run -p 8000:8000 -e GROQ_API_KEY=... -e JINA_API_KEY=... querymind`

**Frontend**
```bash
cd frontend
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

**Live eval** (from `backend/`; uses the deployed or local API, not a mock)
```bash
python eval/run_live_eval.py --base-url https://query-mind.onrender.com --mode csv --limit 10
```