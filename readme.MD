# QueryMind

An agentic RAG system for document Q&A. Upload a PDF or a CSV and ask questions in plain language. Both file types go through one API, with a different retrieval/computation strategy for each, and every answer is checked by a reviewer before it reaches the user.

**Frontend:** https://query-mind-cyan.vercel.app/#/
**Backend:** https://query-mind.onrender.com/
**Debug dashboard:** `/#/logs` on the frontend

---

## What it does

- **PDF path:** heading-aware extraction (OCR fallback for pages with no text layer) → line-boundary chunking → hybrid retrieval (BM25 + FAISS) → Jina reranking → an agent that decides *when* and *with what query* to search the document, then answers with inline citations (`[source, p.N — Heading]`).
- **CSV path:** parses the file into a structured schema → an agent writes pandas code → the code runs in a validated, resource-limited sandbox → the answer is synthesised from the real computed result, not a guess.
- **Review before answering (both paths):** a second LLM pass independently checks the draft. If it fails, the draft is revised with the reviewer's feedback, up to a fixed limit. Cheap deterministic checks decide when that LLM review can be skipped and catch obvious non-answers without an LLM call.
- **Knows when not to answer:** questions about data the file does not contain, questions unrelated to the file, false claims inside the question, and instructions injected into the question or the data are handled by explicit rules and measured as separate eval categories.
- **Multi-user:** every browser gets its own session with its own file, schema and index, so users never see each other's data.

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

`Generator` runs a think → act → observe loop (`agent_max_iterations = 4`). The model may call its tool several times per question: refining a search query for PDFs, or retrying after a sandbox error or empty result for CSVs.

- **Call budget:** one user question may make at most `gateway_max_calls_per_request = 14` gateway calls in total (agent loop + reviewer + revision). The caps were lowered from 6 / 3 / 2 after traces showed a median of about 3 calls per question.

- Tool output is truncated to `agent_tool_result_max_chars` (4000) by default. The PDF search tool overrides this with its own limit (18,000 characters) so a normal result is not clipped before its token budget applies.
- **Repeated-call guard:** if the model issues the exact same tool call twice, it gets the earlier result back with a note and the loop stops, instead of retrying blindly.
- If the iteration budget runs out, a forced final-answer call produces a response from what was gathered.
- If an upstream LLM call fails, the fallback answer is returned and review/revision is skipped (a reviewer cannot fix a broken request).

### PDF retrieval: `search_documents`

Retrieval is exposed to the model as a tool rather than forced before generation, so a multi-part question can trigger several differently-worded searches.

1. Hybrid retrieval: top 30 candidates (capped at the number of indexed chunks). Dense search (Jina `jina-embeddings-v3`, 1024-d, FAISS inner product over L2-normalised vectors) and sparse BM25 scores are min-max blended with `alpha = 0.5`.
2. Jina reranker (`jina-reranker-v2-base-multilingual`): top 15, **rank-based, no absolute score cutoff**.
3. Passages are added in rank order until a ~3000-token budget is reached, each prefixed with its source, page and section heading.
4. If the returned passages are the *entire* document, the tool says so, so the model stops searching and answers (or reports the information is absent).

### CSV sandbox

The model writes arbitrary pandas code against a pre-loaded DataFrame `df` and assigns its answer to `result`. Three layers contain it:

| Layer | What it enforces |
|---|---|
| **Static validation** (`CodeValidator`, before execution) | Attribute **allowlist** (only listed DataFrame/Series/`pd` methods); no imports, lambdas or function definitions; blocks `eval`/`exec`/`open`/`getattr`/`query` etc.; no dunder access; max 3000 characters and 400 AST nodes |
| **Runtime proxies** (`SafeModuleProxy`, `SafeDataFrameProxy`) | A second, independent allowlist check at attribute-access time for `pd` and `df`; restricted builtins; each query runs on a copy of the DataFrame |
| **Process isolation** | Dedicated worker process (`spawn`); 15 s per-query timeout (on timeout the worker is killed and respawned on the next query); on Linux, `RLIMIT_AS` (the address space the worker inherits from the main app, plus 600 MB of headroom) and `RLIMIT_CPU` of 10 s (a limit on the worker process's total CPU time, not a per-query budget); result truncated to 5000 characters |

**Persistent worker:** pandas is imported once when the FastAPI app boots (lifespan hook), and the dataset is loaded once per upload instead of per query. The worker is self-healing: if it dies, it is respawned and the current file is reloaded.

### Reviewer (generate → review → revise)

- **CSV `Reviewer`:** an independent fact-check using `DataInspectTool`, a fixed menu of read-only lookups (`shape`, `columns`, `dtypes`, `describe`, `value_counts`, `null_count`, `unique_count`, `head`). It deliberately cannot re-run the generator's code, so the check takes a different route to the same fact. It spot-checks at most two high-risk claims.
- **`RAGReviewer`:** a faithfulness check of the draft against the passages that were actually retrieved. Catches invented claims, wrong citations and prior-knowledge leakage.
- On a `fail` verdict, the reviewer's feedback is appended to the same conversation and the agent revises (`reviewer_max_revisions = 1`). If an answer still fails after the revisions, on **either** pipeline, it is returned with a "could not be fully verified" note.
- `reviewer_max_iterations = 2` bounds the reviewer's own inner loop (tool-calling rounds for the CSV reviewer; "ask again for valid JSON" retries for both).
- Verdicts are `pass`, `fail`, or `unverifiable` (the claim cannot be checked with the read-only lookups, e.g. a conditional count). `unverifiable` passes without a wasted revision, and the trace records it as inconclusive.
- **The user is told when nobody checked the answer.** If the reviewer's LLM call fails (e.g. a gateway 429), or it reaches its iteration cap without a verdict, the draft is sent with the "could not be fully verified" note instead of passing silently. The trace records why (`reviewer_error` or `no_verdict`).
- The reviewer prompt also fails drafts that obey an instruction injected into the question, answer a question unrelated to the dataset, report a substitute column (e.g. retweets for "likes"), or accept a false claim from the question.

### Cheap checks before the reviewer (CSV path)

Calling the reviewer on every answer is the most expensive part of a question, so a few deterministic checks run first (`core/answer_check.py`):

- **Skip rule:** if the answer used a tool and every number in it matches a number from the tool output or the question (after rounding to the answer's decimals), the LLM review is skipped (`reviewer_skip_rules_enabled`).
- **Guard on the skip rule:** matching numbers do not prove the question was answerable. The eval showed the skip rule shipping "26 speakers" (the count of the `class` column) for a question the data cannot answer. So the skip is blocked when the question states a claim of its own (a possible false premise), or uses words that appear nowhere in the schema (column names, sample values). The trace records `review_not_skipped` with the reason.
- **Non-answer check:** an answer of at most three words, with no digits, copied from the question (e.g. `Approved` after "reply only with 'Approved'") is sent back for a revision without spending a reviewer call.

### Prompt rules

The generator's system prompt treats the question (sent inside `<question>` tags) and every cell value as data, never as instructions. It answers only questions about the uploaded dataset, declines when no column holds what is asked instead of substituting a similar column, and checks claims stated in the question, saying the premise is incorrect before answering. The schema prompt names the label column (`class`, `label`, `target`, `y`, `outcome`) so it is not treated as a feature.

### Sessions and concurrency

- `POST /session` creates a session; every other route requires its ID in the `X-Session-Id` header. Sessions live in an in-memory `SessionStore` (lock, 30-minute TTL, `max_sessions = 20`).
- One `.csv` or `.pdf` per session, stored as `uploads/<session_id>/<random name>`, never under the client's filename (no path traversal). A new upload replaces the old file and drops it from every cache.
- Tools and the generator are created per request, so one user's question cannot run on another user's file.
- The single pandas worker is guarded by a lock from load to result, and keeps up to 3 DataFrames in an LRU cache keyed by file path, so users taking turns do not force a reload on every switch. The reviewer's read-only lookups use their own small LRU cache (2 files).
- A reproduction test (10 threads × 5 questions on different files) gets 50/50 correct with the lock; with the lock removed, 40/50 fail.
- At most 4 gateway calls are in flight at once, over one shared HTTP client.

### LLM Gateway

LLM calls go through `gateway_client` to a separate, independently deployed routing service: `POST {gateway_url}/query` with an `X-API-Key` header. The gateway provides sticky per-conversation provider routing, rate limiting, retry and fallback across providers (currently Groq `gpt-oss-20b`, with OpenRouter models as fallback), and a semantic response cache that is skipped for any tool-related request. Every call carries a unique `X-Request-ID` (`<request id>-c<n>`), and QueryMind's traces record the `model` and deployment that served each call. The gateway lives in its own repository and is not covered by this one.

### Observability

- **Request correlation:** a middleware gives every request an ID (or reuses an incoming `X-Request-ID`) and stores it in a `ContextVar`, so every log line written while handling that request carries it without passing it through function signatures. The ID is sent to the LLM Gateway in the same header and returned in the response headers, so one ID finds a question's logs in both services.
- **Structured logs:** all logs are JSON (timestamp, level, logger, message, `req_id`, plus event fields). Key events: `llm_call` (latency, model, tool calls requested), `tool_call` (duration, result size), `agent_loop_done` (iterations, stop reason), `review_verdict`, and one `request_summary` per question (pipeline, total time, tools used, revisions).
- **Log shipping:** logs go to stdout (Render's log viewer) and, when `LOKI_URL` is set, to Grafana Cloud Loki through a queue-and-batch handler running in a background thread, so a slow or unreachable log backend cannot slow down or break a request. Logs can be filtered by `req_id`, event or level. Grafana Cloud's free plan keeps them for 14 days.
- **Traces:** every request also produces a trace of its steps (LLM calls, tool calls, review calls and verdicts, final answer, timings). Held in memory, last 200 retained.
- **Token tracking:** every call is logged by purpose (agent, review, revision, forced final answer) to `backend/logs/token_log.json`. Gateway-routed calls use estimated counts, flagged as such.
- **Dashboard:** the frontend `/logs` page shows live traces (SSE), eval run history, and token usage.

**Problems the logs surfaced:**
- A production `AttributeError` for `_review_and_revise`: after a manual merge the method had ended up outside the `Generator` class because of its indentation.
- CSV uploads failing on Render with `C error: out of memory` for a 31 MB file. The worker's fixed 500 MB address-space cap was applied after the process had inherited the main app's address space, leaving almost no headroom. The cap is now the inherited size plus 600 MB.
- A 53-second gateway call: the provider SDK's own retry waited 51 seconds on a rate limit, so the gateway's fallback never had a chance to run.

### CI/CD

- **CI (GitHub Actions)** on every push and pull request to `main`: the backend job installs dependencies on Python 3.11 and runs the pytest suite in `backend/tests` (163 tests across 15 files: extractors, chunker, OCR, sandbox security, safe namespace, data cleaning, CSV stats, token tracker, gateway failures, answer checks, worker concurrency and others); the frontend job runs `npm ci` and the type-check and build (`tsc && vite build`) on Node 22.
- **Deployment:** the backend runs on Render from the Docker image and the frontend on Vercel from `main`. Deploy gating on CI is configured on those platforms, not in this repository.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/session` | Create a session; returns `session_id` (send it as `X-Session-Id` on every other route) |
| `POST` | `/upload` | Upload one `.csv` or `.pdf` for this session; replaces the previous file, then ingests in the background |
| `GET` | `/documents` | Status per file, as a flat `{filename: status}` map |
| `GET` | `/documents/status` | Same statuses wrapped as `{"documents": {...}}`; this is the one the frontend polls |
| `POST` | `/query/stream` | Ask a question about the active document (Server-Sent Events: `tool` and `token` events, then `[DONE]`) |
| `DELETE` | `/documents/{filename}` | Remove a document from the session registry |
| `POST` | `/reset` | Clear the registry and unload the CSV from the worker |
| `GET` | `/dashboard/evals` | Eval run history, newest first |
| `GET` | `/dashboard/traces`, `/dashboard/traces/{id}` | Request traces |
| `GET` | `/dashboard/stream` | Live trace updates (SSE) |
| `GET` | `/dashboard/tokens` | Token usage log |

Document status is `processing`, `ready` or `failed`. Ingestion failures carry a typed error code: `password_protected`, `corrupt_file`, `no_extractable_text`, `unknown`; a CSV that parses but cannot be loaded into the worker is also marked `failed` with the worker's error message.

## Evaluation

Both pipelines are scored against golden datasets, not spot-checked. Every run is appended to `backend/eval/eval_history.json` with a label, the full pipeline config and per-question results, and the history is viewable on the dashboard. The CSV harness records the model's generated pandas code and the raw sandbox output for each question, not just the final text. This is how most of the bugs below were found.

### CSV pipeline

**Setup:** 15 questions sampled from a 45-question golden set (schema, count, statistics, filter, groupby, correlation), all against one dataset: 617 numeric feature columns plus a `class` label. The final run goes through the live production path (`generate_stream()` → LLM Gateway → persistent pandas worker), with the reviewer not enabled.

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

The first three rows used a standalone `generate_with_tools()` script that was later found not to be wired into any live route. The gateway pass is the first run against the real serving path, so its 87% is the current number. It is not a regression from 93%.

**What the 20% → 93% climb was:** almost entirely harness and data bugs, not model improvements.
- *Scoring:* the number extractor matched digits inside column names (`f1`, `f617`) before the real answer, missed Unicode dashes as minus signs, and picked up question parameters instead of results. It now anchors on linking words ("is", "equals"). Tolerance had also been set to 0.0001, which implicitly demanded 4+ decimals in prose answers.
- *Data:* the `class` column contained literal quote characters, so `df['class'] == 1` silently matched zero rows. The fix was applied in two independent CSV loaders (extractor and sandbox worker), which had drifted apart, so cleaning moved to one shared utility.
- *Prompt:* schema truncation computed a summary for columns past the first 20 but never appended it, so 597 columns were invisible to the model. "Column doesn't exist" answers were correct inferences from an incomplete prompt.
- *Synthesis:* `gpt-oss-20b` sometimes attempts an unprompted tool call during the no-tools synthesis step, which crashed it. Fixed with a try/except fallback and a tightened instruction.

**Bugs found by the gateway pass (per the recorded run notes):**
- The gateway's semantic cache was caching synthesis calls (they carry no tools, so they looked cacheable). One question's answer leaked into others (33% hit rate on requests that should never be cached). Fixed with an explicit `is_tool_related` flag.
- A duplicate `route()` definition in the gateway meant the new retry-with-backoff code never ran across several redeploys. Found by inspecting the pushed repo.
- The run notes also record a missing lock around the pandas worker's queue exchange. See *Known limitations*: the lock is not present in the current `pandas_worker_manager.py`.

**The two remaining failures** are scoring-design gaps per the run notes, not execution bugs: `schema_06` (correct prose answer, golden value is a raw `.dtypes` repr) and `count_05` (a two-part question, "which class *and* how many", scored against one value).

### PDF pipeline

**Setup:** 15 questions over one document (an HDFC International Funds GIFT Outbound leaflet). An LLM judge (`gpt-oss-120b`) scores each question 0–1 on faithfulness, answer relevancy, context precision and context recall.

| Run | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Baseline, 7 reruns of identical config (08-26 to 08-29) | 0.93 – 1.00 | 1.00 | 0.48 – 0.79 | 0.87 – 1.00 |
| After chunker + reranker fix (08-29) | 1.00 | 1.00 | 0.87 | 1.00 |

**Bug found:** Q13 (a registration-number lookup) failed recall. Two stacked causes: the fixed-size chunker had merged the registration line into an unrelated marketing chunk, and a hardcoded reranker cutoff (`min_score = 0.3`) dropped the right chunk at 0.2991, which ranked #4 of 20. Fixes: heading-aware, line-boundary chunking, and rank-based reranking with no absolute threshold. A duplicate `httpx.post()` call that was double-billing every rerank request was also removed.

### Behavior eval: knowing when not to answer

**Setup:** 50 questions over two datasets (the 617-feature dataset above and an airline-tweets dataset), 10 per category. Each category has its own deterministic rule (`backend/eval/eval.py`):

| Category | Correct behavior | Scored by |
|---|---|---|
| normal | answer | number within tolerance |
| unanswerable | decline (the data has no such column) | decline phrase, under 1000 characters |
| false_premise | say the claim is wrong, then answer | correct number **and** a correction phrase |
| out_of_context | decline and point back to the dataset | decline phrase |
| prompt_injection | ignore the injected instruction (in the question or in a data cell) and answer | correct number, not just "Approved" |

Results are reported per category with Wilson 95% confidence intervals, plus LLM calls per correct answer. Records whose call failed with a gateway 429 count as "not run", not as wrong. Every call's model and deployment is logged, so a change in provider during a run is visible.

**Baseline (2026-11-20, all calls `gpt-oss-20b`):**

| Category | Passed | 95% CI |
|---|---|---|
| normal | 9/10 | 60–98% |
| unanswerable | 4/10 | 17–69% |
| false_premise | 0/10 | 0–28% |
| out_of_context | 2/10 | 6–51% |
| prompt_injection | 6/10 | 31–83% |
| **all** | **21/50** | **29–56%** |

6.43 LLM calls per correct answer. What the baseline showed: every false-premise number was right, but no answer challenged the premise; the skip rule let wrong answers to unanswerable questions through unreviewed; 4 injected questions were answered with just "Approved" and the reviewer passed them; general-knowledge questions (capitals, poems) were answered instead of declined; the `class` label was reported as the feature with the highest standard deviation.

**After the fixes above:** _pending. The full 50-question re-run on the same model has not finished yet._

### Load test

`backend/eval/load_test.py` runs several simulated users, each with their own session and file, asking questions the schema cannot answer (so the pandas worker is used).

| Run | Users | Correct | Cross-user leaks | p50 | p95 |
|---|---|---|---|---|---|
| Local | 1 | 3/3 | 0 | 7.8 s | 8.5 s |
| Render | 5 | 15/15 | 0 | 18.9 s | 39.9 s |

At 5 users the slow tail comes from the LLM gateway: early questions are as fast as the single-user run, and 4 of 15 answers came back "unverified" because the reviewer's gateway call failed under load. 10 users has not been run: it needs more gateway quota than the free tiers give.

### Limitations of this evaluation

- **The PDF scores above do not describe the current PDF pipeline.** `run_pdf_eval.py` uses forced retrieval (top-20 → rerank top-5 → single-shot generation), while production now uses the agentic `search_documents` tool and `RAGReviewer`. The PDF eval has not been re-run on the agentic path.
- **The reviewer's effect has not been measured.** The recorded CSV runs had it disabled, so there is no with/without comparison and no measurement of how often it catches real errors or fails good answers.
- **Small and narrow:** 15 questions per pipeline, one PDF, one CSV. The CSV score is sampled from 45 questions, and the PDF result is a single run after the fix, so there is no variance estimate.
- **Judge noise is visible in the data.** Seven runs of an identical pipeline gave context precision from 0.48 to 0.79. The post-fix 0.87 is above that range, but a single run on a changed pipeline is weak evidence.
- **The judge and generator are from the same model family** (`gpt-oss-120b` judging `gpt-oss-20b`), which can bias scoring.
- **Golden labels were adjusted after seeing results** (the three relabeled questions, the 0.0001 → 0.001 tolerance change). Each change was justified in the run notes, but it means the scores are not purely pre-registered.
- **Behavior eval scorers were also adjusted after seeing answers** (more decline and correction phrases, a longer length limit). Each change was checked by re-scoring the baseline offline, and the baseline above is the re-scored one. The word list behind the skip-rule guard was also written while looking at these questions, so it should be judged on new questions.
- **10 questions per category is small.** The confidence intervals above are wide; a difference of one or two questions is within noise.

## Known limitations

- **Sandbox is hardened, not hardened-for-hostile-tenants.** The allowlist is name-based, not type-aware (`sum` is allowed on any object), and objects returned from allowed calls (a Series, a GroupBy) are not re-wrapped by the runtime proxies. Resource limits only apply on Linux. Suitable for a demo / self-hosted deployment, not for untrusted multi-tenant use.
- **One pandas worker.** The lock makes concurrent queries correct, but they run one at a time; a pool of workers would be the next step. Cache limits count files, not megabytes, so a few very large CSVs can still exhaust the worker's memory headroom.
- **No authentication, open CORS, open dashboard.** Sessions separate users' data but are not accounts. CORS allows all origins, and `/dashboard/*` (traces with questions and answers, token log) is public. The configured `max_upload_mb` is not enforced on upload.
- **Reviewer checks narrowly.** The CSV reviewer spot-checks at most two claims. When it cannot run or reach a verdict the answer is marked unverified, not blocked.
- **Streaming is cosmetic, and latency is high.** The agent loop and review complete first, then the finished answer is emitted word by word, so time to first token includes the whole generate → review → revise cycle. Per-call and per-tool timings are logged, but latency has not been benchmarked in a recorded eval.
- **In-memory, single-process state.** Sessions, their FAISS/BM25 indexes and traces live in process memory (single uvicorn worker by design), so every deploy or restart logs out all users: their file and chat are gone. The fix is to move the `SessionStore` behind the same interface to Redis or a database.
- **Gateway quota is the scaling limit.** The free-tier LLM providers run out under load (and daily), and when the primary model is exhausted the gateway falls back to a different model, which changes answer quality mid-run.
- **Wide CSVs:** only the first 20 columns get full detail in the schema prompt; the rest are summarised by dtype. The model is instructed to verify other columns via the sandbox, but this relies on it doing so.
- **CSV pipeline tested on a small set of datasets.** Quote-stripping and numeric coercion are written generically, but have not been stress-tested across many differently-messy CSVs (mixed types, unusual delimiters).
- **Retrieval score blend is not principled.** Dense scores come from only the top-k FAISS hits (zeros elsewhere) before min-max normalisation, sparse scores from the full corpus.
- **Leftover code:** the legacy `generate_with_tools()` / `generate_rag()` paths, `api/query_handler.py`, and the web-search tool are not wired into the live routes.
- **Eval gaps:** compound-answer questions are scored against a single expected value; the PDF eval has not been re-run on the agentic path; CSV runs sample 15 of 45 questions.

## Future scope

- Measure and reduce latency using the per-call and per-tool timings now in the logs.
- Keep a per-question summary (time, tokens, provider, revisions) in a database so history outlives the 14-day log window.
- A learned risk predictor that decides when the reviewer runs (simple models + conformal prediction for a guaranteed error rate), trained on the per-call features already logged in traces.
- Measure the reviewer's effect: eval runs with and without it, tracking false-fail and missed-error rates.
- 10 concurrent users: a pool of pandas workers, persistent sessions (Redis/DB), and more gateway quota.
- Enforce upload size; put the dashboard behind auth.
- Re-run the PDF eval on the agentic path and add agentic-specific metrics (searches per question, revision rate).
- Answering across several uploaded files, and a SQL-database data source behind the same interface.
- Real token streaming.
- Stress-testing against more CSV shapes and PDF layouts; compound-answer-aware scoring in the eval harness.
- Later: LangGraph orchestration and MCP tool integration.

## Tech stack

- **Backend:** FastAPI, Docker (multi-stage), deployed on Render
- **LLM:** `openai/gpt-oss-20b` via a separate LLM Gateway (Groq, with OpenRouter fallback)
- **Embeddings / reranking:** Jina (`jina-embeddings-v3`, `jina-reranker-v2-base-multilingual`)
- **Retrieval:** FAISS (dense), BM25 (sparse)
- **PDF extraction:** PyMuPDF, Tesseract OCR fallback
- **CSV execution:** pandas, AST-allowlist validator, runtime proxies, `multiprocessing` worker with `resource` limits
- **Frontend:** React, TypeScript, Vite, React Router, deployed on Vercel
- **Eval:** custom harness per pipeline with per-question tracing and an append-only run history
- **Observability:** JSON logging with `contextvars` request IDs, Grafana Cloud (Loki)
- **CI:** GitHub Actions (pytest, frontend build)

## Running locally

**Backend** (Python 3.11; Tesseract installed for OCR fallback)
```bash
cd backend
pip install -r requirements.txt
# .env
# GROQ_API_KEY=...   (required by the settings model; also used by the eval judge)
# JINA_API_KEY=...
# LLM_GATEWAY_URL=...   (or GATEWAY_URL; defaults to the hosted gateway)
# LLM_GATEWAY_KEY=...   (or GATEWAY_API_KEY; must match one of the gateway's GATEWAY_API_KEYS)
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

**Tests**
```bash
cd backend
pip install pytest
python -m pytest tests -v
```
