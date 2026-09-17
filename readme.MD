# QueryMind

A RAG system for querying PDF and CSV documents, built from scratch. Upload a file, ask questions about it in plain language.

**Frontend:** https://query-mind-cyan.vercel.app/#/
**Backend:** https://query-mind.onrender.com/

---

## What it does

- **PDF path:** extracts text with heading structure preserved → chunks without cutting mid-line → hybrid retrieval (BM25 + FAISS) → reranking (Jina) → generates a grounded answer with citations.
- **CSV path:** parses the file into a structured schema → the model writes real pandas code → executes it in a sandboxed, timeout-guarded process → synthesizes a plain-language answer from the actual computed result.

## Architecture

**PDF pipeline:**
`PDFExtractor (heading-aware) → TextChunker (line-boundary-respecting) → HybridRetriever (BM25 + FAISS) → Reranker (Jina, rank-based top_k) → Generator (RAG, streamed)`

**CSV pipeline:**
`CSVExtractor → CSVSchema (with shared data-cleaning: strips stray quote characters, converts to numeric where possible) → Generator (2-call tool-calling: model writes pandas code → PandasSandboxTool executes in an isolated process → synthesizes answer)`

Both extractors return a discriminated union (`PDFExtractionResult` / `CSVExtractionResult`) so the API can branch on document type without forcing CSV data into a shape built for PDFs.

## Evaluation

Both pipelines are checked against golden datasets with known-correct answers, not just spot-checked by eye.

**CSV path** — custom harness that traces each answer back to the model's actual generated code and raw tool output, not just its final text. This mattered: several early "wrong answer" cases turned out to be correct model output being mis-scored by the eval script itself, not model failures.
- First run: 20% answer correctness.
- After finding and fixing nine distinct bugs — extraction-regex errors on multi-digit and negative values, a Unicode dash mismatch, a scoring crash on range comparisons, stale golden-dataset values, a source-data quoting issue that silently broke two independent CSV-loading paths, a synthesis-call crash specific to the deployed model — final run: **~87–93% answer correctness** across repeated runs (score varies slightly by run due to two known, documented scoring gaps below, not by model inconsistency).

**PDF path** — evaluated with faithfulness, answer relevancy, context precision, and context recall metrics on a 15-question set.
- Faithfulness: 100%, Answer Relevancy: 100%, Context Recall: 100%, Context Precision: ~86%.
- One real bug found and fixed this way: a fixed-size chunker was merging a specific data line into an unrelated chunk, diluting its retrievability; a hardcoded reranker score threshold was also separately dropping a correct chunk that ranked #4 by a margin of 0.0009. Fixed by moving to heading-aware chunking and rank-based (not absolute-threshold) reranking.

**Live debug dashboard** — a separate view (traces, per-run evaluations, and token usage by call purpose) deployed alongside the app, used during development to inspect individual requests rather than relying on aggregate scores alone.

## Design decisions

- **Single active document, by design.** `/upload` clears any previous document and resets the retrieval index before ingesting a new one — not a multi-document system.
- **Rank-based reranking, not absolute score thresholds.** An early version filtered by a fixed score cutoff; testing showed the same numerical score meant "relevant" for one query and "noise" for another, and it silently dropped a correct answer sitting just below the cutoff. Reranking now uses relative rank from Jina's sorted output.
- **Sandboxed code execution for CSV, not a fixed set of supported operations.** The model writes arbitrary pandas code, run inside an AST-validated, restricted-builtins, timeout-guarded, memory-limited-in-a-separate-process sandbox — trading a larger surface for real analytical flexibility, contained by the sandbox.
- **Eval-driven, not spot-checked.** Every fix in this project traces to a specific, reproduced failure — not "the number went up so it's fine."

## Known limitations

- **Sandbox uses AST-based denylisting, not allowlisting.** Adequate for self-run/demo use; not hardened against untrusted multi-tenant input. Known gaps: `pd.read_pickle`/`to_pickle` (deserialization risk), `read_csv(url)` (SSRF risk), `df.to_csv(path)` (arbitrary file write), no memory `rlimit`. A production version needs an attribute allowlist and OS-level resource limits, not just an AST blocklist.
- **Single tool call per turn.** The current CSV flow (decide → synthesize) doesn't yet support the model making a follow-up tool call after seeing an initial result — fine for today's single-step questions, will need a proper loop for multi-step ones.
- **Two known eval scoring gaps, not model failures:** (1) compound-answer questions (e.g. "which class has the fewest samples, and how many") are scored against only one expected value, so a partially-correct-by-design answer can score 0 even when the computed number is right; (2) occasional sandboxed queries take longer than expected under back-to-back batch load (likely per-call process spin-up overhead), independent of query complexity — a worker-pool design would remove this rather than a longer timeout.
- **CSV pipeline has been tested against one real dataset so far.** The fixes made (quote-stripping, numeric coercion, schema truncation handling) are written generically, not hardcoded to this dataset, but haven't yet been stress-tested against a second, differently-messy CSV (missing values, mixed types, different delimiter conventions).
- **Retrieval scoring blends dense and sparse scores at different normalization scales** — dense scores are normalized across the full corpus (mostly zeros, so effectively `score/max`), while sparse scores get a true min-max normalization over the same set. Functionally fine today, but not a principled blend.

## Future scope

- Multi-step tool-calling (a real reasoning loop, not single-call-then-synthesize) for both the CSV agentic path and PDF retrieval.
- Allowlist-based sandbox hardening + OS-level resource limits, ahead of any untrusted-input use.
- A persistent worker pool for CSV sandbox execution, removing per-query process/import/CSV-load overhead.
- Multi-document support.
- PDF-side stress testing against a wider variety of document layouts and lengths.
- Compound-answer-aware scoring for the eval harness.

## Tech stack

- **Backend:** FastAPI, Docker, deployed on Render
- **LLM:** Groq API (`llama-3.1-8b-instant`, `openai/gpt-oss-20b`)
- **Retrieval:** FAISS (dense), BM25 (sparse), Jina reranker, Jina embeddings
- **PDF extraction:** PyMuPDF, heading-aware line-boundary chunking
- **CSV execution:** pandas, AST-validated sandboxed subprocess with timeout
- **Frontend:** React, TypeScript, deployed on Vercel
- **Eval:** custom harness per pipeline, with per-question tracing and an append-only labeled run history for before/after comparison, viewable via a live debug dashboard (traces, evaluations, token usage)