# QueryMind

An agentic RAG system for document Q&A — supports both unstructured (PDF) and structured (CSV) data through a single API, with distinct retrieval strategies for each.

**Live demo:** https://query-mind-cyan.vercel.app/#/ · **API:** https://query-mind.onrender.com

---

## What it does

Upload a PDF or CSV and ask questions about it in plain language.

- **PDF path:** extracts text with heading context preserved → chunks without cutting mid-line → retrieves via hybrid search (BM25 + FAISS) → reranks with Jina → generates a streamed, grounded answer.
- **CSV path:** parses the file into a structured schema → the model writes and executes real pandas code in a sandboxed environment → synthesizes a plain-language answer from the actual computed result, not a guess.

## Architecture

**PDF pipeline:**
`PDFExtractor → TextChunker (heading-aware) → HybridRetriever (BM25 + FAISS) → Reranker (Jina, top_k) → Generator (RAG stream)`

**CSV pipeline:**
`CSVExtractor → CSVSchema → Generator (2-call tool-calling: writes pandas code → PandasSandboxTool executes → synthesizes answer)`

Both extractors return a discriminated union (`PDFExtractionResult` / `CSVExtractionResult`) so the API can branch on document type without forcing CSV data into a shape designed for PDFs.

## Scope & design decisions

- **Single active document, by design.** `/upload` clears any previous document before ingesting a new one. This is a deliberate simplification for a portfolio-scale system, not an oversight — multi-document support is a known, deferred extension.
- **Rank-based reranking, not absolute thresholds.** Early versions filtered reranked chunks by a fixed score cutoff; testing showed the same numerical score could mean "relevant" for one query and "noise" for another. Reranking now relies purely on Jina's relative rank (`top_k`), not an absolute score.
- **Retrieval is explicit, not tool-based, on the PDF path.** Early designs let the model decide whether to retrieve via tool-calling; this was unreliable — the model sometimes answered without retrieving. Retrieval is now forced at the route layer for PDFs. Tool-calling is reserved for CSV computation, where the "tool" (pandas execution) genuinely needs the model's judgment about *what* to compute.
- **Sandboxed code execution for CSV, not templated queries.** Rather than a fixed set of supported operations, the model writes arbitrary pandas code, which runs inside an AST-validated, restricted-builtins sandbox with a hard process-level timeout — trading a larger attack surface for real analytical flexibility, contained by the sandbox.
- **Eval-driven development, not spot-checking.** A custom evaluation harness (not an off-the-shelf framework) traces each question down to the model's actual generated code and raw tool output, not just its final text answer — this was essential: several early "wrong answer" cases turned out to be correct model output being mis-scored by the eval itself, not model failures.

## Known limitations / in progress

- **Schema truncation on wide CSVs.** For CSVs with many columns, only the first N columns get full detail in the model's prompt (to stay under the LLM's token limit); the rest are meant to be summarized but currently aren't surfaced at all, which can cause the model to incorrectly claim a column doesn't exist. Root-caused; fix in progress (`get_column_info` on-demand lookup tool).
- **Single tool call per turn.** The current 2-call architecture (decide → synthesize) doesn't yet support the model making a follow-up tool call after seeing an initial tool result. Fine for today's single-step questions; will need revisiting once look-up tools are added.
- **LLM Gateway (parallel project).** A standalone routing/cost/rate-limiting service (token bucket algorithm) exists as a separate deployed component; integration with QueryMind is designed but not yet wired in.

## Tech stack

- **Backend:** FastAPI, Docker, deployed on Render
- **LLM:** Groq API (`llama-3.1-8b-instant`, `openai/gpt-oss-20b`)
- **Retrieval:** FAISS (dense), BM25 (sparse), Jina reranker
- **PDF extraction:** PyMuPDF, Tesseract OCR
- **CSV execution:** pandas, AST-validated sandbox with `multiprocessing`-based timeout
- **Frontend:** React, TypeScript
- **Eval:** custom harness with per-question tracing, append-only labeled run history for before/after comparison
