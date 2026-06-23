# QueryMind v2

An agentic RAG system that goes beyond simple document retrieval. QueryMind v2 implements query-intent-aware routing, hybrid retrieval, and tool-based architecture to answer questions from documents, structured datasets, and the web.

---

## Architecture

QueryMind is built around a clear separation between the ingestion pipeline and the query pipeline.

**Ingestion Pipeline**
File upload triggers extraction, chunking, embedding, and indexing — completely decoupled from query processing. Supports PDF (with OCR fallback for scanned pages) and CSV formats.

**Query Pipeline**
Queries are routed based on file type and intent. PDF queries go through hybrid BM25 + FAISS retrieval with cross-encoder reranking. CSV queries are routed to a statistical analysis pipeline. General queries fall through to the LLM with web search capability.

**Tool-Based Generator**
The generator uses Groq's tool calling to decide between three tools — document search, CSV statistical analysis, and web search. For document queries, retrieval is handled explicitly in the application layer rather than delegated to the LLM, a deliberate architectural decision made after observing that LLM-driven retrieval routing is unreliable in production.

---

## Key Design Decisions

**Hybrid CSV Strategy**
CSV files support two distinct query paths. Lookup questions use row-based retrieval where each row becomes an ExtractedPage. Analytical and ML questions trigger a statistical analysis pipeline built on pandas and scipy, returning a comprehensive stats report — column distributions, correlation matrix, outlier detection, normality tests, and feature-target significance tests. The LLM interprets this report rather than computing statistics itself, ensuring deterministic and trustworthy results.

**Explicit RAG Routing**
Rather than asking the LLM to decide when to retrieve documents, retrieval is triggered explicitly in the application layer when a document is uploaded and ready. This makes the system predictable and debuggable. Tool calling is reserved for web search and CSV stats — decisions that genuinely benefit from LLM reasoning.

**SOLID Architecture**
Each component has a single responsibility. `PDFExtractor` and `CSVExtractor` handle extraction only. `TextChunker` handles chunking only. `IngestionPipeline` orchestrates the ingestion flow. `PromptBuilder` handles prompt construction. `ToolRegistry` manages tool registration and execution. `Generator` only orchestrates LLM calls. `FileHandler` and `QueryHandler` separate file management from query routing in the API layer.

---

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **LLM**: Groq (llama-3.1-8b-instant)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Retrieval**: FAISS (dense), BM25 (sparse), hybrid scoring
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **PDF Extraction**: PyMuPDF, Tesseract OCR
- **CSV Analysis**: pandas, scipy
- **Web Search**: DuckDuckGo Search
- **Frontend**: React

---

## Features

- Upload PDF or CSV files and ask questions about their content
- Hybrid retrieval combining semantic and keyword search
- Cross-encoder reranking for improved answer relevance
- Statistical analysis pipeline for CSV datasets — correlations, distributions, outlier detection, class imbalance
- Web search for queries requiring current information
- Streaming responses via Server-Sent Events
- Conversation history maintained across turns
- OCR fallback for scanned PDF pages

---

## Running Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Set your environment variables in a `.env` file:
