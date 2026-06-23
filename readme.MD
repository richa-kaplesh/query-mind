# QueryMind v2

An agentic RAG system that goes beyond simple document retrieval. QueryMind v2 implements query-intent-aware routing, hybrid retrieval, and tool-based architecture to answer questions from documents, structured datasets, and the web.

---

## Demo

**Conversation Memory — QueryMind remembers context across turns**
![Conversation Memory](D:\query-mind\screenshots\image.png)

**CSV Statistical Analysis — Upload a dataset and get a full stats report**
![CSV Stats](D:\query-mind\screenshots\image1.png)

**PDF Query — Upload a document and ask questions with source citations**
![PDF Query](D:\query-mind\screenshots\image2.png)

---

## Architecture

QueryMind is built around a clear separation between the ingestion pipeline and the query pipeline.

**Ingestion Pipeline**
File upload triggers extraction, chunking, embedding, and indexing — completely decoupled from query processing. Supports PDF (with OCR fallback for scanned pages) and CSV formats.

**Query Pipeline**
Queries are routed based on file type and intent. PDF queries go through hybrid BM25 + FAISS retrieval with cross-encoder reranking. CSV queries are routed to a statistical analysis pipeline. General queries fall through to the LLM with web search capability.

**Tool-Based Generator**
The generator uses Groq's tool calling to orchestrate three tools — document search, CSV statistical analysis, and web search. Document retrieval is handled explicitly in the application layer rather than delegated to the LLM, a deliberate decision to keep the system predictable and debuggable.

---

## Key Design Decisions

**Hybrid CSV Strategy**
CSV files support two distinct query paths. Lookup questions use row-based retrieval where each row becomes an ExtractedPage. Analytical and ML questions trigger a statistical analysis pipeline built on pandas and scipy, returning a comprehensive stats report — column distributions, correlation matrix, outlier detection, normality tests, and feature-target significance tests. The LLM interprets this report rather than computing statistics itself, ensuring deterministic and trustworthy results.

**Explicit RAG Routing**
Rather than asking the LLM to decide when to retrieve documents, retrieval is triggered explicitly in the application layer when a document is uploaded and ready. Tool calling is reserved for web search and CSV stats — decisions that genuinely benefit from LLM reasoning.

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
- **Web Search**: DuckDuckGo Search (free tier)
- **Frontend**: React

---

## Features

- Upload PDF or CSV files and ask questions about their content
- Hybrid retrieval combining semantic and keyword search
- Cross-encoder reranking for improved answer relevance
- Statistical analysis pipeline for CSV datasets — correlations, distributions, outlier detection, class imbalance
- Web search capability via DuckDuckGo (rate limited on free tier — see Future Scope)
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

```
GROQ_API_KEY=your_key_here
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload PDF or CSV file |
| GET | /documents | List uploaded documents and status |
| POST | /query | Non-streaming query |
| POST | /query/stream | Streaming query via SSE |
| DELETE | /documents/{filename} | Delete a document |
| POST | /reset | Reset all documents and index |

---

## Future Scope

**Containerization**
Docker support for consistent deployment across environments. Dockerfile is written and ready — pending virtualization support on the deployment machine.

**Web Search Reliability**
Current web search uses DuckDuckGo which has rate limiting and inconsistent results on the free tier. Planned upgrade to a production-grade search API (Serper or Tavily) for reliable real-time web results.

**LLM Gateway Integration**
A separate open-source package has been built with semantic caching, cost tracking, fallback routing across providers, and SQLite-backed request logging. Planned as a pip-installable dependency for QueryMind to add production-grade LLM infrastructure without coupling it to the core RAG system.

**MCP Server Integration**
Extend the tool-based architecture to support Model Context Protocol servers, enabling QueryMind to connect to external services like Google Drive, Notion, Slack, and other data sources as retrievable knowledge bases.

**Voice Recognition**
Add voice input support so users can speak their queries directly instead of typing, making the interface more accessible and natural for document Q&A workflows.

**Persistent Chat History**
Current conversation history is maintained in-memory per session. Future versions will persist chat history to a database, enabling users to resume conversations across sessions and search past interactions.

**Additional File Format Support**
DOCX and Excel extractors are planned for v3. JSON and PPT support deprioritized based on realistic use case frequency.

**System Design Improvements**
Multi-file routing to handle concurrent uploads of different file types. A dedicated query classifier layer for more deterministic intent detection. Evaluation pipeline using RAGAS metrics already prototyped in the repository.

---

## Version History

- **v1.0.0** — Basic RAG with PDF support, hybrid retrieval, cross-encoder reranking
- **v2.0.0** — Agentic tool calling, CSV statistical analysis pipeline, OCR support, SOLID refactor, conversation history, web search