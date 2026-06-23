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

**LLM Gateway Integration**
A separate open-source package has been built with semantic caching, cost tracking, fallback routing across providers, and SQLite-backed request logging. Planned as a pip-installable dependency for QueryMind to add production-grade LLM infrastructure without coupling it to the core RAG system.

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
