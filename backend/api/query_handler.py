import logging
from typing import List
from core.tools.csv_stats_tool import CSVStatsTool

log = logging.getLogger("query_handler")


def get_current_file(documents: dict) -> dict | None:
    return next((v for v in documents.values() if v.get("status") == "ready"), None)


def retrieve_chunks(request, query: str) -> List[dict]:
    embedder = request.app.state.embedder
    retriever = request.app.state.retriever
    reranker = request.app.state.reranker

    query_embedding = embedder.embed_query(query)
    chunks = retriever.retrieve(query=query, query_embedding=query_embedding, top_k=20)
    chunks = reranker.rerank(query=query, chunks=chunks, top_k=5)
    log.info(f"[QUERY] Retrieved {len(chunks)} chunks")
    return chunks


def resolve_query(request, query: str, documents: dict, conversation_history: list = []) -> tuple:
    generator = request.app.state.generator
    current_file = get_current_file(documents)
    is_csv = current_file and current_file.get("file_type") == ".csv"

    if current_file and not is_csv:
        log.info("[QUERY] PDF path — running retrieval...")
        chunks = retrieve_chunks(request, query)
        return generator.generate(query=query, chunks=chunks, conversation_history=conversation_history), chunks

    if is_csv:
        log.info("[QUERY] CSV path — registering stats tool...")
        generator.registry.register(CSVStatsTool(current_file["file_path"]))
        return generator.generate(query=query, chunks=[], conversation_history=conversation_history), []

    return generator.generate(query=query, chunks=[], conversation_history=conversation_history), []


def build_sources(chunks: List[dict]) -> List[dict]:
    return [
        {
            "source": chunk["metadata"].get("source", "unknown"),
            "page": chunk["metadata"].get("page", "N/A"),
            "text": chunk["text"],
            "rerank_score": chunk.get("rerank_score", 0)
        }
        for chunk in chunks
    ]