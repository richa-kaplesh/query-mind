from core.tools.base_tool import BaseTool
from core.embedder import Embedder
from core.retriever import HybridRetriever
from core.reranker import Reranker


def format_passages(chunks: list[dict]) -> str:
    if not chunks:
        return "No passages found for this search query. Try a different or broader query."
    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page")
        heading = meta.get("heading")

        citation = f"[Source: {source}"
        if page is not None:
            citation += f" | Page: {page}"
        if heading:
            citation += f" | Section: {heading}"
        citation += "]"

        parts.append(f"{citation}\n{chunk['text'].strip()}")
    return "\n\n---\n\n".join(parts)


class RAGTool(BaseTool):
    """Lets the agent loop decide WHEN to search the document and WITH WHAT QUERY,
    instead of always retrieving before the model has even seen the question.
    Can be called more than once per question with a refined query."""

    def __init__(self, retriever: HybridRetriever, embedder: Embedder, reranker: Reranker):
        self.retriever = retriever
        self.embedder = embedder
        self.reranker = reranker

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search the uploaded document for passages relevant to a query. Returns the most "
            "relevant passages with citations (source, page, section). Call again with a "
            "refined or narrower query if the first search doesn't find what you need."
        )

    def run(self, query: str) -> str:
        if self.retriever.indexer.faiss_index is None:
            return "No documents uploaded yet."
        query_embedding = self.embedder.embed_query(query)
        chunks = self.retriever.retrieve(query, query_embedding)
        reranked = self.reranker.rerank(query, chunks)
        return format_passages(reranked)