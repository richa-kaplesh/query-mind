from core.tools.base_tool import BaseTool
from core.embedder import Embedder
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.token_utils import estimate_tokens

MAX_RESULT_TOKENS = 3000


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

    # Sized around MAX_RESULT_TOKENS (~3000 tokens, roughly 4 chars/token) plus
    # headroom for citation formatting and the full-document notice, with a
    # safety margin for a worst-case oversized single chunk. Deliberately set
    # on this tool, not left to the shared default — a flat global character
    # limit sized for short pandas results would silently clip a normal RAG
    # result before the token budget above even kicks in.
    max_result_chars = 18000

    def __init__(self, retriever: HybridRetriever, embedder: Embedder, reranker: Reranker):
        self.retriever = retriever
        self.embedder = embedder
        self.reranker = reranker

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        total = len(self.retriever.indexer.chunks)
        return (
            f"Search the uploaded document ({total} indexed passages total) for passages relevant "
            "to a query. Returns as many of the most relevant passages as fit in a safe response size "
            "— for a short document that may be everything in one call; for a longer one, only the "
            "best matches. Call again with a refined or narrower query if you need something a "
            "search didn't return."
        )

    def run(self, query: str) -> str:
        if self.retriever.indexer.faiss_index is None:
            return "No documents uploaded yet."

        total_chunks = len(self.retriever.indexer.chunks)
        query_embedding = self.embedder.embed_query(query)
        candidates = self.retriever.retrieve(query, query_embedding, top_k=min(total_chunks, 30))
        reranked = self.reranker.rerank(query, candidates, top_k=min(total_chunks, 15))

        selected = []
        tokens_so_far = 0
        for chunk in reranked:
            chunk_tokens = estimate_tokens(chunk["text"])
            if selected and tokens_so_far + chunk_tokens > MAX_RESULT_TOKENS:
                break
            selected.append(chunk)
            tokens_so_far += chunk_tokens

        passages = format_passages(selected)
        if selected and len(selected) >= total_chunks:
            passages = (
                "[This is the ENTIRE content of the document — every indexed passage. If what "
                "you're looking for isn't below, it does not appear in this document. Do not "
                "search again — answer accordingly.]\n\n" + passages
            )
        return passages