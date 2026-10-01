from core.tools.base_tool import BaseTool
from core.embedder import Embedder
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.token_utils import estimate_tokens

# How much of a search result's content budget is "safe" to hand back in one
# tool call. Not a page or chunk-count guess — an actual constraint (context
# cost), so it scales naturally: many small chunks can all fit; a few large
# ones won't.
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
        # Cast a generous net before reranking, then let the token budget below —
        # not a chunk-count guess — decide how many of the best matches actually
        # get returned.
        candidates = self.retriever.retrieve(query, query_embedding, top_k=min(total_chunks, 30))
        reranked = self.reranker.rerank(query, candidates, top_k=min(total_chunks, 15))

        selected = []
        tokens_so_far = 0
        for chunk in reranked:
            chunk_tokens = estimate_tokens(chunk["text"])
            if selected and tokens_so_far + chunk_tokens > MAX_RESULT_TOKENS:
                break  # always keep at least the single best match, even if it alone is large
            selected.append(chunk)
            tokens_so_far += chunk_tokens

        return format_passages(selected)