from backend.core.tools.base_tool import BaseTool
from backend.core.embedder import Embedder
from backend.core.retriever import HybridRetriever
from backend.core.reranker import Reranker
from typing import List

from backend.core.tools.base_tool import BaseTool
from backend.core.embedder import Embedder
from backend.core.retriever import HybridRetriever
from backend.core.reranker import Reranker
from typing import List

class RAGTool(BaseTool):

    def __init__(self, retriever: HybridRetriever, embedder: Embedder, reranker: Reranker):
        self.retriever = retriever
        self.embedder = embedder
        self.reranker = reranker

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return "Search through uploaded documents to answer questions based on their content"

    def run(self, query: str) -> dict:
        query_embedding = self.embedder.embed_query(query)
        chunks = self.retriever.retrieve(query, query_embedding)
        reranked = self.reranker.rerank(query, chunks)
        return {"chunks": reranked}