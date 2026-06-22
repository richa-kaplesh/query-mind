from core.tools.base_tool import BaseTool
from core.embedder import Embedder
from core.retriever import HybridRetriever
from core.reranker import Reranker
from typing import List

from core.tools.base_tool import BaseTool
from core.embedder import Embedder
from core.retriever import HybridRetriever
from core.reranker import Reranker
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