import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extractors.pdf_extractor import PDFExtractor
from core.chunker import TextChunker
from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = r"D:\query-mind\backend\eval\Leaflet - HDFC International Funds - GIFT Outbound Retail - Class B.pdf"

embedder  = Embedder()
indexer   = Indexer()
retriever = HybridRetriever(indexer=indexer)
reranker  = Reranker()


def setup_pipeline(pdf_path: str):
    extractor = PDFExtractor()
    chunker = TextChunker()
    result = extractor.extract(pdf_path)
    all_chunks = chunker.chunk_pages(result.pages)
    all_chunks = embedder.embed_chunks(all_chunks)
    indexer.index(all_chunks)
    print(f"Indexed {len(all_chunks)} chunks from {pdf_path}")


def inspect(question: str, top_k: int = 20, rerank_k: int = 5):
    query_embedding = embedder.embed_query(question)
    chunks = retriever.retrieve(query=question, query_embedding=query_embedding, top_k=top_k)

    print(f"\n--- RETRIEVED (top {top_k}) ---")
    for c in chunks:
        snippet = c["text"][:70].replace("\n", " ")
        print(f"  score={c['retriever_score']:.4f} | {snippet}")

    reranked = reranker.rerank(query=question, chunks=chunks, top_k=rerank_k)

    print(f"\n--- RERANKED (top {rerank_k}) ---")
    for c in reranked:
        snippet = c["text"][:70].replace("\n", " ")
        print(f"  rerank_score={c.get('rerank_score', 0):.3f} | {snippet}")


if __name__ == "__main__":
    setup_pipeline(PDF_PATH)

    while True:
        q = input("\nEnter question (or 'quit'): ")
        if q.lower() == "quit":
            break
        inspect(q)