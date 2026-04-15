import json
import time
from groq import Groq
from core.pdf_extractor import PDFExtractor
from core.chunker import TextChunker
from core.embedder import Embedder
from core.indexer import Indexer
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import Generator

embedder = Embedder()
indexer = Indexer()
retriever = HybridRetriever(indexer = indexer)
reranker = Reranker()
generator = Generator()
groq_client = Groq()

def setup_pipeline(pdf_path: str):
    extractor = PDFExtractor()
    chunker= TextChunker()

    pages = extractor.extract(pdf_path)

    all_chunks = []
    for page in pages:
        chunks= chunker.chunk_text(page["text"],metadata= page["metadata"])
        all_chunks.extend(chunks)

    all_chunks= embedder.embed_chunks(all_chunks)
    indexer.index(all_chunks)

    print(f"indexed {len(all_chunks)} chunks from {pdf_path}")