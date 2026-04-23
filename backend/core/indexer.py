import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List

class Indexer:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension         # size of each embedding vector
        self.chunks = []                   # all chunks stored for lookup
        self.bm25 = None                   # keyword search index
        self.faiss_index = None
    

    def index(self, chunks: List[dict]) -> None:
        self.chunks.extend(chunks)             # ✅ append, don't overwrite

        # rebuild BM25 over ALL chunks
        tokenized = [chunk["text"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        # rebuild FAISS over ALL chunks
        embeddings = np.array([chunk["embedding"] for chunk in self.chunks]).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

# No scoring happening yet. No query exists yet. we are just organizing data so searching becomes fast later.

