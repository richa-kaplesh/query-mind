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
        self.chunks = chunks               # store chunks for later lookup

        #bm25 - tokenize and build keyword index ---take all chunk words → calculate IDF(rarity) for every word → store it → done
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

        #faiss - extrat embeddings , normalize , build vector index ---take all chunk vectors → put them in a searchable structure → done
        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)


# No scoring happening yet. No query exists yet. we are just organizing data so searching becomes fast later.

