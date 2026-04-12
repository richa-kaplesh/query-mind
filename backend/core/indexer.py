import faiss
import numpy as np
from rank_bm25 import BM250kapi
from typing import List

class HyrbidRetriever:

    def __init__(self,alpha:float=0.5):
        self.alpha = alpha
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
    
    def index(self , chunks:List[dict]) ->None:
        self.chunks = chunks
        
        #bm25 - tokenize and build keyword index ---take all chunk words → calculate IDF(rarity) for every word → store it → done
        tokenized = [chunk["text"].lower().split()for chunk in chunks]
        self.bm25 = BM250kapi(tokenized)

        #faiss - extrat embeddings , normalize , build vector index ---take all chunk vectors → put them in a searchable structure → done

        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)

# No scoring happening yet. No query exists yet. we are just organizing data so searching becomes fast later.
