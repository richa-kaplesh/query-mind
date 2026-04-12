# Retriever = search and score.
import faiss
import numpy as np
from typing import List
from core.indexer import Indexer

class HybridRetriever:

    def __init__(self , indexer, alpha:float=0.5):
        self.indexer = indexer
        self.alpha = alpha
#faiss search
    def retrieve(self , query:str , query_embedding:np.ndarray , top_k: int = 20) -> List[dict]:
        query_embedding = query_embedding.astype("float32").reshape(1,-1)
        faiss.normalize_L2(query_embedding)
        dense_scores , dense_indices = self.indexer.faiss_index.search(query_embedding, top_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        #bm25 dearch
        tokenized_query  = query.lower().split()
        sparse_scores = self.indexer.bm25.get_Scores(tokenized_query)


        #normalize scores
        def normalize(scores):
           min_s , max_s= scores.min(),scores.max()
           if max_s - min_s == 0:
            return scores
        
           return (scores-min_s)/(max_s-min_s)
    
        dense_all = np.zeros(len(self.indexer.chunks))
        dense_all[dense_indices]=dense_scores

        dense_all = normalize(dense_all)
        sparse_all = normalize(sparse_scores)

        hybrid_Scores = self.alpha*dense_all+(1-self.alpha) *sparse_all

        top_indices = np.argsort(hybrid_Scores)[::-1][:top_k] 
        results = []
        for idx in top_indices:
           chunk = self.indexer.chunl[idx].copy()
           chunk["retriever_score"]= float(hybrid_Scores[idx])
           results.append(chunk)

        
        return results