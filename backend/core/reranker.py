from sentence_transformers import CrossEncoder
from typing import List

class Reranker:

    def __init__(self, model_name : str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):

        self.model = CrossEncoder(model_name)

    def rerank(self, query:str, chunks: List[dict],top_k:int =5) -> List[dict]:
        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        for i , chunk in enumerate(chunks):
            chunk["rerank_score"]= float(scores[i])


        reranked = sorted(chunks, key=lambda x:x["rerank_score"], reverse=True)

        return reranked[:top_k]
    