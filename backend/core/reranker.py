import httpx
from typing import List
from config import settings

# Jina rerank endpoint
_RERANK_URL = "https://api.jina.ai/v1/rerank"


class Reranker:
    """
    Calls the Jina AI Rerank API (jina-reranker-v2-base-multilingual).

    Keeps the same rerank(query, chunks, top_k) signature so routes.py
    and any other call sites need zero changes.
    """

    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._model = settings.jina_rerank_model   # "jina-reranker-v2-base-multilingual"

    def rerank(self, query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
        """
        Rerank chunks against query using the Jina Rerank API.

        Returns up to top_k chunks, each with chunk["rerank_score"] set to
        the Jina relevance_score (0–1).  Order is highest-score first.
        """
        if not chunks:
            return []

        documents = [chunk["text"] for chunk in chunks]

        payload = {
            "model":     self._model,
            "query":     query,
            "documents": documents,
            "top_n":     top_k,
        }

        resp = httpx.post(
            _RERANK_URL,
            headers=self._headers,
            json=payload,
            timeout=30.0,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Jina rerank API error {resp.status_code}: {resp.text[:400]}"
            )

        data = resp.json()
        # Response: {"results": [{"index": i, "relevance_score": f, "document": {...}}, ...]}
        # Already sorted highest→lowest by the API, but we set rerank_score explicitly.
        reranked: list[dict] = []
        for result in data["results"]:
            chunk = chunks[result["index"]].copy()
            chunk["rerank_score"] = float(result["relevance_score"])
            reranked.append(chunk)

        return reranked