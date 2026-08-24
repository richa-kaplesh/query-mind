import httpx
import numpy as np
from config import settings

# Jina embeddings endpoint
_EMBED_URL = "https://api.jina.ai/v1/embeddings"

# Jina's documented max texts per request
_BATCH_SIZE = 128


class Embedder:
    """
    Calls the Jina AI Embeddings API (jina-embeddings-v3).

    Output dimension: 1024 (default — matches FAISS index init in indexer.py).

    task values (required by v3+):
      "retrieval.passage" — ingestion / document chunks
      "retrieval.query"   — single user query at search time
    """

    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._model = settings.jina_embed_model        # "jina-embeddings-v3"
        self._dim   = settings.embedding_dimension     # 1024

    # ── Ingestion path ────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Embed all document chunks for ingestion.

        Batches in groups of _BATCH_SIZE to stay within Jina's per-request
        limit.  Mutates each chunk dict in-place: adds chunk["embedding"]
        (np.ndarray float32, shape (1024,)).
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._embed_texts(texts, task="retrieval.passage")

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    # ── Query path ────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query. Returns np.ndarray shape (1024,)."""
        return self._embed_texts([query], task="retrieval.query")[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str], task: str) -> list[np.ndarray]:
        """
        POST to Jina embeddings endpoint in batches.
        Returns embeddings in the same order as `texts`.
        Raises RuntimeError on non-200 responses with the API error body.
        """
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]

            payload = {
                "model":      self._model,
                "task":       task,
                "dimensions": self._dim,
                "input":      batch,
            }

            resp = httpx.post(
                _EMBED_URL,
                headers=self._headers,
                json=payload,
                timeout=60.0,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Jina embed API error {resp.status_code}: {resp.text[:400]}"
                )

            data = resp.json()
            # Response: {"data": [{"index": i, "embedding": [...], ...}, ...]}
            # Items are returned sorted by index, but sort explicitly to be safe.
            items = sorted(data["data"], key=lambda x: x["index"])
            for item in items:
                all_embeddings.append(np.array(item["embedding"], dtype=np.float32))

        return all_embeddings