from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)