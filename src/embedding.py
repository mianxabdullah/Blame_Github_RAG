from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from langchain_core.documents import Document

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.embedding_dim: int | None = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"[INFO] Successfully loaded model: {self.model_name}\nModel dimensions: {self.embedding_dim}")
        except Exception as e:
            print(f"[ERROR] Error loading model {self.model_name}: {e}")
            self.model = None

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of raw strings. Returns array of shape (len(texts), embedding_dim)."""
        if not self.model:
            raise ValueError("[ERROR] Model is not loaded. Cannot generate embeddings.")
        if not texts:
            print("[INFO] No texts provided, returning empty array.")
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # unit-length vectors -> cosine similarity == dot product
            )
            print(f"[INFO] Generated embeddings for {len(texts)} texts. Shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"[ERROR] Error generating embeddings: {e}")
            return np.array([])

    def embed_documents(self, chunks: List[Document], batch_size: int = 32) -> np.ndarray:
        """Convenience wrapper: extract page_content from Documents and embed them."""
        texts = [chunk.page_content for chunk in chunks]
        return self.generate_embeddings(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query."""
        return self.generate_embeddings([query])

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("[ERROR] Model is not loaded. Cannot get embedding dimension.")
        return self.embedding_dim