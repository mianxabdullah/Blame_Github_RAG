from sentence_transformers import SentenceTransformer

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