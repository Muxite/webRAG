from sentence_transformers import SentenceTransformer
from typing import List
import os


class LocalEmbeddingFunction:
    """ChromaDB-compatible embedding function using local SentenceTransformer model."""

    def __init__(self, local_path: str = '/app/model'):
        """
        Load SentenceTransformer model from local directory.
        
        :param local_path: Directory with saved model files. Defaults to '/app/model'.
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model directory not found: {local_path}")
        self.model = SentenceTransformer(local_path)
        self.name = "local-all-MiniLM-L6-v2"
        self.is_legacy = True

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for texts.
        
        :param texts: List of strings to encode.
        :return: List of embedding vectors (list of floats).
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Callable wrapper for ChromaDB compatibility."""
        return self.embed(texts)
