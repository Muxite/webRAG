from sentence_transformers import SentenceTransformer
from typing import List

class LocalEmbeddingFunction:
    def __init__(self, local_path='/app/model'):
        """
        Load a SentenceTransformer model from a local directory.
        :param local_path: Directory containing the downloaded model files.
        """
        self.model = SentenceTransformer(local_path)
        self.name = "local-all-MiniLM-L6-v2"
        self.is_legacy = True

    def embed(self, texts) -> List[List[float]]:
        """
        Compute embeddings for a list of texts using the loaded model.
        :param texts: List[str] - strings to encode.
        :return: List[List[float]] - list of embeddings as vectors.
        """
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Make the embedding function directly callable to satisfy Chroma.
        """
        return self.embed(texts)
