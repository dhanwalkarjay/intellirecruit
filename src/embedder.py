"""
embedder.py — Thin wrapper around SentenceTransformer.
All embeddings are L2-normalised so dot-product == cosine similarity.
No API calls — model runs entirely locally.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class Embedder:
    """
    Singleton-style wrapper. Instantiate once, reuse across the pipeline.
    Model is downloaded on first use and cached in ~/.cache/torch/sentence_transformers/
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string.
        Returns: float32 ndarray of shape (384,), L2-normalised.
        """
        model = self._load()
        vec = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 512,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings efficiently.
        Returns: float32 ndarray of shape (N, 384), each row L2-normalised.
        """
        model = self._load()
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Since both are normalised, this is just the dot product.
        """
        return float(np.dot(a, b))
