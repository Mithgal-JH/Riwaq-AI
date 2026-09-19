from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_DIM, MODEL_NAME
from src.schemas.events import PostUpsertEvent
from src.schemas.post import PostRecord


class EmbeddingService:
    """
    Service wrapper around SentenceTransformer ('all-MiniLM-L6-v2')
    for generating dense 384-dimensional text embeddings.
    Embeddings are L2-normalized to unit vectors for fast cosine similarity dot products.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model on first access to keep service initialization fast."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a normalized 384-dimensional vector.
        """
        if not text or not text.strip():
            # Return zero vector if text is empty, or handle gracefully
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        vector: np.ndarray = self.model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    encode_text = embed_text

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of text strings into an (N, 384) array of normalized vectors.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        cleaned_texts = [t.strip() if t and t.strip() else "" for t in texts]
        vectors: np.ndarray = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def prepare_post_text(post: PostRecord | PostUpsertEvent | dict[str, Any]) -> str:
        """
        Compose post content for embedding: title + body (+ ocr text if available).
        """
        if isinstance(post, dict):
            title = post.get("title", "")
            body = post.get("body", "")
            ocr = post.get("ocr_extracted_text", None)
        else:
            title = post.title
            body = post.body
            ocr = getattr(post, "ocr_extracted_text", None)

        components = [title, body]
        if ocr and ocr.strip():
            components.append(ocr.strip())

        return ". ".join(part.strip() for part in components if part and part.strip())
