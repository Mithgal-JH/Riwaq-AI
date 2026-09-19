import numpy as np

from src.services.vector_store import VectorStore


class SimilarityEngine:
    """
    Computes exact cosine similarity between user profile vectors and candidate post vectors
    using high-performance vectorized NumPy dot products.
    Ensures all scores are bounded within [0.0, 1.0].
    """

    @staticmethod
    def compute_cosine_similarity(
        user_vector: np.ndarray,
        candidate_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a single 1D user vector (384,)
        and a 2D candidate matrix (K, 384).
        Both inputs are assumed to be unit-normalized, making dot product equivalent to cosine similarity.
        Returns 1D array of shape (K,) with scores clipped to [0.0, 1.0].
        """
        if candidate_matrix.size == 0 or user_vector.size == 0:
            return np.empty((0,), dtype=np.float32)

        # Ensure correct shapes and dtypes
        u = np.asarray(user_vector, dtype=np.float32).flatten()
        v = np.asarray(candidate_matrix, dtype=np.float32)

        # Safety check: normalize if norms deviate from 1.0
        u_norm = np.linalg.norm(u)
        if u_norm > 0:
            u = u / u_norm

        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        v = v / v_norms

        # Dot product
        raw_scores = np.dot(v, u)

        # Strictly clamp scores into [0.0, 1.0]
        clamped_scores = np.clip(raw_scores, 0.0, 1.0)
        return clamped_scores.astype(np.float32)

    def retrieve_and_score(
        self,
        user_vector: np.ndarray,
        candidate_ids: list[str],
        vector_store: VectorStore,
    ) -> list[tuple[str, float]]:
        """
        Retrieve candidate vectors from vector store, score against user vector,
        and return (post_id, score) pairs sorted descending by score.
        """
        if not candidate_ids or len(vector_store) == 0:
            return []

        found_ids, candidate_matrix = vector_store.get_vectors_for_candidates(
            candidate_ids
        )
        if not found_ids:
            return []

        scores = self.compute_cosine_similarity(user_vector, candidate_matrix)

        scored_items = [
            (pid, round(float(score), 4)) for pid, score in zip(found_ids, scores)
        ]

        # Sort descending by score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items
