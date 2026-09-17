from typing import cast

import numpy as np

from src.config import (
    EMBEDDING_DIM,
    WARM_LEARNER_DECLARED_WEIGHT,
    WARM_LEARNER_INTERACTION_WEIGHT,
)
from src.schemas.common import TopicTaxonomy
from src.schemas.recommendation import UserInteractionItem
from src.services.embedding_service import EmbeddingService
from src.services.vector_store import VectorStore


class ProfileService:
    """
    Constructs 384-dimensional user interest profile vectors for both:
    1. Cold-Start learners: Centroid of declared onboarding topics & learning direction.
    2. Warm learners: Blended profile combining declared interests (0.60)
       and recent positive interaction vectors (0.40).
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def build_declared_profile(
        self,
        declared_topics: list[TopicTaxonomy],
        learning_direction: str,
    ) -> np.ndarray:
        """
        Compute normalized centroid vector for cold-start learners from declared topics
        and learning direction.
        """
        texts_to_embed: list[str] = []

        for topic in declared_topics:
            topic_str = topic.value if hasattr(topic, "value") else str(topic)
            if topic_str.strip():
                texts_to_embed.append(topic_str.strip())

        if learning_direction and learning_direction.strip():
            texts_to_embed.append(learning_direction.strip())

        if not texts_to_embed:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        vectors = self.embedding_service.embed_batch(texts_to_embed)
        centroid = cast(np.ndarray, np.mean(vectors, axis=0, dtype=np.float32))

        norm = float(np.linalg.norm(centroid))
        if norm < 1e-8:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        return centroid / norm

    def build_user_profile(
        self,
        declared_topics: list[TopicTaxonomy],
        learning_direction: str,
        recent_interactions: list[UserInteractionItem] | None = None,
        vector_store: VectorStore | None = None,
    ) -> np.ndarray:
        """
        Compute learner profile vector.
        If recent interactions and vector_store are provided and have matching post vectors,
        combines declared interests (60%) and interactions (40%) via linear blend.
        Otherwise, returns cold-start declared profile vector.
        """
        v_declared = self.build_declared_profile(declared_topics, learning_direction)

        if not recent_interactions or vector_store is None:
            return v_declared

        # Extract vectors for interacted posts from vector store
        interaction_vectors: list[np.ndarray] = []
        for item in recent_interactions:
            vec = vector_store.get_vector(item.post_id)
            if vec is not None:
                interaction_vectors.append(vec)

        if not interaction_vectors:
            return v_declared

        # Compute centroid of positive interactions
        v_interactions = cast(
            np.ndarray, np.mean(interaction_vectors, axis=0, dtype=np.float32)
        )
        norm_interactions = float(np.linalg.norm(v_interactions))
        if norm_interactions < 1e-8:
            return v_declared

        v_interactions = v_interactions / norm_interactions

        # Combine via declared-interest blend weights (0.60 declared + 0.40 interactions)
        v_combined = (
            WARM_LEARNER_DECLARED_WEIGHT * v_declared
            + WARM_LEARNER_INTERACTION_WEIGHT * v_interactions
        )

        norm_combined = float(np.linalg.norm(v_combined))
        if norm_combined < 1e-8:
            return v_declared

        return v_combined / norm_combined

    # Alias per ANTIGRAVITY_PLAN.md specification
    build_blended_profile = build_user_profile
