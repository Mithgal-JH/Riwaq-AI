from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    EMBEDDING_DIM,
    POST_EMBEDDINGS_FILE,
    POST_METADATA_FILE,
)
from src.schemas.common import InteractionType, TopicTaxonomy
from src.schemas.post import PostRecord
from src.schemas.recommendation import UserInteractionItem
from src.services.embedding_service import EmbeddingService
from src.services.profile_service import ProfileService
from src.services.similarity_engine import SimilarityEngine
from src.services.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    return EmbeddingService()


@pytest.fixture(scope="module")
def loaded_vector_store() -> VectorStore:
    store = VectorStore()
    loaded = store.load_from_disk(POST_EMBEDDINGS_FILE, POST_METADATA_FILE)
    assert loaded, "Failed to load precomputed embeddings from disk."
    return store


# ---------------------------------------------------------------------------
# 1. EmbeddingService Tests
# ---------------------------------------------------------------------------


def test_embedding_dimension_and_norm(embedding_service: EmbeddingService) -> None:
    text = "Understanding PostgreSQL indexing strategies and query plans"
    vec = embedding_service.embed_text(text)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBEDDING_DIM,)
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embedding_batch(embedding_service: EmbeddingService) -> None:
    texts = [
        "Python asynchronous programming with asyncio",
        "Deep learning transformer attention mechanisms",
        "Robotics kinematics and motion planning",
    ]
    matrix = embedding_service.embed_batch(texts)

    assert matrix.shape == (3, EMBEDDING_DIM)
    for i in range(3):
        norm = float(np.linalg.norm(matrix[i]))
        assert norm == pytest.approx(1.0, abs=1e-5)


def test_embedding_empty_text(embedding_service: EmbeddingService) -> None:
    vec = embedding_service.embed_text("")
    assert vec.shape == (EMBEDDING_DIM,)
    assert np.all(vec == 0.0)


def test_prepare_post_text() -> None:
    record = PostRecord(
        id="test_1",
        creator_id="usr_1",
        title="Sample Title",
        body="Sample Body Content",
        created_at=datetime.now(timezone.utc),
        primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
        ocr_extracted_text="Code snippet OCR text",
    )
    text = EmbeddingService.prepare_post_text(record)
    assert "Sample Title" in text
    assert "Sample Body Content" in text
    assert "Code snippet OCR text" in text


# ---------------------------------------------------------------------------
# 2. VectorStore Tests
# ---------------------------------------------------------------------------


def test_vector_store_in_memory_operations() -> None:
    store = VectorStore()
    assert len(store) == 0

    dummy_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    store.upsert("post_dummy_1", dummy_vec)

    assert len(store) == 1
    assert store.contains("post_dummy_1")
    retrieved = store.get_vector("post_dummy_1")
    assert retrieved is not None
    assert float(np.linalg.norm(retrieved)) == pytest.approx(1.0, abs=1e-5)


def test_vector_store_disk_persistence(tmp_path: Path) -> None:
    store = VectorStore()
    vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    store.upsert("p1", vec1)
    store.upsert("p2", vec2)

    emb_path = tmp_path / "test_emb.npy"
    meta_path = tmp_path / "test_meta.json"
    store.save_to_disk(emb_path, meta_path)

    assert emb_path.exists()
    assert meta_path.exists()

    new_store = VectorStore()
    loaded = new_store.load_from_disk(emb_path, meta_path)
    assert loaded is True
    assert len(new_store) == 2
    assert new_store.contains("p1")
    assert new_store.contains("p2")

    np.testing.assert_allclose(
        new_store.get_vector("p1"), store.get_vector("p1"), rtol=1e-5
    )


def test_candidate_vector_extraction(loaded_vector_store: VectorStore) -> None:
    candidates = ["pst_101", "pst_102", "pst_non_existent"]
    found_ids, matrix = loaded_vector_store.get_vectors_for_candidates(candidates)

    assert found_ids == ["pst_101", "pst_102"]
    assert matrix.shape == (2, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# 3. ProfileService Tests
# ---------------------------------------------------------------------------


def test_cold_start_profile(embedding_service: EmbeddingService) -> None:
    profile_service = ProfileService(embedding_service)
    vec = profile_service.build_declared_profile(
        declared_topics=[TopicTaxonomy.PROGRAMMING_WEB, TopicTaxonomy.AI_DATA],
        learning_direction="Backend APIs with Python",
    )

    assert vec.shape == (EMBEDDING_DIM,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_warm_learner_profile_with_interactions(
    embedding_service: EmbeddingService,
    loaded_vector_store: VectorStore,
) -> None:
    profile_service = ProfileService(embedding_service)
    declared_topics = [TopicTaxonomy.PROGRAMMING_WEB]
    direction = "Web Development"

    # Interacted with two posts
    interactions = [
        UserInteractionItem(
            post_id="pst_101",
            interaction_type=InteractionType.LIKE,
            timestamp=datetime.now(timezone.utc),
        ),
        UserInteractionItem(
            post_id="pst_102",
            interaction_type=InteractionType.SAVE,
            timestamp=datetime.now(timezone.utc),
        ),
    ]

    v_warm = profile_service.build_user_profile(
        declared_topics=declared_topics,
        learning_direction=direction,
        recent_interactions=interactions,
        vector_store=loaded_vector_store,
    )

    assert v_warm.shape == (EMBEDDING_DIM,)
    assert float(np.linalg.norm(v_warm)) == pytest.approx(1.0, abs=1e-5)

    # Verify warm profile is different from cold start due to interaction blend
    v_cold = profile_service.build_declared_profile(declared_topics, direction)
    cosine_sim = float(np.dot(v_warm, v_cold))
    assert 0.0 <= cosine_sim <= 1.0


def test_warm_learner_degenerate_norm_fallback(
    embedding_service: EmbeddingService,
) -> None:
    profile_service = ProfileService(embedding_service)
    declared_topics = [TopicTaxonomy.PROGRAMMING_WEB]
    direction = "Python Web"
    v_declared = profile_service.build_declared_profile(declared_topics, direction)

    # Mock vector store that returns opposing vector to simulate near-zero cancellation:
    # 0.60 * v_declared + 0.40 * (-1.5 * v_declared) = 0.0
    class DegenerateVectorStore(VectorStore):
        def get_vector(self, post_id: str) -> np.ndarray | None:
            return (-1.5 * v_declared).astype(np.float32)

    deg_store = DegenerateVectorStore()
    interactions = [
        UserInteractionItem(
            post_id="pst_cancel",
            interaction_type=InteractionType.LIKE,
            timestamp=datetime.now(timezone.utc),
        )
    ]

    v_result = profile_service.build_user_profile(
        declared_topics=declared_topics,
        learning_direction=direction,
        recent_interactions=interactions,
        vector_store=deg_store,
    )

    # Safely falls back to v_declared rather than dividing by zero or producing NaNs
    assert np.allclose(v_result, v_declared, atol=1e-5)
    assert float(np.linalg.norm(v_result)) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. SimilarityEngine & Ranking Relevance Tests
# ---------------------------------------------------------------------------


def test_cosine_similarity_bounds() -> None:
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_candidates = np.array(
        [
            [1.0, 0.0, 0.0],  # Identical -> score 1.0
            [0.0, 1.0, 0.0],  # Orthogonal -> score 0.0
            [-1.0, 0.0, 0.0],  # Opposite -> clipped to 0.0
        ],
        dtype=np.float32,
    )

    scores = SimilarityEngine.compute_cosine_similarity(u, v_candidates)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert scores[1] == pytest.approx(0.0, abs=1e-4)
    assert scores[2] == pytest.approx(0.0, abs=1e-4)


def test_semantic_retrieval_relevance(
    embedding_service: EmbeddingService,
    loaded_vector_store: VectorStore,
) -> None:
    """
    Learner interested in 'Asyncio & Python Backend' should get higher semantic
    fit for Programming/Web posts than for unrelated Natural Sciences or Robotics posts.
    """
    profile_service = ProfileService(embedding_service)
    similarity_engine = SimilarityEngine()

    user_vec = profile_service.build_declared_profile(
        declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
        learning_direction="Python Asyncio Event Loops and High Performance Web APIs",
    )

    all_ids = list(loaded_vector_store._vectors.keys())
    ranked = similarity_engine.retrieve_and_score(
        user_vector=user_vec,
        candidate_ids=all_ids,
        vector_store=loaded_vector_store,
    )

    assert len(ranked) == 40
    top_post_id, top_score = ranked[0]
    top_record = loaded_vector_store.get_record(top_post_id)
    assert top_record is not None

    # Top retrieved post must be related to Programming/Web
    assert top_record.primary_topic == TopicTaxonomy.PROGRAMMING_WEB
    assert top_score > 0.50

    # Ensure ranking is strictly descending
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i + 1][1]
