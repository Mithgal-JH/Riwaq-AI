from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.config import MAX_CANDIDATES_PER_REQUEST, MODEL_VERSION
from src.main import app
from src.schemas.common import InteractionType, TopicTaxonomy


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """TestClient that triggers app lifespan warmup on startup."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == MODEL_VERSION
    assert data["vectors_count"] >= 40
    assert data["catalog_count"] >= 40
    assert "X-Process-Time-Ms" in response.headers
    process_time = float(response.headers["X-Process-Time-Ms"])
    assert process_time > 0


def test_cold_start_recommendation_request(client: TestClient) -> None:
    all_candidate_ids = [f"pst_{d}0{i}" for d in range(1, 9) for i in range(1, 6)]
    payload = {
        "request_id": "req_cold_01",
        "user_id": "usr_cold_01",
        "declared_topics": [TopicTaxonomy.PROGRAMMING_WEB.value],
        "learning_direction": "Fullstack Web Development",
        "eligible_candidate_ids": all_candidate_ids,
        "exclude_post_ids": [],
        "limit": 5,
        "recent_interactions": None,
    }

    response = client.post("/api/v1/recommendations/posts", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["request_id"] == "req_cold_01"
    assert data["model_version"] == MODEL_VERSION
    assert data["count"] == 5
    assert len(data["items"]) == 5

    # Check structure of each returned item
    for idx, item in enumerate(data["items"], start=1):
        assert item["rank"] == idx
        assert 0.0 <= item["score"] <= 1.0
        assert isinstance(item["reason_codes"], list)
        assert item["primary_topic"] in [t.value for t in TopicTaxonomy]

    # Top items should reflect the declared topic
    top_topics = [item["primary_topic"] for item in data["items"]]
    assert TopicTaxonomy.PROGRAMMING_WEB.value in top_topics


def test_warm_learner_recommendation_request(client: TestClient) -> None:
    all_candidate_ids = [f"pst_{d}0{i}" for d in range(1, 9) for i in range(1, 6)]
    payload = {
        "request_id": "req_warm_01",
        "user_id": "usr_warm_01",
        "declared_topics": [TopicTaxonomy.PROGRAMMING_WEB.value],
        "learning_direction": "Backend Engineering",
        "eligible_candidate_ids": all_candidate_ids,
        "exclude_post_ids": ["pst_101"],
        "limit": 5,
        "recent_interactions": [
            {
                "post_id": "pst_101",
                "interaction_type": InteractionType.LIKE.value,
                "timestamp": datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            }
        ],
    }

    response = client.post("/api/v1/recommendations/posts", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 5
    returned_ids = [item["id"] for item in data["items"]]
    # Excluded post must NOT appear in output
    assert "pst_101" not in returned_ids


def test_exclusion_filters_all_candidates(client: TestClient) -> None:
    payload = {
        "request_id": "req_all_excluded",
        "user_id": "usr_test",
        "declared_topics": [TopicTaxonomy.AI_DATA.value],
        "learning_direction": "Data Science",
        "eligible_candidate_ids": ["pst_001", "pst_002"],
        "exclude_post_ids": ["pst_001", "pst_002"],
        "limit": 5,
    }

    response = client.post("/api/v1/recommendations/posts", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["items"] == []


def test_candidate_batch_size_validation_error(client: TestClient) -> None:
    # Exceeding MAX_CANDIDATES_PER_REQUEST (100) must trigger 422 Unprocessable Entity
    oversized_candidates = [f"pst_{i:04d}" for i in range(MAX_CANDIDATES_PER_REQUEST + 1)]
    payload = {
        "request_id": "req_oversized",
        "user_id": "usr_test",
        "declared_topics": [TopicTaxonomy.AI_DATA.value],
        "learning_direction": "Machine Learning",
        "eligible_candidate_ids": oversized_candidates,
        "limit": 10,
    }

    response = client.post("/api/v1/recommendations/posts", json=payload)
    assert response.status_code == 422


def test_post_upsert_event_and_immediate_recommendation(client: TestClient) -> None:
    new_post_id = "pst_dynamic_999"
    event_payload = {
        "post_id": new_post_id,
        "creator_id": "usr_creator_99",
        "title": "Quantum Computing Fundamentals in Python",
        "body": "Qubits, superposition, and quantum gates simulated with Qiskit.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Ingest post via event endpoint
    upsert_resp = client.post("/api/v1/events/post-upserted", json=event_payload)
    assert upsert_resp.status_code == 200
    upsert_data = upsert_resp.json()
    assert upsert_data["status"] == "success"
    assert upsert_data["post_id"] == new_post_id

    # 2. Immediately request recommendation including the new post
    rec_payload = {
        "request_id": "req_after_upsert",
        "user_id": "usr_quantum_fan",
        "declared_topics": [TopicTaxonomy.PROGRAMMING_WEB.value],
        "learning_direction": "Quantum Algorithms",
        "eligible_candidate_ids": [new_post_id, "pst_002", "pst_003"],
        "limit": 3,
    }

    rec_resp = client.post("/api/v1/recommendations/posts", json=rec_payload)
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()

    returned_ids = [item["id"] for item in rec_data["items"]]
    assert new_post_id in returned_ids
