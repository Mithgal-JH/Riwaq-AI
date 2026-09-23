"""
End-to-End Stress and Edge-Case Test Suite for BinX Post Recommendation Engine.
Tests boundary conditions, multi-page pagination (infinite scroll), degenerate metadata,
extreme user profiles, and Content Analysis integration signals.
"""

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.config import MAX_CANDIDATES_PER_REQUEST
from app.main import app
from app.schemas.common import ReasonCode


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """TestClient triggering app lifespan startup warmup."""
    with TestClient(app) as test_client:
        yield test_client


def test_maximum_allowed_candidate_batch_boundary(client: TestClient) -> None:
    """Verify that exactly 100 candidate IDs executes cleanly."""
    candidate_ids = [f"pst_{d}0{i}" for d in range(1, 9) for i in range(1, 6)]
    # Pad to exactly 100 items
    exact_100_candidates = (candidate_ids * 3)[:MAX_CANDIDATES_PER_REQUEST]
    assert len(exact_100_candidates) == 100

    payload = {
        "request_id": "req_boundary_100",
        "user_id": "usr_stress_tester",
        "declared_topics": ["AI_DATA", "PROGRAMMING_WEB"],
        "learning_direction": "High Performance Microservices",
        "eligible_candidate_ids": exact_100_candidates,
        "limit": 10,
    }

    resp = client.post("/api/v1/recommendations/posts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 10
    assert len(data["items"]) == 10
    assert "X-Process-Time-Ms" in resp.headers


def test_infinite_scroll_multi_page_pagination(client: TestClient) -> None:
    """
    Simulate a user scrolling through 3 consecutive pages of feed recommendations.
    Verify that excluded posts from previous pages never reappear on subsequent pages.
    """
    all_candidates = [f"pst_{d}0{i}" for d in range(1, 9) for i in range(1, 6)]
    accumulated_exclusions: list[str] = []
    seen_post_ids: list[str] = []

    # Page 1, Page 2, Page 3
    for page in range(1, 4):
        payload = {
            "request_id": f"req_scroll_page_{page}",
            "user_id": "usr_infinite_scroller",
            "declared_topics": ["PROGRAMMING_WEB", "AI_DATA"],
            "learning_direction": "Full Stack Engineering",
            "eligible_candidate_ids": all_candidates,
            "exclude_post_ids": accumulated_exclusions,
            "limit": 5,
        }

        resp = client.post("/api/v1/recommendations/posts", json=payload)
        assert resp.status_code == 200
        page_items = resp.json()["items"]
        assert len(page_items) <= 5

        page_ids = [item["id"] for item in page_items]
        # Verify no item from this page was seen in any earlier page
        for pid in page_ids:
            assert pid not in seen_post_ids, f"Duplicate post {pid} on page {page}"
            seen_post_ids.append(pid)
            accumulated_exclusions.append(pid)


def test_extreme_cold_start_unseen_domain_direction(client: TestClient) -> None:
    """Verify system handles a learner with novel direction text not found in any mock post."""
    payload = {
        "request_id": "req_novel_direction",
        "user_id": "usr_astrophysics_fan",
        "declared_topics": ["NATURAL_SCIENCES"],
        "learning_direction": "Gravitational Wave Astrophysics and Interferometry",
        "eligible_candidate_ids": ["pst_801", "pst_802", "pst_101", "pst_201"],
        "limit": 3,
    }

    resp = client.post("/api/v1/recommendations/posts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    # Natural science posts should rank first because of topic alignment
    assert data["items"][0]["primary_topic"] in ["Natural Sciences", "NATURAL_SCIENCES"]


def test_dense_warm_learner_profile_stability(client: TestClient) -> None:
    """Verify warm learner profile adapts stably with 10+ divergent historical interactions."""
    dense_interactions = [
        {"post_id": f"pst_10{i}", "interaction_type": "like", "timestamp": "2026-09-15T10:00:00Z"}
        for i in range(1, 5)
    ] + [
        {"post_id": f"pst_20{i}", "interaction_type": "save", "timestamp": "2026-09-17T12:00:00Z"}
        for i in range(1, 5)
    ]

    payload = {
        "request_id": "req_dense_warm",
        "user_id": "usr_power_learner",
        "declared_topics": ["PROGRAMMING_WEB"],
        "learning_direction": "Web Infrastructure",
        "eligible_candidate_ids": ["pst_101", "pst_102", "pst_201", "pst_202", "pst_301"],
        "recent_interactions": dense_interactions,
        "limit": 5,
    }

    resp = client.post("/api/v1/recommendations/posts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 5
    for item in data["items"]:
        assert 0.0 <= item["score"] <= 1.0


def test_post_with_extreme_age_does_not_break_decay(client: TestClient) -> None:
    """Verify an educational post published 2 years ago decays smoothly without zero-division or crash."""
    ancient_post_id = "pst_ancient_2024"
    event_payload = {
        "post_id": ancient_post_id,
        "creator_id": "usr_pioneer",
        "title": "Legacy Python 3.8 Async Patterns",
        "body": "Foundational coroutine syntax introduced in earlier Python releases.",
        "created_at": "2024-01-01T00:00:00Z",
        "topics": {"primary_topics": [{"topic": "PROGRAMMING_WEB", "confidence": 0.95}]},
    }
    upsert_resp = client.post("/api/v1/events/post-upserted", json=event_payload)
    assert upsert_resp.status_code == 200

    rec_payload = {
        "request_id": "req_ancient_test",
        "user_id": "usr_test",
        "declared_topics": ["PROGRAMMING_WEB"],
        "learning_direction": "Python",
        "eligible_candidate_ids": [ancient_post_id, "pst_101"],
        "limit": 2,
    }
    rec_resp = client.post("/api/v1/recommendations/posts", json=rec_payload)
    assert rec_resp.status_code == 200
    items = rec_resp.json()["items"]
    assert len(items) == 2
    for item in items:
        assert 0.0 <= item["score"] <= 1.0

    # The 2-year old post must NEVER receive the FRESH_CONTENT reason code badge
    ancient_item = next(it for it in items if it["id"] == ancient_post_id)
    assert ReasonCode.FRESH_CONTENT.value not in ancient_item["reason_codes"]


def test_all_candidates_filtered_by_safety_returns_empty_cleanly(client: TestClient) -> None:
    """Verify that if every candidate is held by safety moderation, response returns empty list (not 500)."""
    unsafe_id_1 = "pst_unsafe_batch_01"
    unsafe_id_2 = "pst_unsafe_batch_02"

    for uid in [unsafe_id_1, unsafe_id_2]:
        client.post(
            "/api/v1/events/post-upserted",
            json={
                "post_id": uid,
                "creator_id": "usr_bad_actor",
                "title": f"Malicious Guide {uid}",
                "body": "Prohibited phishing and exploit content.",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "safety": {"status": "REVIEW_REQUIRED", "recommendation_signal": "DOWNRANK_OR_HOLD"},
            },
        )

    rec_payload = {
        "request_id": "req_all_unsafe",
        "user_id": "usr_student",
        "declared_topics": ["CYBERSECURITY"],
        "learning_direction": "Defensive Security",
        "eligible_candidate_ids": [unsafe_id_1, unsafe_id_2],
        "limit": 5,
    }

    rec_resp = client.post("/api/v1/recommendations/posts", json=rec_payload)
    assert rec_resp.status_code == 200
    data = rec_resp.json()
    assert data["count"] == 0
    assert data["items"] == []
