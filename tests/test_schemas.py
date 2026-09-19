import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src import config
from src.schemas import (
    InteractionType,
    PostRecord,
    PostUpsertEvent,
    RankedPostItem,
    ReasonCode,
    RecommendationRequest,
    RecommendationResponse,
    TopicTaxonomy,
)


def test_config_weights_and_constants():
    """Verify ranking formula weights sum exactly to 1.00 and constants are valid."""
    total_weight = (
        config.WEIGHT_SEMANTIC
        + config.WEIGHT_TOPIC
        + config.WEIGHT_CREATOR
        + config.WEIGHT_FRESHNESS
    )
    assert pytest.approx(total_weight, rel=1e-5) == 1.00
    assert len(config.TAXONOMY_TOPICS) == 8
    assert config.FRESHNESS_HALF_LIFE_DAYS == 14.0
    assert config.MAX_CANDIDATES_PER_REQUEST == 100
    assert config.DEFAULT_CREATOR_QUALITY == 0.50


def test_valid_recommendation_request():
    """Verify standard valid recommendation request deserializes properly."""
    payload = {
        "request_id": "req_8849b2c1-d41a-4d2a-89a1-8e50b8ef1092",
        "user_id": "usr_learner_100",
        "declared_topics": ["Programming/Web", "AI/Data"],
        "learning_direction": "Backend Development",
        "eligible_candidate_ids": ["pst_101", "pst_102", "pst_201"],
        "exclude_post_ids": ["pst_999"],
        "limit": 5,
        "recent_interactions": [
            {
                "post_id": "pst_101",
                "interaction_type": "like",
                "timestamp": "2026-09-14T10:00:00Z",
            }
        ],
    }

    req = RecommendationRequest(**payload)
    assert req.request_id == "req_8849b2c1-d41a-4d2a-89a1-8e50b8ef1092"
    assert req.user_id == "usr_learner_100"
    assert req.declared_topics == [
        TopicTaxonomy.PROGRAMMING_WEB,
        TopicTaxonomy.AI_DATA,
    ]
    assert len(req.eligible_candidate_ids) == 3
    assert req.limit == 5
    assert req.recent_interactions is not None
    assert req.recent_interactions[0].interaction_type == InteractionType.LIKE


def test_invalid_topic_in_request_fails():
    """Verify requesting an unsupported topic taxonomy fails validation."""
    payload = {
        "request_id": "req_invalid_topic",
        "user_id": "usr_learner_100",
        "declared_topics": ["InvalidNonExistentTopic"],
        "learning_direction": "Astronomy",
        "eligible_candidate_ids": ["pst_101"],
    }
    with pytest.raises(ValidationError) as exc_info:
        RecommendationRequest(**payload)
    assert "declared_topics" in str(exc_info.value)


def test_candidate_batch_size_limit():
    """Verify passing more than 100 candidate IDs raises a validation error."""
    payload = {
        "request_id": "req_too_many_candidates",
        "user_id": "usr_learner_100",
        "declared_topics": ["Programming/Web"],
        "learning_direction": "Backend Development",
        "eligible_candidate_ids": [f"pst_{i}" for i in range(101)],
    }
    with pytest.raises(ValidationError) as exc_info:
        RecommendationRequest(**payload)
    assert "Candidate list exceeds maximum batch size" in str(exc_info.value)


def test_recommendation_response_serialization():
    """Verify recommendation response conforms to Section 5.1 JSON contract."""
    response = RecommendationResponse(
        request_id="req_8849b2c1-d41a-4d2a-89a1-8e50b8ef1092",
        model_version="sentence-transformer-v1.0-all-MiniLM-L6-v2",
        count=3,
        items=[
            RankedPostItem(
                id="pst_101",
                rank=1,
                score=0.865,
                reason_codes=[
                    ReasonCode.SIMILAR_TO_INTERESTS,
                    ReasonCode.TOPIC_MATCH,
                ],
                primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
            ),
            RankedPostItem(
                id="pst_104",
                rank=2,
                score=0.782,
                reason_codes=[ReasonCode.FRESH_CONTENT, ReasonCode.TOPIC_MATCH],
                primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
            ),
            RankedPostItem(
                id="pst_103",
                rank=3,
                score=0.691,
                reason_codes=[ReasonCode.HIGH_RATED_CREATOR],
                primary_topic=TopicTaxonomy.AI_DATA,
            ),
        ],
    )

    data = response.model_dump(mode="json")
    assert data["request_id"] == "req_8849b2c1-d41a-4d2a-89a1-8e50b8ef1092"
    assert data["model_version"] == "sentence-transformer-v1.0-all-MiniLM-L6-v2"
    assert data["count"] == 3
    assert len(data["items"]) == 3
    assert data["items"][0]["id"] == "pst_101"
    assert data["items"][0]["rank"] == 1
    assert data["items"][0]["score"] == 0.865
    assert data["items"][0]["reason_codes"] == [
        "SIMILAR_TO_INTERESTS",
        "TOPIC_MATCH",
    ]
    assert data["items"][0]["primary_topic"] == "Programming/Web"


def test_post_upsert_event_validation():
    """Verify PostUpsertEvent parses backend ingestion event correctly."""
    event_data = {
        "post_id": "pst_new_99",
        "creator_id": "usr_author_12",
        "title": "Quantum Error Correction in Neutral Atom Systems",
        "body": "Detailed explanation of surface codes and logical qubits.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    event = PostUpsertEvent(**event_data)
    assert event.post_id == "pst_new_99"
    assert event.creator_id == "usr_author_12"
    assert "surface codes" in event.body


def test_mock_posts_dataset_integrity():
    """Verify data/mock_posts.json loads and all 40 posts conform to PostRecord schema."""
    assert config.MOCK_POSTS_FILE.exists(), (
        f"Missing mock posts file at {config.MOCK_POSTS_FILE}"
    )

    with open(config.MOCK_POSTS_FILE, "r", encoding="utf-8") as f:
        raw_posts = json.load(f)

    assert len(raw_posts) >= 30, f"Expected at least 30 posts, found {len(raw_posts)}"

    # Parse and validate every post against PostRecord
    validated_posts = [PostRecord(**p) for p in raw_posts]
    assert len(validated_posts) == len(raw_posts)

    # Check topic coverage
    covered_topics = {p.primary_topic for p in validated_posts}
    all_expected_topics = set(TopicTaxonomy)
    assert covered_topics == all_expected_topics, (
        f"Missing topic coverage: {all_expected_topics - covered_topics}"
    )

    # Check teaching quality bounds
    for p in validated_posts:
        assert 0.0 <= p.creator_teaching_quality <= 1.0
        assert len(p.title) > 5
        assert len(p.body) > 20
