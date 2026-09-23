"""
Integration tests ensuring the PostUpsertEvent handler in routes.py
correctly aligns with the outputs of Haitham's Content Analysis API
and Zayan's Creator Quality model.
"""

from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.common import TopicTaxonomy
from app.schemas.post import PostRecord

client = TestClient(app)


def test_safety_status_key_alignment():
    """Verify both legacy 'status' and Haitham's 'safety_status' keys are read correctly."""
    # Test 1: Haitham's exact key
    haitham_payload = {
        "post_id": "test_safety_1",
        "creator_id": "user_1",
        "title": "Title 1",
        "body": "Body 1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "safety_status": "REVIEW_REQUIRED",
            "review_required": True,
            "recommendation_signal": "DOWNRANK_OR_HOLD"
        }
    }
    
    resp = client.post("/api/v1/events/post-upserted", json=haitham_payload)
    assert resp.status_code == 200
    
    catalog: dict[str, PostRecord] = app.state.catalog
    post = catalog["test_safety_1"]
    assert post.safety_status == "REVIEW_REQUIRED"
    assert post.recommendation_signal == "DOWNRANK_OR_HOLD"


def test_topics_alignment_primary_secondary_and_scores():
    """Verify Haitham's nested topic structure is parsed correctly."""
    payload = {
        "post_id": "test_topics_1",
        "creator_id": "user_2",
        "title": "Data Science basics",
        "body": "Python and pandas tutorial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topics": {
            "classification_status": "classified",
            "topic_count": 2,
            "primary_topics": [
                {"topic": "AI_DATA", "confidence": 0.95}
            ],
            "secondary_topics": [
                {"topic": "PROGRAMMING_WEB", "confidence": 0.82}
            ],
            "all_scores": {
                "AI_DATA": 0.95,
                "PROGRAMMING_WEB": 0.82,
                "DESIGN": 0.15
            }
        }
    }
    
    resp = client.post("/api/v1/events/post-upserted", json=payload)
    assert resp.status_code == 200
    
    post = app.state.catalog["test_topics_1"]
    
    # Primary extracted
    assert post.primary_topic == TopicTaxonomy.AI_DATA
    
    # Secondary extracted to predicted_topics
    assert TopicTaxonomy.PROGRAMMING_WEB in post.predicted_topics
    assert len(post.predicted_topics) == 1
    
    # all_scores populated
    assert post.topic_confidence_scores[TopicTaxonomy.AI_DATA.value] == 0.95
    assert post.topic_confidence_scores[TopicTaxonomy.PROGRAMMING_WEB.value] == 0.82
    assert post.topic_confidence_scores[TopicTaxonomy.DESIGN.value] == 0.15


def test_unclassified_fallback_behavior():
    """Verify 'unclassified' gracefully falls back without crashing."""
    payload = {
        "post_id": "test_unclassified",
        "creator_id": "user_3",
        "title": "Machine learning",
        "body": "Some text",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topics": {
            "classification_status": "unclassified",
            "primary_topics": [],
            "secondary_topics": [],
            "topic_count": 0,
            "reason_code": "NO_TOPIC_ABOVE_THRESHOLD",
            "all_scores": {
                "AI_DATA": 0.45
            }
        }
    }
    
    resp = client.post("/api/v1/events/post-upserted", json=payload)
    assert resp.status_code == 200
    
    post = app.state.catalog["test_unclassified"]
    assert isinstance(post.primary_topic, TopicTaxonomy)
    # The all_scores should still be captured
    assert post.topic_confidence_scores[TopicTaxonomy.AI_DATA.value] == 0.45


def test_creator_quality_zayan_alignment():
    """Verify Zayan's creator_teaching_quality is read correctly."""
    payload = {
        "post_id": "test_zayan_1",
        "creator_id": "user_4",
        "title": "Title",
        "body": "Body",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "creator_teaching_quality": 0.92
    }
    
    resp = client.post("/api/v1/events/post-upserted", json=payload)
    assert resp.status_code == 200
    
    post = app.state.catalog["test_zayan_1"]
    assert post.creator_teaching_quality == 0.92


def test_new_taxonomy_topics():
    """Verify the 4 newly added topics from Haitham flow through properly."""
    payload = {
        "post_id": "test_new_topic",
        "creator_id": "user_5",
        "title": "Cardiology Basics",
        "body": "Medical study",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topics": {
            "primary_topics": [
                {"topic": "HEALTH_MEDICINE", "confidence": 0.99}
            ]
        }
    }
    
    resp = client.post("/api/v1/events/post-upserted", json=payload)
    assert resp.status_code == 200
    
    post = app.state.catalog["test_new_topic"]
    assert post.primary_topic == TopicTaxonomy.HEALTH_MEDICINE
