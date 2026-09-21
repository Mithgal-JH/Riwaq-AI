import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from app.config import (
    DEFAULT_CREATOR_QUALITY,
    MOCK_POSTS_FILE,
    TOPIC_SECONDARY_CREDIT,
    WEIGHT_CREATOR,
    WEIGHT_FRESHNESS,
    WEIGHT_SEMANTIC,
    WEIGHT_TOPIC,
)
from app.schemas.common import TopicTaxonomy
from app.schemas.post import PostRecord
from app.services.ranking_engine import (
    RankingEngine,
    ScoreBreakdown,
    compute_composite_score,
    compute_creator_score,
    compute_freshness_score,
    compute_semantic_score,
    compute_topic_score,
)


@pytest.fixture
def sample_post() -> PostRecord:
    return PostRecord(
        id="pst_test_01",
        creator_id="usr_creator_01",
        title="Async Python Concurrency",
        body="Learn asyncio event loops, coroutines, and tasks in Python.",
        created_at=datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc),
        primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
        predicted_topics=[TopicTaxonomy.AI_DATA],
        topic_confidence_scores={"AI/Data": 0.80},
        creator_teaching_quality=0.85,
    )


@pytest.fixture
def mock_catalog() -> dict[str, PostRecord]:
    with open(MOCK_POSTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: PostRecord.model_validate(item) for item in data}


def test_weights_sum_to_one() -> None:
    total = WEIGHT_SEMANTIC + WEIGHT_TOPIC + WEIGHT_CREATOR + WEIGHT_FRESHNESS
    assert math.isclose(total, 1.0, rel_tol=1e-6)


def test_semantic_score_clamping() -> None:
    assert compute_semantic_score(0.75) == 0.75
    assert compute_semantic_score(-0.3) == 0.0
    assert compute_semantic_score(1.5) == 1.0
    assert compute_semantic_score(float("nan")) == 0.0


def test_topic_score_exact_primary_match(sample_post: PostRecord) -> None:
    score = compute_topic_score(
        sample_post,
        [TopicTaxonomy.PROGRAMMING_WEB, TopicTaxonomy.CYBERSECURITY],
    )
    assert score == 1.0


def test_topic_score_containment_secondary_match(sample_post: PostRecord) -> None:
    # Post topics = {Programming/Web, AI/Data} (len=2)
    # Declared = {AI/Data} -> overlap=1 -> ratio = 1/2 = 0.5
    # score = TOPIC_SECONDARY_CREDIT * 0.5 = 0.60 * 0.5 = 0.30
    score = compute_topic_score(
        sample_post,
        [TopicTaxonomy.AI_DATA],
    )
    expected = TOPIC_SECONDARY_CREDIT * 0.50
    assert math.isclose(score, expected, rel_tol=1e-4)


def test_topic_score_no_match(sample_post: PostRecord) -> None:
    score = compute_topic_score(
        sample_post,
        [TopicTaxonomy.DESIGN, TopicTaxonomy.NATURAL_SCIENCES],
    )
    assert score == 0.0


def test_topic_score_empty_declared(sample_post: PostRecord) -> None:
    assert compute_topic_score(sample_post, []) == 0.0


def test_creator_score(sample_post: PostRecord) -> None:
    assert compute_creator_score(sample_post) == 0.85


def test_creator_score_fallback() -> None:
    post_unrated = PostRecord(
        id="pst_unrated",
        creator_id="usr_02",
        title="Test Title",
        body="Test Body",
        created_at=datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc),
        primary_topic=TopicTaxonomy.ROBOTICS,
    )
    assert compute_creator_score(post_unrated) == DEFAULT_CREATOR_QUALITY


def test_true_half_life_freshness_decay() -> None:
    ref_time = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)

    # 0 days elapsed -> freshness 1.0
    created_now = ref_time
    assert math.isclose(
        compute_freshness_score(created_now, ref_time), 1.0, rel_tol=1e-5
    )

    # Future post anomaly -> clamped to 1.0
    created_future = ref_time + timedelta(days=2)
    assert compute_freshness_score(created_future, ref_time) == 1.0

    # Exactly 14 days elapsed (one true half-life) -> exactly 0.50
    created_14d_ago = ref_time - timedelta(days=14)
    assert math.isclose(
        compute_freshness_score(created_14d_ago, ref_time), 0.50, rel_tol=1e-4
    )

    # Exactly 28 days elapsed (two half-lives) -> exactly 0.25
    created_28d_ago = ref_time - timedelta(days=28)
    assert math.isclose(
        compute_freshness_score(created_28d_ago, ref_time), 0.25, rel_tol=1e-4
    )


def test_freshness_timezone_handling() -> None:
    created_aware = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    ref_naive = datetime(2026, 9, 12, 12, 0, 0)  # noqa: DTZ001
    score = compute_freshness_score(created_aware, ref_naive)
    assert 0.0 <= score <= 1.0


def test_composite_score_formula() -> None:
    # 0.70 * 0.80 + 0.15 * 1.00 + 0.10 * 0.90 + 0.05 * 0.50
    # = 0.56 + 0.15 + 0.09 + 0.025 = 0.8250
    score = compute_composite_score(
        semantic_score=0.80,
        topic_score=1.00,
        creator_score=0.90,
        freshness_score=0.50,
    )
    assert score == 0.8250
    assert 0.0 <= score <= 1.0


def test_ranking_engine_candidate_ordering(sample_post: PostRecord) -> None:
    post_low = PostRecord(
        id="pst_test_02",
        creator_id="usr_creator_02",
        title="Intro to Biology",
        body="Cellular respiration and photosynthesis.",
        created_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc),
        primary_topic=TopicTaxonomy.NATURAL_SCIENCES,
        creator_teaching_quality=0.40,
    )

    catalog = {
        sample_post.id: sample_post,
        post_low.id: post_low,
    }

    engine = RankingEngine(catalog)

    candidate_ids = [post_low.id, sample_post.id]
    semantic_scores = {
        sample_post.id: 0.88,
        post_low.id: 0.12,
    }
    declared_topics = [TopicTaxonomy.PROGRAMMING_WEB]

    results: list[ScoreBreakdown] = engine.rank_candidates(
        candidate_ids=candidate_ids,
        semantic_scores=semantic_scores,
        declared_topics=declared_topics,
        reference_time=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert len(results) == 2
    assert results[0].post_id == sample_post.id
    assert results[1].post_id == post_low.id
    assert results[0].final_score > results[1].final_score
    assert results[0].primary_topic == TopicTaxonomy.PROGRAMMING_WEB


def test_ranking_engine_skips_missing_catalog_posts(sample_post: PostRecord) -> None:
    catalog = {sample_post.id: sample_post}
    engine = RankingEngine(catalog)

    results = engine.rank_candidates(
        candidate_ids=["non_existent_id", sample_post.id],
        semantic_scores={sample_post.id: 0.90},
        declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
    )

    assert len(results) == 1
    assert results[0].post_id == sample_post.id


def test_end_to_end_with_mock_catalog(mock_catalog: dict[str, PostRecord]) -> None:
    assert len(mock_catalog) == 40

    engine = RankingEngine(mock_catalog)

    all_candidate_ids = list(mock_catalog.keys())
    semantic_scores = {
        pid: (0.85 if post.primary_topic == TopicTaxonomy.PROGRAMMING_WEB else 0.20)
        for pid, post in mock_catalog.items()
    }

    ranked = engine.rank_candidates(
        candidate_ids=all_candidate_ids,
        semantic_scores=semantic_scores,
        declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
        reference_time=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert len(ranked) == 40
    for i in range(len(ranked) - 1):
        assert ranked[i].final_score >= ranked[i + 1].final_score

    assert ranked[0].primary_topic == TopicTaxonomy.PROGRAMMING_WEB
    assert ranked[0].topic_score == 1.0
    assert ranked[0].final_score > 0.70


def test_ranking_engine_safety_gate_exclusion(sample_post: PostRecord) -> None:
    """Verify posts with DOWNRANK_OR_HOLD or REVIEW_REQUIRED are excluded by safety gate."""
    safe_post = sample_post
    unsafe_post = PostRecord(
        id="pst_unsafe_99",
        creator_id="usr_creator_99",
        title="Harmful Content Guide",
        body="Content that violates platform safety policy.",
        created_at=datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
        primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
        safety_status="REVIEW_REQUIRED",
        recommendation_signal="DOWNRANK_OR_HOLD",
    )

    catalog = {safe_post.id: safe_post, unsafe_post.id: unsafe_post}
    engine = RankingEngine(catalog)

    ranked = engine.rank_candidates(
        candidate_ids=[safe_post.id, unsafe_post.id],
        semantic_scores={safe_post.id: 0.90, unsafe_post.id: 0.99},
        declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
    )

    # Only safe_post should pass through, despite unsafe_post having higher semantic score
    assert len(ranked) == 1
    assert ranked[0].post_id == safe_post.id
