from datetime import datetime, timedelta, timezone

from src.config import (
    HIGH_RATED_CREATOR_THRESHOLD,
    REASON_FRESH_THRESHOLD,
    REASON_SEMANTIC_THRESHOLD,
)
from src.schemas.common import InteractionType, ReasonCode, TopicTaxonomy
from src.schemas.post import PostRecord
from src.schemas.recommendation import UserInteractionItem
from src.services.diversity_service import (
    DiversityService,
    apply_topic_frequency_capping,
    filter_excluded_candidates,
)
from src.services.ranking_engine import ScoreBreakdown
from src.services.reason_service import (
    ReasonService,
    generate_reason_codes_for_candidate,
)


def make_breakdown(
    post_id: str,
    topic: TopicTaxonomy,
    score: float = 0.8,
    semantic: float = 0.7,
    topic_score: float = 0.5,
    creator_score: float = 0.5,
    freshness_score: float = 0.5,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        post_id=post_id,
        final_score=score,
        semantic_score=semantic,
        topic_score=topic_score,
        creator_score=creator_score,
        freshness_score=freshness_score,
        primary_topic=topic,
    )


# ---------------------------------------------------------------------------
# Exclusion and Deduplication Tests
# ---------------------------------------------------------------------------


def test_filter_excluded_candidates() -> None:
    candidates = ["p1", "p2", "p3", "p4", "p5"]
    excluded = ["p2", "p4"]

    filtered = filter_excluded_candidates(candidates, excluded)
    assert filtered == ["p1", "p3", "p5"]


def test_filter_candidates_deduplication() -> None:
    candidates = ["p1", "p2", "p1", "p3", "p2", "p4"]
    filtered = filter_excluded_candidates(candidates)
    assert filtered == ["p1", "p2", "p3", "p4"]


def test_filter_candidates_empty_and_none() -> None:
    assert filter_excluded_candidates([]) == []
    assert filter_excluded_candidates(["p1", "p2"], None) == ["p1", "p2"]
    assert filter_excluded_candidates(["p1", "p2"], []) == ["p1", "p2"]
    assert filter_excluded_candidates(["p1", "p2"], ["p1", "p2"]) == []


def test_diversity_service_filter_exclusions() -> None:
    svc = DiversityService()
    result = svc.filter_exclusions(["p1", "p2", "p3"], ["p1"])
    assert result == ["p2", "p3"]


# ---------------------------------------------------------------------------
# Topic Frequency Capping Tests
# ---------------------------------------------------------------------------


def test_topic_capping_under_limit() -> None:
    # 2 Programming, 1 AI, 1 Programming -> streak never reaches 3
    items = [
        make_breakdown("p1", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p2", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p3", TopicTaxonomy.AI_DATA),
        make_breakdown("p4", TopicTaxonomy.PROGRAMMING_WEB),
    ]

    reordered = apply_topic_frequency_capping(items, max_consecutive=3)
    assert [x.post_id for x in reordered] == ["p1", "p2", "p3", "p4"]


def test_topic_capping_exact_limit() -> None:
    # 3 Programming posts -> exactly reaches max_consecutive 3, no violation
    items = [
        make_breakdown("p1", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p2", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p3", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p4", TopicTaxonomy.AI_DATA),
    ]

    reordered = apply_topic_frequency_capping(items, max_consecutive=3)
    assert [x.post_id for x in reordered] == ["p1", "p2", "p3", "p4"]


def test_topic_capping_exceeding_limit_pulls_alternate() -> None:
    # 4 Programming posts followed by 1 AI post
    # At index 3, streak is 3 -> p5 (AI) must be pulled up before p4
    items = [
        make_breakdown("p1", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p2", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p3", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p4", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("p5", TopicTaxonomy.AI_DATA),
    ]

    reordered = apply_topic_frequency_capping(items, max_consecutive=3)
    # Expected: p1, p2, p3, p5 (alternate pulled forward), p4
    assert [x.post_id for x in reordered] == ["p1", "p2", "p3", "p5", "p4"]
    # Check that total items and elements are strictly preserved
    assert len(reordered) == len(items)
    assert {x.post_id for x in reordered} == {x.post_id for x in items}


def test_topic_capping_multiple_violations() -> None:
    # A, A, A, A, A, B, C
    # -> A, A, A, B, A, A, C
    items = [
        make_breakdown("a1", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("a2", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("a3", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("a4", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("a5", TopicTaxonomy.PROGRAMMING_WEB),
        make_breakdown("b1", TopicTaxonomy.AI_DATA),
        make_breakdown("c1", TopicTaxonomy.CYBERSECURITY),
    ]

    reordered = apply_topic_frequency_capping(items, max_consecutive=3)
    assert [x.post_id for x in reordered] == ["a1", "a2", "a3", "b1", "a4", "a5", "c1"]


def test_topic_capping_all_same_topic_does_not_drop_items() -> None:
    # When all posts share the same topic, none can be swapped.
    # Algorithm must NOT discard posts or hang in infinite loop.
    items = [
        make_breakdown(f"p{i}", TopicTaxonomy.PROGRAMMING_WEB) for i in range(6)
    ]

    reordered = apply_topic_frequency_capping(items, max_consecutive=3)
    assert len(reordered) == 6
    assert [x.post_id for x in reordered] == [f"p{i}" for i in range(6)]


def test_topic_capping_empty_and_single() -> None:
    assert apply_topic_frequency_capping([]) == []
    single = [make_breakdown("p1", TopicTaxonomy.DESIGN)]
    assert apply_topic_frequency_capping(single) == single


def test_diversity_service_apply_capping() -> None:
    svc = DiversityService(max_consecutive_same_topic=2)
    items = [
        make_breakdown("p1", TopicTaxonomy.ROBOTICS),
        make_breakdown("p2", TopicTaxonomy.ROBOTICS),
        make_breakdown("p3", TopicTaxonomy.ROBOTICS),
        make_breakdown("p4", TopicTaxonomy.MATHEMATICS),
    ]
    result = svc.apply_capping(items)
    assert [x.post_id for x in result] == ["p1", "p2", "p4", "p3"]


# ---------------------------------------------------------------------------
# Reason Code Generation Tests
# ---------------------------------------------------------------------------


def test_reason_code_similar_to_interests_by_score() -> None:
    # Semantic score >= 0.75 triggers SIMILAR_TO_INTERESTS
    b = make_breakdown(
        "p1",
        TopicTaxonomy.AI_DATA,
        semantic=REASON_SEMANTIC_THRESHOLD,
        topic_score=0.0,
        creator_score=0.5,
        freshness_score=0.5,
    )
    codes = generate_reason_codes_for_candidate(b)
    assert ReasonCode.SIMILAR_TO_INTERESTS in codes
    assert ReasonCode.TOPIC_MATCH not in codes


def test_reason_code_similar_to_interests_by_interaction_history() -> None:
    # Semantic score low (0.30), but post_id matches recent interaction
    b = make_breakdown(
        "p_interacted",
        TopicTaxonomy.AI_DATA,
        semantic=0.30,
        topic_score=0.0,
        creator_score=0.5,
        freshness_score=0.5,
    )
    interactions = [
        UserInteractionItem(
            post_id="p_interacted",
            interaction_type=InteractionType.LIKE,
            timestamp=datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
    ]
    codes = generate_reason_codes_for_candidate(
        b, recent_interactions=interactions
    )
    assert ReasonCode.SIMILAR_TO_INTERESTS in codes


def test_reason_code_topic_match_declared() -> None:
    b = make_breakdown(
        "p1",
        TopicTaxonomy.CYBERSECURITY,
        semantic=0.40,
        topic_score=1.0,
        creator_score=0.5,
        freshness_score=0.5,
    )
    codes = generate_reason_codes_for_candidate(
        b, declared_topics=[TopicTaxonomy.CYBERSECURITY]
    )
    assert ReasonCode.TOPIC_MATCH in codes
    assert ReasonCode.SIMILAR_TO_INTERESTS not in codes


def test_reason_code_fresh_content_by_score() -> None:
    b = make_breakdown(
        "p1",
        TopicTaxonomy.DESIGN,
        semantic=0.40,
        topic_score=0.0,
        creator_score=0.5,
        freshness_score=REASON_FRESH_THRESHOLD,
    )
    codes = generate_reason_codes_for_candidate(b)
    assert ReasonCode.FRESH_CONTENT in codes


def test_reason_code_fresh_content_by_timestamp() -> None:
    ref_time = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
    post_time = ref_time - timedelta(hours=48)  # 2 days old (<= 72 hrs)

    post = PostRecord(
        id="p1",
        creator_id="usr_01",
        title="Modern UI Design",
        body="Principles of UI Design.",
        created_at=post_time,
        primary_topic=TopicTaxonomy.DESIGN,
    )
    b = make_breakdown(
        "p1",
        TopicTaxonomy.DESIGN,
        semantic=0.30,
        topic_score=0.0,
        creator_score=0.5,
        freshness_score=0.70,  # below score threshold
    )
    codes = generate_reason_codes_for_candidate(
        b, post=post, reference_time=ref_time
    )
    assert ReasonCode.FRESH_CONTENT in codes


def test_reason_code_high_rated_creator() -> None:
    b = make_breakdown(
        "p1",
        TopicTaxonomy.ROBOTICS,
        semantic=0.30,
        topic_score=0.0,
        creator_score=HIGH_RATED_CREATOR_THRESHOLD,
        freshness_score=0.5,
    )
    codes = generate_reason_codes_for_candidate(b)
    assert ReasonCode.HIGH_RATED_CREATOR in codes


def test_reason_code_all_flags_triggered() -> None:
    b = make_breakdown(
        "p_star",
        TopicTaxonomy.PROGRAMMING_WEB,
        semantic=0.92,
        topic_score=1.0,
        creator_score=0.95,
        freshness_score=0.98,
    )
    codes = generate_reason_codes_for_candidate(
        b, declared_topics=[TopicTaxonomy.PROGRAMMING_WEB]
    )
    assert len(codes) == 4
    assert set(codes) == {
        ReasonCode.SIMILAR_TO_INTERESTS,
        ReasonCode.TOPIC_MATCH,
        ReasonCode.FRESH_CONTENT,
        ReasonCode.HIGH_RATED_CREATOR,
    }


def test_reason_service_class() -> None:
    svc = ReasonService()
    b = make_breakdown(
        "p1",
        TopicTaxonomy.MATHEMATICS,
        semantic=0.20,
        topic_score=0.0,
        creator_score=0.40,
        freshness_score=0.20,
    )
    codes = svc.get_reason_codes(b)
    assert codes == []
