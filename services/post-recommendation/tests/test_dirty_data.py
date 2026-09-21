"""
Dirty Data Stress Test Suite for BinX Post Recommendation Engine.

Uses mock_posts_dirty.json to exercise code paths the clean mock data never reaches:
 - Skewed topic distribution forcing diversity capping under pressure
 - Secondary-only topic matches (primary != declared, but predicted_topics overlap)
 - Safety-flagged posts excluded from ranking
 - Extreme creator scores (0.0 and 1.0)
 - Missing creator_teaching_quality field (default fallback)
 - Very old (2019) and future-dated (2030) posts
 - Empty predicted_topics
 - Posts with difficulty metadata
 - OCR text stored but not used in embeddings (verifies no crash)
 - Mixed safe and unsafe candidates in the same request
"""

import json
import math
from datetime import datetime, timezone

import pytest

from app.config import (
    DEFAULT_CREATOR_QUALITY,
    FRESHNESS_HALF_LIFE_DAYS,
    HIGH_RATED_CREATOR_THRESHOLD,
    MAX_CONSECUTIVE_SAME_TOPIC,
    TOPIC_SECONDARY_CREDIT,
    WEIGHT_CREATOR,
    WEIGHT_FRESHNESS,
    WEIGHT_SEMANTIC,
    WEIGHT_TOPIC,
)
from app.schemas.common import ReasonCode, TopicTaxonomy
from app.schemas.post import PostRecord
from app.services.diversity_service import DiversityService, apply_topic_frequency_capping
from app.services.ranking_engine import (
    RankingEngine,
    ScoreBreakdown,
    compute_composite_score,
    compute_creator_score,
    compute_freshness_score,
    compute_topic_score,
)
from app.services.reason_service import ReasonService, generate_reason_codes_for_candidate

# ─── Fixture: load dirty catalog ───────────────────────────────────────────────

DIRTY_DATA_PATH = "data/mock_posts_dirty.json"


@pytest.fixture
def dirty_catalog() -> dict[str, PostRecord]:
    """Load the dirty mock data catalog."""
    with open(DIRTY_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: PostRecord.model_validate(item) for item in data}


@pytest.fixture
def ranking_engine(dirty_catalog: dict[str, PostRecord]) -> RankingEngine:
    return RankingEngine(dirty_catalog)


@pytest.fixture
def diversity_service() -> DiversityService:
    return DiversityService(max_consecutive_same_topic=MAX_CONSECUTIVE_SAME_TOPIC)


@pytest.fixture
def reason_service() -> ReasonService:
    return ReasonService()


# ─── 1. Data Loading and Schema Validation ─────────────────────────────────────


class TestDirtyDataLoading:
    """Verify all dirty records parse without errors and field defaults apply."""

    def test_all_records_load(self, dirty_catalog: dict[str, PostRecord]) -> None:
        assert len(dirty_catalog) == 25

    def test_missing_creator_quality_gets_default(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """Post 'dirty_no_creator_score' omits creator_teaching_quality -- should get 0.50."""
        post = dirty_catalog["dirty_no_creator_score"]
        assert post.creator_teaching_quality == DEFAULT_CREATOR_QUALITY

    def test_zero_creator_quality_is_valid(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_zero_creator"]
        assert post.creator_teaching_quality == 0.0

    def test_max_creator_quality_is_valid(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_max_creator"]
        assert post.creator_teaching_quality == 1.0

    def test_empty_predicted_topics(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_no_predicted"]
        assert post.predicted_topics == []
        assert post.topic_confidence_scores == {}

    def test_safety_fields_loaded(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        flagged = dirty_catalog["dirty_unsafe_flagged"]
        assert flagged.safety_status == "REVIEW_REQUIRED"
        assert flagged.recommendation_signal == "DOWNRANK_OR_HOLD"

        safe = dirty_catalog["dirty_safe_cyber"]
        assert safe.safety_status == "SAFE"
        assert safe.recommendation_signal == "ALLOW"

    def test_difficulty_fields_loaded(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_with_difficulty"]
        assert post.difficulty_level == "INTERMEDIATE"
        assert post.difficulty_confidence == 0.88

    def test_ocr_text_stored(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_ocr_heavy"]
        assert post.ocr_extracted_text is not None
        assert "AVL rotation" in post.ocr_extracted_text


# ─── 2. Topic Scoring: Secondary-Only Matches ──────────────────────────────────


class TestSecondaryTopicScoring:
    """
    These posts have a primary_topic that differs from the user's declared topics,
    but their predicted_topics DO overlap. The clean mock data never triggers
    the secondary credit branch.
    """

    def test_secondary_credit_when_primary_mismatches(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """
        dirty_secondary_only: primary=Electronics/Embedded, predicted=[AI/Data, Mathematics]
        User declares [AI/Data] -> primary doesn't match -> secondary path.
        all_post_topics = {Electronics/Embedded, AI/Data, Mathematics} (3 topics)
        overlap with {AI/Data} = 1 topic
        score = TOPIC_SECONDARY_CREDIT * (1/3)
        """
        post = dirty_catalog["dirty_secondary_only"]
        score = compute_topic_score(post, [TopicTaxonomy.AI_DATA])

        expected = TOPIC_SECONDARY_CREDIT * (1.0 / 3.0)
        assert math.isclose(score, expected, rel_tol=1e-4)
        assert 0.0 < score < 1.0

    def test_secondary_credit_multiple_overlap(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """
        dirty_secondary_only: primary=Electronics/Embedded, predicted=[AI/Data, Mathematics]
        User declares [AI/Data, Mathematics] -> primary doesn't match.
        all_post_topics = {Electronics/Embedded, AI/Data, Mathematics} (3 topics)
        overlap = 2 topics -> score = TOPIC_SECONDARY_CREDIT * (2/3)
        """
        post = dirty_catalog["dirty_secondary_only"]
        score = compute_topic_score(
            post, [TopicTaxonomy.AI_DATA, TopicTaxonomy.MATHEMATICS]
        )

        expected = TOPIC_SECONDARY_CREDIT * (2.0 / 3.0)
        assert math.isclose(score, expected, rel_tol=1e-4)

    def test_zero_overlap_secondary(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """User declares [Design] -- no overlap with {Electronics/Embedded, AI/Data, Mathematics}."""
        post = dirty_catalog["dirty_secondary_only"]
        score = compute_topic_score(post, [TopicTaxonomy.DESIGN])
        assert score == 0.0

    def test_primary_match_still_returns_one(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """If user declares Electronics/Embedded (the primary), score should be 1.0 via early return."""
        post = dirty_catalog["dirty_secondary_only"]
        score = compute_topic_score(
            post, [TopicTaxonomy.ELECTRONICS_EMBEDDED]
        )
        assert score == 1.0

    def test_empty_predicted_topics_topic_score(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """
        dirty_no_predicted: primary=AI/Data, predicted_topics=[]
        User declares [Mathematics] -> primary doesn't match.
        all_post_topics = {AI/Data} (just primary, no predicted)
        overlap = 0 -> score = 0.0
        """
        post = dirty_catalog["dirty_no_predicted"]
        score = compute_topic_score(post, [TopicTaxonomy.MATHEMATICS])
        assert score == 0.0


# ─── 3. Creator Score Edge Cases ────────────────────────────────────────────────


class TestCreatorScoreEdgeCases:

    def test_zero_creator_quality(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_zero_creator"]
        assert compute_creator_score(post) == 0.0

    def test_max_creator_quality(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_max_creator"]
        assert compute_creator_score(post) == 1.0

    def test_default_fallback_for_omitted_field(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        post = dirty_catalog["dirty_no_creator_score"]
        assert compute_creator_score(post) == DEFAULT_CREATOR_QUALITY


# ─── 4. Freshness: Extreme Dates ───────────────────────────────────────────────


class TestFreshnessExtremeDates:

    def test_ancient_2019_post_near_zero_freshness(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """A post from March 2019 should have near-zero freshness in Sep 2026."""
        post = dirty_catalog["dirty_ancient"]
        ref_time = datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc)
        score = compute_freshness_score(post.created_at, ref_time)

        # ~2745 days old, half-life is 14 days -> score is astronomically small
        assert score < 0.001
        assert score >= 0.0

    def test_future_post_gets_max_freshness(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """A post dated 2030-01-01 should clamp to freshness 1.0 (negative delta)."""
        post = dirty_catalog["dirty_future"]
        ref_time = datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc)
        score = compute_freshness_score(post.created_at, ref_time)
        assert score == 1.0

    def test_very_recent_post_near_one(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """dirty_001 is from today -- freshness should be very close to 1.0."""
        post = dirty_catalog["dirty_001"]
        ref_time = datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc)
        score = compute_freshness_score(post.created_at, ref_time)
        assert score > 0.95


# ─── 5. Safety Gate in Ranking Engine ───────────────────────────────────────────


class TestSafetyGateWithDirtyData:

    def test_unsafe_posts_excluded_from_ranking(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """Both dirty_unsafe_flagged and dirty_unsafe_partial should be excluded."""
        candidate_ids = [
            "dirty_unsafe_flagged",
            "dirty_unsafe_partial",
            "dirty_safe_cyber",
        ]
        semantic_scores = {pid: 0.90 for pid in candidate_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=candidate_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.CYBERSECURITY],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        result_ids = {r.post_id for r in ranked}
        assert "dirty_unsafe_flagged" not in result_ids
        assert "dirty_unsafe_partial" not in result_ids
        assert "dirty_safe_cyber" in result_ids
        assert len(ranked) == 1

    def test_all_unsafe_returns_empty(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """If every candidate is flagged, result should be empty, not an error."""
        candidate_ids = ["dirty_unsafe_flagged", "dirty_unsafe_partial"]
        semantic_scores = {pid: 0.95 for pid in candidate_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=candidate_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.CYBERSECURITY],
        )

        assert len(ranked) == 0


# ─── 6. Diversity Capping Under Topic Skew ──────────────────────────────────────


class TestDiversityCappingUnderSkew:
    """
    The dirty data has 9 consecutive Programming/Web posts (dirty_001..dirty_008
    plus dirty_no_creator_score). The capping logic should never allow more than 3
    consecutive same-topic posts.
    """

    def test_no_more_than_3_consecutive_same_topic(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
        diversity_service: DiversityService,
    ) -> None:
        """
        Rank all 25 dirty posts with a user who only cares about Programming/Web.
        All Programming/Web posts will score highest. After diversity capping,
        verify no run of >3 consecutive same-topic posts.
        """
        all_ids = list(dirty_catalog.keys())
        # Give Programming/Web posts high semantic scores to make them dominate
        semantic_scores = {}
        for pid, post in dirty_catalog.items():
            if post.primary_topic == TopicTaxonomy.PROGRAMMING_WEB:
                semantic_scores[pid] = 0.90
            else:
                semantic_scores[pid] = 0.30

        ranked = ranking_engine.rank_candidates(
            candidate_ids=all_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        # The unsafe posts should already be excluded
        assert not any(
            r.post_id.startswith("dirty_unsafe") for r in ranked
        )

        capped = diversity_service.apply_capping(ranked)

        # Verify no run of >3 consecutive same topic
        streak = 0
        prev_topic: str | None = None
        for item in capped:
            topic = (
                item.primary_topic.value
                if hasattr(item.primary_topic, "value")
                else str(item.primary_topic)
            )
            if topic == prev_topic:
                streak += 1
            else:
                streak = 1
                prev_topic = topic

            assert streak <= MAX_CONSECUTIVE_SAME_TOPIC, (
                f"Consecutive streak of {streak} for topic '{topic}' "
                f"exceeds max allowed {MAX_CONSECUTIVE_SAME_TOPIC}"
            )

    def test_all_candidates_retained_after_capping(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
        diversity_service: DiversityService,
    ) -> None:
        """Diversity capping must not drop any candidates, only reorder them."""
        all_ids = list(dirty_catalog.keys())
        semantic_scores = {pid: 0.50 for pid in all_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=all_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        capped = diversity_service.apply_capping(ranked)
        assert len(capped) == len(ranked)

    def test_extreme_single_topic_input(self) -> None:
        """If every candidate is the same topic, capping allows all since there is no alternative."""
        breakdowns = [
            ScoreBreakdown(
                post_id=f"mono_{i}",
                final_score=1.0 - (i * 0.01),
                semantic_score=0.90,
                topic_score=1.0,
                creator_score=0.80,
                freshness_score=0.70,
                primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
            )
            for i in range(10)
        ]

        service = DiversityService(max_consecutive_same_topic=3)
        result = service.apply_capping(breakdowns)

        # All 10 should still be present -- no alternative topic to pull forward
        assert len(result) == 10
        assert [r.post_id for r in result] == [f"mono_{i}" for i in range(10)]


# ─── 7. Reason Codes With Dirty Data ───────────────────────────────────────────


class TestReasonCodesWithDirtyData:

    def test_fresh_content_for_today_post(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """dirty_001 was created Sep 20, ref time is Sep 21 -> within 3 days -> FRESH_CONTENT."""
        post = dirty_catalog["dirty_001"]
        breakdown = ScoreBreakdown(
            post_id="dirty_001",
            final_score=0.85,
            semantic_score=0.90,
            topic_score=1.0,
            creator_score=0.95,
            freshness_score=0.99,
            primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
        )

        codes = generate_reason_codes_for_candidate(
            score_breakdown=breakdown,
            declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
            post=post,
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        assert ReasonCode.FRESH_CONTENT in codes
        assert ReasonCode.TOPIC_MATCH in codes
        assert ReasonCode.HIGH_RATED_CREATOR in codes

    def test_no_fresh_content_for_ancient_post(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """dirty_ancient (2019) must never get FRESH_CONTENT."""
        post = dirty_catalog["dirty_ancient"]
        breakdown = ScoreBreakdown(
            post_id="dirty_ancient",
            final_score=0.40,
            semantic_score=0.30,
            topic_score=1.0,
            creator_score=0.65,
            freshness_score=0.0,
            primary_topic=TopicTaxonomy.PROGRAMMING_WEB,
        )

        codes = generate_reason_codes_for_candidate(
            score_breakdown=breakdown,
            declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
            post=post,
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        assert ReasonCode.FRESH_CONTENT not in codes

    def test_high_rated_creator_boundary(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """dirty_max_creator has creator_score=1.0 -> must get HIGH_RATED_CREATOR."""
        post = dirty_catalog["dirty_max_creator"]
        breakdown = ScoreBreakdown(
            post_id="dirty_max_creator",
            final_score=0.75,
            semantic_score=0.50,
            topic_score=0.60,
            creator_score=1.0,
            freshness_score=0.70,
            primary_topic=TopicTaxonomy.ROBOTICS,
        )

        codes = generate_reason_codes_for_candidate(
            score_breakdown=breakdown,
            declared_topics=[TopicTaxonomy.ROBOTICS],
            post=post,
        )

        assert ReasonCode.HIGH_RATED_CREATOR in codes

    def test_zero_creator_no_high_rated_badge(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """dirty_zero_creator has creator_score=0.0 -> must NOT get HIGH_RATED_CREATOR."""
        post = dirty_catalog["dirty_zero_creator"]
        breakdown = ScoreBreakdown(
            post_id="dirty_zero_creator",
            final_score=0.30,
            semantic_score=0.40,
            topic_score=1.0,
            creator_score=0.0,
            freshness_score=0.60,
            primary_topic=TopicTaxonomy.ELECTRONICS_EMBEDDED,
        )

        codes = generate_reason_codes_for_candidate(
            score_breakdown=breakdown,
            declared_topics=[TopicTaxonomy.ELECTRONICS_EMBEDDED],
            post=post,
        )

        assert ReasonCode.HIGH_RATED_CREATOR not in codes

    def test_secondary_topic_no_topic_match_badge(
        self, dirty_catalog: dict[str, PostRecord]
    ) -> None:
        """
        dirty_secondary_only: primary=Electronics/Embedded.
        User declares [AI/Data]. TOPIC_MATCH should NOT fire because the
        primary_topic doesn't match declared_topics.
        """
        post = dirty_catalog["dirty_secondary_only"]
        breakdown = ScoreBreakdown(
            post_id="dirty_secondary_only",
            final_score=0.55,
            semantic_score=0.50,
            topic_score=0.20,
            creator_score=0.75,
            freshness_score=0.70,
            primary_topic=TopicTaxonomy.ELECTRONICS_EMBEDDED,
        )

        codes = generate_reason_codes_for_candidate(
            score_breakdown=breakdown,
            declared_topics=[TopicTaxonomy.AI_DATA],
            post=post,
        )

        assert ReasonCode.TOPIC_MATCH not in codes


# ─── 8. Full Ranking Pipeline With Dirty Data ──────────────────────────────────


class TestFullRankingPipelineDirtyData:

    def test_composite_scores_in_valid_range(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """Every composite score must be in [0.0, 1.0]."""
        all_ids = list(dirty_catalog.keys())
        semantic_scores = {pid: 0.50 for pid in all_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=all_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.AI_DATA, TopicTaxonomy.PROGRAMMING_WEB],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        for item in ranked:
            assert 0.0 <= item.final_score <= 1.0, (
                f"Post {item.post_id} has out-of-range score: {item.final_score}"
            )
            assert 0.0 <= item.semantic_score <= 1.0
            assert 0.0 <= item.topic_score <= 1.0
            assert 0.0 <= item.creator_score <= 1.0
            assert 0.0 <= item.freshness_score <= 1.0

    def test_descending_sort_order(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """Results must be sorted in strict descending final_score order."""
        all_ids = list(dirty_catalog.keys())
        semantic_scores = {pid: 0.50 for pid in all_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=all_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.PROGRAMMING_WEB],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        for i in range(len(ranked) - 1):
            assert ranked[i].final_score >= ranked[i + 1].final_score

    def test_unsafe_posts_never_in_results(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """No matter the semantic score, flagged posts must not appear."""
        all_ids = list(dirty_catalog.keys())
        # Give unsafe posts the highest possible semantic scores
        semantic_scores = {}
        for pid in all_ids:
            if "unsafe" in pid:
                semantic_scores[pid] = 1.0
            else:
                semantic_scores[pid] = 0.10

        ranked = ranking_engine.rank_candidates(
            candidate_ids=all_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.CYBERSECURITY],
        )

        result_ids = {r.post_id for r in ranked}
        assert "dirty_unsafe_flagged" not in result_ids
        assert "dirty_unsafe_partial" not in result_ids

    def test_future_post_ranks_highest_in_freshness(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """dirty_future (2030) gets clamped freshness=1.0 -> should have highest freshness_score."""
        candidate_ids = ["dirty_future", "dirty_ancient", "dirty_with_difficulty"]
        semantic_scores = {pid: 0.50 for pid in candidate_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=candidate_ids,
            semantic_scores=semantic_scores,
            declared_topics=[TopicTaxonomy.AI_DATA, TopicTaxonomy.MATHEMATICS],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        future_item = next(r for r in ranked if r.post_id == "dirty_future")
        ancient_item = next(r for r in ranked if r.post_id == "dirty_ancient")

        assert future_item.freshness_score == 1.0
        assert ancient_item.freshness_score < 0.001

    def test_creator_score_differentiation(
        self,
        dirty_catalog: dict[str, PostRecord],
        ranking_engine: RankingEngine,
    ) -> None:
        """
        Given identical semantic scores, identical topic matches, and similar freshness:
        dirty_max_creator (1.0) should rank above dirty_zero_creator (0.0)
        because creator score differs.
        """
        candidate_ids = ["dirty_max_creator", "dirty_zero_creator"]
        # Give both the same semantic score
        semantic_scores = {pid: 0.50 for pid in candidate_ids}

        ranked = ranking_engine.rank_candidates(
            candidate_ids=candidate_ids,
            semantic_scores=semantic_scores,
            # Both have different primary topics, so use a topic that matches neither
            # to isolate the creator signal
            declared_topics=[TopicTaxonomy.DESIGN],
            reference_time=datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc),
        )

        assert len(ranked) == 2
        assert ranked[0].creator_score > ranked[1].creator_score
        assert ranked[0].post_id == "dirty_max_creator"
        assert ranked[1].post_id == "dirty_zero_creator"


# ─── 9. Composite Score Math Verification With Edge Values ──────────────────────


class TestCompositeScoreMathEdges:

    def test_all_zeros_score(self) -> None:
        score = compute_composite_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_all_ones_score(self) -> None:
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0

    def test_only_semantic_signal(self) -> None:
        """When only semantic signal is present, score = WEIGHT_SEMANTIC * 1.0."""
        score = compute_composite_score(1.0, 0.0, 0.0, 0.0)
        assert math.isclose(score, WEIGHT_SEMANTIC, rel_tol=1e-4)

    def test_only_topic_signal(self) -> None:
        score = compute_composite_score(0.0, 1.0, 0.0, 0.0)
        assert math.isclose(score, WEIGHT_TOPIC, rel_tol=1e-4)

    def test_only_creator_signal(self) -> None:
        score = compute_composite_score(0.0, 0.0, 1.0, 0.0)
        assert math.isclose(score, WEIGHT_CREATOR, rel_tol=1e-4)

    def test_only_freshness_signal(self) -> None:
        score = compute_composite_score(0.0, 0.0, 0.0, 1.0)
        assert math.isclose(score, WEIGHT_FRESHNESS, rel_tol=1e-4)
