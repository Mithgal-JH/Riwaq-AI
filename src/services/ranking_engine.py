import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import (
    DEFAULT_CREATOR_QUALITY,
    FRESHNESS_HALF_LIFE_DAYS,
    TOPIC_SECONDARY_CREDIT,
    WEIGHT_CREATOR,
    WEIGHT_FRESHNESS,
    WEIGHT_SEMANTIC,
    WEIGHT_TOPIC,
)
from src.schemas.common import TopicTaxonomy
from src.schemas.post import PostRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreBreakdown:
    """
    Detailed candidate scoring breakdown.
    Preserves individual sub-scores for auditing, debugging, and reason code tagging.
    """

    post_id: str
    final_score: float
    semantic_score: float
    topic_score: float
    creator_score: float
    freshness_score: float
    primary_topic: TopicTaxonomy


# Alias for backward and forward compatibility
ScoredCandidate = ScoreBreakdown


def clamp_unit_interval(val: float) -> float:
    """
    Clamp floating value strictly into [0.0, 1.0].
    """
    if math.isnan(val):
        return 0.0
    return max(0.0, min(1.0, float(val)))


def compute_semantic_score(raw_similarity: float) -> float:
    """
    Ensure semantic similarity is clamped strictly in [0.0, 1.0].
    """
    return clamp_unit_interval(raw_similarity)


def compute_topic_score(
    post: PostRecord,
    declared_topics: Sequence[TopicTaxonomy],
) -> float:
    """
    Calculate topic alignment score using containment over post topics:
    - If primary topic matches declared_topics: returns 1.00.
    - If only secondary / predicted topics match: returns TOPIC_SECONDARY_CREDIT (0.60) * overlap.
    - Returns 0.00 if there is zero topic overlap.
    """
    if not declared_topics:
        return 0.0

    declared_values = {
        t.value if hasattr(t, "value") else str(t) for t in declared_topics
    }
    post_primary = (
        post.primary_topic.value
        if hasattr(post.primary_topic, "value")
        else str(post.primary_topic)
    )

    # 1. Primary-topic hit -> 1.0
    if post_primary in declared_values:
        return 1.0

    # 2. Containment over the post's topic set
    post_predicted_values = {
        t.value if hasattr(t, "value") else str(t) for t in post.predicted_topics
    }
    all_post_topics = {post_primary} | post_predicted_values
    if not all_post_topics:
        return 0.0

    overlap = len(all_post_topics & declared_values) / len(all_post_topics)
    return clamp_unit_interval(TOPIC_SECONDARY_CREDIT * overlap)


def compute_creator_score(post: PostRecord) -> float:
    """
    Extract author's peer teaching quality score.
    Falls back to DEFAULT_CREATOR_QUALITY (0.50) if missing or unrated.
    Clamped strictly in [0.0, 1.0].
    """
    quality = getattr(post, "creator_teaching_quality", None)
    if quality is None or math.isnan(quality):
        return DEFAULT_CREATOR_QUALITY
    return clamp_unit_interval(quality)


def compute_freshness_score(
    created_at: datetime,
    reference_time: datetime | None = None,
) -> float:
    """
    Calculate true exponential freshness half-life decay:
        Freshness(Δt) = exp(-ln(2) * Δt / 14 days)
    Guarantees a post aged exactly 14 days receives score 0.50.
    Clamped strictly in [0.0, 1.0].
    """
    if reference_time is None:
        if created_at.tzinfo is not None:
            reference_time = datetime.now(timezone.utc)
        else:
            reference_time = datetime.utcnow()  # noqa: DTZ003
    else:
        # Synchronize timezone awareness if needed
        if created_at.tzinfo is not None and reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        elif created_at.tzinfo is None and reference_time.tzinfo is not None:
            created_at = created_at.replace(tzinfo=timezone.utc)

    delta_seconds = (reference_time - created_at).total_seconds()
    if delta_seconds <= 0:
        return 1.0

    delta_days = delta_seconds / 86400.0
    # True half-life: S_fresh(14d) == 0.50
    freshness = math.exp(-math.log(2.0) * delta_days / FRESHNESS_HALF_LIFE_DAYS)
    return clamp_unit_interval(freshness)


def compute_composite_score(
    semantic_score: float,
    topic_score: float,
    creator_score: float,
    freshness_score: float,
) -> float:
    """
    Compute weighted composite PostScore:
        0.70 * Semantic + 0.15 * Topic + 0.10 * Creator + 0.05 * Freshness
    Guaranteed clamped in [0.0, 1.0].
    """
    composite = (
        WEIGHT_SEMANTIC * semantic_score
        + WEIGHT_TOPIC * topic_score
        + WEIGHT_CREATOR * creator_score
        + WEIGHT_FRESHNESS * freshness_score
    )
    clamped = clamp_unit_interval(composite)
    return round(clamped, 4)


class RankingEngine:
    """
    Multi-factor candidate ranking engine.
    Combines semantic, topic, creator, and freshness signals into a single score.
    """

    _warned_synthetic_creator = False

    def __init__(
        self,
        post_catalog: dict[str, PostRecord] | Callable[[str], PostRecord | None],
    ) -> None:
        if isinstance(post_catalog, dict):
            self._get_post: Callable[[str], PostRecord | None] = post_catalog.get
        else:
            self._get_post = post_catalog

        if not RankingEngine._warned_synthetic_creator:
            logger.warning(
                "WARNING: creator quality scores are hand-authored synthetic values from "
                "mock_posts.json, not model output. They affect ranking but carry no real signal."
            )
            RankingEngine._warned_synthetic_creator = True

    def rank_candidates(
        self,
        candidate_ids: Sequence[str],
        semantic_scores: dict[str, float],
        declared_topics: Sequence[TopicTaxonomy],
        reference_time: datetime | None = None,
    ) -> list[ScoreBreakdown]:
        """
        Score and rank candidate post IDs in descending order of composite PostScore.
        Skips candidates missing from the post catalog.
        """
        scored_candidates: list[ScoreBreakdown] = []

        for pid in candidate_ids:
            post = self._get_post(pid)
            if post is None:
                continue

            raw_sem = semantic_scores.get(pid, 0.0)
            sem_score = compute_semantic_score(raw_sem)
            top_score = compute_topic_score(post, declared_topics)
            cre_score = compute_creator_score(post)
            fre_score = compute_freshness_score(
                post.created_at, reference_time=reference_time
            )

            final_score = compute_composite_score(
                semantic_score=sem_score,
                topic_score=top_score,
                creator_score=cre_score,
                freshness_score=fre_score,
            )

            scored_candidates.append(
                ScoreBreakdown(
                    post_id=pid,
                    final_score=final_score,
                    semantic_score=round(sem_score, 4),
                    topic_score=round(top_score, 4),
                    creator_score=round(cre_score, 4),
                    freshness_score=round(fre_score, 4),
                    primary_topic=post.primary_topic,
                )
            )

        # Sort descending by final_score, breaking ties on semantic_score
        scored_candidates.sort(
            key=lambda c: (c.final_score, c.semantic_score),
            reverse=True,
        )

        return scored_candidates
