import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from src.config import (
    FRESH_CONTENT_MAX_DAYS,
    HIGH_RATED_CREATOR_THRESHOLD,
    REASON_FRESH_THRESHOLD,
    REASON_SEMANTIC_THRESHOLD,
)
from src.schemas.common import ReasonCode, TopicTaxonomy
from src.schemas.post import PostRecord
from src.schemas.recommendation import UserInteractionItem
from src.services.ranking_engine import ScoreBreakdown

logger = logging.getLogger(__name__)


def generate_reason_codes_for_candidate(
    score_breakdown: ScoreBreakdown,
    declared_topics: Sequence[TopicTaxonomy] | None = None,
    recent_interactions: Sequence[UserInteractionItem] | None = None,
    post: PostRecord | None = None,
    reference_time: datetime | None = None,
    interacted_topics: set[str] | None = None,
    interacted_post_ids: set[str] | None = None,
) -> list[ReasonCode]:
    """
    Generate explainable reason codes for a scored candidate post.
    
    Quantitative Criteria:
    - SIMILAR_TO_INTERESTS:
        Triggered if semantic_score >= REASON_SEMANTIC_THRESHOLD (0.75),
        or candidate matches user's recent interaction history (post ID or topic).
    - TOPIC_MATCH:
        Triggered if candidate primary topic matches user's declared topics,
        or topic_score == 1.0.
    - FRESH_CONTENT:
        Triggered if freshness_score >= REASON_FRESH_THRESHOLD (0.85),
        or post was published within the last 72 hours (3 days).
    - HIGH_RATED_CREATOR:
        Triggered if creator_score >= HIGH_RATED_CREATOR_THRESHOLD (0.80).
    """
    codes: list[ReasonCode] = []

    # 1. SIMILAR_TO_INTERESTS
    candidate_topic_str = (
        score_breakdown.primary_topic.value
        if hasattr(score_breakdown.primary_topic, "value")
        else str(score_breakdown.primary_topic)
    )

    is_semantically_high = score_breakdown.semantic_score >= REASON_SEMANTIC_THRESHOLD

    matches_interaction_history = False
    if (
        (interacted_post_ids and score_breakdown.post_id in interacted_post_ids)
        or (interacted_topics and candidate_topic_str in interacted_topics)
    ):
        matches_interaction_history = True
    elif recent_interactions:
        interaction_pids = {item.post_id for item in recent_interactions}
        if score_breakdown.post_id in interaction_pids:
            matches_interaction_history = True

    if is_semantically_high or matches_interaction_history:
        codes.append(ReasonCode.SIMILAR_TO_INTERESTS)

    # 2. TOPIC_MATCH
    is_declared_topic = False
    if declared_topics:
        declared_str_set = {
            t.value if hasattr(t, "value") else str(t) for t in declared_topics
        }
        if candidate_topic_str in declared_str_set:
            is_declared_topic = True
    elif score_breakdown.topic_score >= 1.0:
        is_declared_topic = True

    if is_declared_topic:
        codes.append(ReasonCode.TOPIC_MATCH)

    # 3. FRESH_CONTENT
    is_fresh = score_breakdown.freshness_score >= REASON_FRESH_THRESHOLD
    if not is_fresh and post is not None:
        ref = reference_time or (
            datetime.now(timezone.utc)
            if post.created_at.tzinfo is not None
            else datetime.utcnow()  # noqa: DTZ003
        )
        if post.created_at.tzinfo is not None and ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        elif post.created_at.tzinfo is None and ref.tzinfo is not None:
            ref = ref.astimezone(timezone.utc)

        age = ref - post.created_at
        if age <= timedelta(days=FRESH_CONTENT_MAX_DAYS):
            is_fresh = True

    if is_fresh:
        codes.append(ReasonCode.FRESH_CONTENT)

    # 4. HIGH_RATED_CREATOR
    if score_breakdown.creator_score >= HIGH_RATED_CREATOR_THRESHOLD:
        codes.append(ReasonCode.HIGH_RATED_CREATOR)

    return codes


class ReasonService:
    """
    Explainability service providing reason codes for recommended posts.
    """

    def __init__(
        self,
        semantic_threshold: float = REASON_SEMANTIC_THRESHOLD,
        freshness_threshold: float = REASON_FRESH_THRESHOLD,
        creator_threshold: float = HIGH_RATED_CREATOR_THRESHOLD,
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.freshness_threshold = freshness_threshold
        self.creator_threshold = creator_threshold

    def get_reason_codes(
        self,
        score_breakdown: ScoreBreakdown,
        declared_topics: Sequence[TopicTaxonomy] | None = None,
        recent_interactions: Sequence[UserInteractionItem] | None = None,
        post: PostRecord | None = None,
        reference_time: datetime | None = None,
        interacted_topics: set[str] | None = None,
        interacted_post_ids: set[str] | None = None,
    ) -> list[ReasonCode]:
        return generate_reason_codes_for_candidate(
            score_breakdown=score_breakdown,
            declared_topics=declared_topics,
            recent_interactions=recent_interactions,
            post=post,
            reference_time=reference_time,
            interacted_topics=interacted_topics,
            interacted_post_ids=interacted_post_ids,
        )
