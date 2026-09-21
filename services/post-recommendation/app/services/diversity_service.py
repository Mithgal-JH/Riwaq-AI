import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from app.config import MAX_CONSECUTIVE_SAME_TOPIC

logger = logging.getLogger(__name__)

T = TypeVar("T")


def filter_excluded_candidates(
    candidate_ids: Sequence[str],
    exclude_post_ids: Sequence[str] | None = None,
) -> list[str]:
    """
    Filter out excluded post IDs and deduplicate candidates.
    Preserves original candidate ordering while guaranteeing uniqueness.
    - candidate_ids: Incoming list of candidate IDs.
    - exclude_post_ids: IDs already viewed, saved, or authored by the learner.
    """
    excluded_set = set(exclude_post_ids or [])
    seen: set[str] = set()
    filtered: list[str] = []

    for cid in candidate_ids:
        if cid not in excluded_set and cid not in seen:
            seen.add(cid)
            filtered.append(cid)

    return filtered


def get_candidate_topic(candidate: Any) -> str:
    """
    Extract the primary topic string from a candidate object.
    Supports ScoreBreakdown, PostRecord, dict, or objects with primary_topic.
    """
    if hasattr(candidate, "primary_topic"):
        topic = candidate.primary_topic
        return topic.value if hasattr(topic, "value") else str(topic)
    if isinstance(candidate, dict) and "primary_topic" in candidate:
        topic = candidate["primary_topic"]
        return topic.value if hasattr(topic, "value") else str(topic)
    return str(candidate)


def apply_topic_frequency_capping(
    candidates: Sequence[T],
    max_consecutive: int = MAX_CONSECUTIVE_SAME_TOPIC,
    topic_fn: Callable[[T], str] | None = None,
) -> list[T]:
    """
    Enforce topic diversity by capping consecutive items of the same primary topic.
    
    If the streak of identical primary topics reaches max_consecutive (default 3),
    the algorithm searches ahead for the next candidate with a different topic and
    pulls it forward to break the streak.
    
    Guarantees:
    - Zero candidate loss: all input items are retained in the reordered list.
    - Relative order within the same topic is preserved.
    - If no alternate topic exists in the remainder of the pool, remaining items
      are kept to prevent premature candidate truncation.
    """
    if not candidates:
        return []

    if max_consecutive <= 0:
        return list(candidates)

    extractor = topic_fn or get_candidate_topic
    remaining = list(candidates)
    result: list[T] = []

    current_streak_topic: str | None = None
    streak_count = 0

    while remaining:
        # Check if the next available candidate would violate the streak limit
        head = remaining[0]
        head_topic = extractor(head)

        if head_topic == current_streak_topic and streak_count >= max_consecutive:
            # We must break the streak: look ahead for the first item with a different topic
            alternate_idx = -1
            for idx in range(1, len(remaining)):
                if extractor(remaining[idx]) != current_streak_topic:
                    alternate_idx = idx
                    break

            if alternate_idx != -1:
                # Found an alternate topic item: pull it up
                chosen = remaining.pop(alternate_idx)
                chosen_topic = extractor(chosen)
                result.append(chosen)
                current_streak_topic = chosen_topic
                streak_count = 1
                continue
            else:
                # No alternative topic exists in the remainder of the pool.
                # Retain the head item without discarding valid content.
                chosen = remaining.pop(0)
                result.append(chosen)
                streak_count += 1
                continue

        # Normal placement
        chosen = remaining.pop(0)
        chosen_topic = head_topic
        result.append(chosen)

        if chosen_topic == current_streak_topic:
            streak_count += 1
        else:
            current_streak_topic = chosen_topic
            streak_count = 1

    return result


class DiversityService:
    """
    Service managing feed diversity, deduplication, and candidate exclusion.
    """

    def __init__(self, max_consecutive_same_topic: int = MAX_CONSECUTIVE_SAME_TOPIC) -> None:
        self.max_consecutive_same_topic = max_consecutive_same_topic

    def filter_exclusions(
        self,
        candidate_ids: Sequence[str],
        exclude_post_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Deduplicate candidates and remove excluded post IDs.
        """
        return filter_excluded_candidates(candidate_ids, exclude_post_ids)

    def apply_capping(
        self,
        ranked_candidates: Sequence[T],
        topic_fn: Callable[[T], str] | None = None,
    ) -> list[T]:
        """
        Apply topic frequency capping to a ranked list of candidates.
        """
        return apply_topic_frequency_capping(
            ranked_candidates,
            max_consecutive=self.max_consecutive_same_topic,
            topic_fn=topic_fn,
        )
