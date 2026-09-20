import logging
from typing import Any

from fastapi import APIRouter, Request

from src.config import DEFAULT_CREATOR_QUALITY, MODEL_VERSION
from src.schemas.common import TopicTaxonomy
from src.schemas.events import PostUpsertEvent
from src.schemas.post import PostRecord
from src.schemas.recommendation import (
    RankedPostItem,
    RecommendationRequest,
    RecommendationResponse,
)
from src.services.diversity_service import DiversityService
from src.services.embedding_service import EmbeddingService
from src.services.profile_service import ProfileService
from src.services.ranking_engine import RankingEngine
from src.services.reason_service import ReasonService
from src.services.similarity_engine import SimilarityEngine
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(request: Request) -> dict[str, Any]:
    """
    Health and diagnostics probe exposing service status, vector counts, and model version.
    """
    state = request.app.state
    vector_store: VectorStore | None = getattr(state, "vector_store", None)
    catalog: dict[str, PostRecord] | None = getattr(state, "catalog", None)

    vectors_count = len(vector_store) if vector_store is not None else 0
    catalog_count = len(catalog) if catalog is not None else 0

    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "vectors_count": vectors_count,
        "catalog_count": catalog_count,
    }


@router.post(
    "/api/v1/recommendations/posts",
    response_model=RecommendationResponse,
    status_code=200,
)
def get_recommendations(
    request_data: RecommendationRequest,
    request: Request,
) -> RecommendationResponse:
    """
    Online Educational Post Recommendation Endpoint.
    Executes full pipeline:
    1. Candidate deduplication & exclusion pre-filtering
    2. Learner profile modeling (Cold-start centroid or Warm blend)
    3. Dense semantic similarity retrieval (all-MiniLM-L6-v2)
    4. Multi-factor composite ranking (Semantic, Topic, Creator, Freshness)
    5. Feed diversity topic frequency capping (max 3 consecutive same topic)
    6. Explainable reason code generation
    7. Calibrated response payload construction
    """
    state = request.app.state
    vector_store: VectorStore = state.vector_store
    catalog: dict[str, PostRecord] = state.catalog
    profile_service: ProfileService = state.profile_service
    similarity_engine: SimilarityEngine = state.similarity_engine
    ranking_engine: RankingEngine = state.ranking_engine
    diversity_service: DiversityService = state.diversity_service
    reason_service: ReasonService = state.reason_service

    # Step 1: Pre-filter candidate IDs (remove exclusions and deduplicate)
    filtered_cids = diversity_service.filter_exclusions(
        candidate_ids=request_data.eligible_candidate_ids,
        exclude_post_ids=request_data.exclude_post_ids,
    )

    if not filtered_cids:
        return RecommendationResponse(
            request_id=request_data.request_id,
            model_version=MODEL_VERSION,
            count=0,
            items=[],
        )

    # Step 2: Build learner profile vector (Cold-start or Warm)
    user_vector = profile_service.build_user_profile(
        declared_topics=request_data.declared_topics,
        learning_direction=request_data.learning_direction,
        recent_interactions=request_data.recent_interactions,
        vector_store=vector_store,
    )

    # Step 3: Compute semantic similarities
    scored_pairs = similarity_engine.retrieve_and_score(
        user_vector=user_vector,
        candidate_ids=filtered_cids,
        vector_store=vector_store,
    )
    semantic_scores_dict = dict(scored_pairs)

    # Step 4: Multi-factor composite ranking
    ranked_breakdowns = ranking_engine.rank_candidates(
        candidate_ids=filtered_cids,
        semantic_scores=semantic_scores_dict,
        declared_topics=request_data.declared_topics,
    )

    if not ranked_breakdowns:
        return RecommendationResponse(
            request_id=request_data.request_id,
            model_version=MODEL_VERSION,
            count=0,
            items=[],
        )

    # Step 5: Diversity topic frequency capping
    capped_breakdowns = diversity_service.apply_capping(ranked_breakdowns)

    # Step 6: Slicing up to requested limit
    final_breakdowns = capped_breakdowns[: request_data.limit]

    # Pre-compute interacted topics & post IDs for reason tagging
    interacted_pids: set[str] = set()
    interacted_topics: set[str] = set()
    if request_data.recent_interactions:
        for item in request_data.recent_interactions:
            interacted_pids.add(item.post_id)
            past_post = catalog.get(item.post_id)
            if past_post:
                interacted_topics.add(
                    past_post.primary_topic.value
                    if hasattr(past_post.primary_topic, "value")
                    else str(past_post.primary_topic)
                )

    # Step 7: Assign explainability reason codes
    items: list[RankedPostItem] = []
    for rank_idx, breakdown in enumerate(final_breakdowns, start=1):
        post = catalog.get(breakdown.post_id)
        codes = reason_service.get_reason_codes(
            score_breakdown=breakdown,
            declared_topics=request_data.declared_topics,
            recent_interactions=request_data.recent_interactions,
            post=post,
            interacted_topics=interacted_topics,
            interacted_post_ids=interacted_pids,
        )
        items.append(
            RankedPostItem(
                id=breakdown.post_id,
                rank=rank_idx,
                score=breakdown.final_score,
                reason_codes=codes,
                primary_topic=breakdown.primary_topic,
            )
        )

    return RecommendationResponse(
        request_id=request_data.request_id,
        model_version=MODEL_VERSION,
        count=len(items),
        items=items,
    )


@router.post(
    "/api/v1/events/post-upserted",
    status_code=200,
)
def handle_post_upsert(
    event: PostUpsertEvent,
    request: Request,
) -> dict[str, Any]:
    """
    Asynchronous event endpoint for post creation or update.
    Encodes post text on the fly and updates the in-memory vector store and catalog
    with zero microservice downtime.
    """
    state = request.app.state
    vector_store: VectorStore = state.vector_store
    catalog: dict[str, PostRecord] = state.catalog
    embedding_service: EmbeddingService = state.embedding_service

    # ── Extract topic from explicit field or nested topics object ──────────
    explicit_topic: TopicTaxonomy | None = None
    predicted_topics: list[TopicTaxonomy] = []
    topic_confidence_scores: dict[str, float] = {}

    if event.primary_topic:
        explicit_topic = TopicTaxonomy.normalize(event.primary_topic)

    if event.topics and isinstance(event.topics, dict):
        # Parse primary_topics from Haitham's Content Analysis response
        prim_list = event.topics.get("primary_topics", [])
        if prim_list and isinstance(prim_list, list) and isinstance(prim_list[0], dict):
            raw_top = prim_list[0].get("topic")
            if raw_top and not explicit_topic:
                try:
                    explicit_topic = TopicTaxonomy.normalize(raw_top)
                except (ValueError, KeyError, TypeError) as exc:
                    logger.debug("Failed to normalize primary topic %s: %s", raw_top, exc)

        # Parse secondary_topics from Haitham's response into predicted_topics
        sec_list = event.topics.get("secondary_topics", [])
        if sec_list and isinstance(sec_list, list):
            for sec_item in sec_list:
                if isinstance(sec_item, dict):
                    raw_sec = sec_item.get("topic")
                    if raw_sec:
                        try:
                            predicted_topics.append(TopicTaxonomy.normalize(raw_sec))
                        except (ValueError, KeyError, TypeError) as exc:
                            logger.debug("Failed to normalize secondary topic %s: %s", raw_sec, exc)

        # Consume all_scores from Haitham's topic response for topic_confidence_scores
        all_scores = event.topics.get("all_scores")
        if all_scores and isinstance(all_scores, dict):
            for label, score in all_scores.items():
                try:
                    normalized_topic = TopicTaxonomy.normalize(label)
                    topic_confidence_scores[normalized_topic.value] = float(score)
                except (ValueError, KeyError, TypeError):
                    pass  # Skip labels not in our taxonomy

    # Extract difficulty metadata with fail-safe defaults
    diff_level: str | None = None
    diff_conf: float | None = None
    if event.difficulty and isinstance(event.difficulty, dict):
        diff_level = event.difficulty.get("level")
        diff_conf = event.difficulty.get("confidence")

    # ── Safety Evaluation: Fail-safe / cautious posture ───────────────────
    # 1. Upstream failure or explicit review flag -> HOLD
    if event.needs_review is True or (
        event.processing_status and event.processing_status.lower() in ("partial", "failed")
    ):
        safety_stat = "REVIEW_REQUIRED"
        rec_sig = "DOWNRANK_OR_HOLD"
    # 2. Safety block present -> inspect status and recommendation signal
    elif event.safety and isinstance(event.safety, dict):
        # Accept both "safety_status" (Haitham's key) and "status" (legacy/backend key)
        safety_stat = event.safety.get("safety_status") or event.safety.get("status", "SAFE")
        rec_sig = event.safety.get("recommendation_signal", "ALLOW")
        if (
            event.safety.get("review_required") is True
            or safety_stat == "REVIEW_REQUIRED"
            or rec_sig == "DOWNRANK_OR_HOLD"
        ):
            safety_stat = "REVIEW_REQUIRED"
            rec_sig = "DOWNRANK_OR_HOLD"
    # 3. Explicitly cleared by upstream pipeline
    elif event.needs_review is False:
        safety_stat = "SAFE"
        rec_sig = "ALLOW"
    # 4. Decoupled baseline fallback when upstream analysis service is not attached
    else:
        safety_stat = "SAFE"
        rec_sig = "ALLOW"

    # ── Infer or fallback topic if upstream classifier is offline ─────────
    if explicit_topic:
        inferred_topic = explicit_topic
    else:
        inferred_topic = TopicTaxonomy.PROGRAMMING_WEB
        text_lower = f"{event.title} {event.body}".lower()
        for topic in TopicTaxonomy:
            topic_name = topic.value.lower()
            if any(part in text_lower for part in topic_name.split("/")):
                inferred_topic = topic
                break

    # Fallback confidence scores if Haitham's all_scores was not provided
    if not topic_confidence_scores:
        topic_confidence_scores = {inferred_topic.value: 1.0}

    # ── Creator quality: use Zayan's score if provided, else fallback ─────
    creator_quality = (
        event.creator_teaching_quality
        if event.creator_teaching_quality is not None
        else DEFAULT_CREATOR_QUALITY
    )

    # Build PostRecord with decoupled fallbacks and Content Analysis metadata
    post_record = PostRecord(
        id=event.post_id,
        creator_id=event.creator_id,
        title=event.title,
        body=event.body,
        created_at=event.created_at,
        primary_topic=inferred_topic,
        predicted_topics=predicted_topics,
        topic_confidence_scores=topic_confidence_scores,
        creator_teaching_quality=creator_quality,
        difficulty_level=diff_level,
        difficulty_confidence=diff_conf,
        safety_status=safety_stat,
        recommendation_signal=rec_sig,
    )

    # Encode embedding dynamically
    combined_text = f"{event.title}. {event.body}"
    embedding = embedding_service.embed_text(combined_text)

    # Update in-memory vector store & catalog
    vector_store.upsert(event.post_id, embedding, record=post_record)
    catalog[event.post_id] = post_record

    return {
        "status": "success",
        "post_id": event.post_id,
        "message": "Post indexed successfully",
    }
