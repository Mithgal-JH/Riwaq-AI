from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import (
    DEFAULT_RECOMMENDATION_LIMIT,
    MAX_CANDIDATES_PER_REQUEST,
    MAX_RECOMMENDATION_LIMIT,
)
from app.schemas.common import InteractionType, ReasonCode, TopicTaxonomy


class UserInteractionItem(BaseModel):
    """A single user interaction event used for warm profile vector updating."""

    model_config = ConfigDict(extra="ignore")

    post_id: str = Field(..., min_length=1, description="Interacted post identifier")
    interaction_type: InteractionType = Field(
        ..., description="Action type: like, save, or repost"
    )
    timestamp: datetime = Field(..., description="Interaction occurrence timestamp")


class RecommendationRequest(BaseModel):
    """
    Inbound recommendation request from the .NET Backend.
    Endpoint: POST /api/v1/recommendations/posts
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(..., min_length=1, description="Unique tracking identifier")
    user_id: str = Field(
        ..., min_length=1, description="The learner user requesting the feed"
    )
    declared_topics: list[TopicTaxonomy] = Field(
        ...,
        min_length=1,
        description="Onboarding topics chosen by user (from the 8-class taxonomy)",
    )
    learning_direction: str = Field(
        ...,
        min_length=1,
        description="Primary learning direction (e.g., 'Backend Development')",
    )
    eligible_candidate_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Candidate post IDs already pre-filtered by backend as published and non-blocked",
    )
    exclude_post_ids: list[str] = Field(
        default_factory=list,
        description="Post IDs to exclude (already displayed, saved, or authored by user)",
    )
    limit: int = Field(
        default=DEFAULT_RECOMMENDATION_LIMIT,
        ge=1,
        le=MAX_RECOMMENDATION_LIMIT,
        description="Number of ranked posts to return",
    )
    recent_interactions: list[UserInteractionItem] | None = Field(
        default=None,
        description="Recent positive interactions for dynamic warm-learner profile modeling",
    )

    @field_validator("eligible_candidate_ids")
    @classmethod
    def validate_candidate_batch_size(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_CANDIDATES_PER_REQUEST:
            raise ValueError(
                f"Candidate list exceeds maximum batch size of {MAX_CANDIDATES_PER_REQUEST}. Received {len(v)} candidates."
            )
        return v


class RankedPostItem(BaseModel):
    """A single ranked post item in the recommendation response."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Post identifier")
    rank: int = Field(
        ..., ge=1, description="1-based rank position in the recommended feed"
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Final composite ranking score in [0.0, 1.0]"
    )
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Explainable reason codes for UI badges on feed cards",
    )
    primary_topic: TopicTaxonomy = Field(..., description="Primary topic of the post")


class RecommendationResponse(BaseModel):
    """
    Outbound recommendation payload strictly adhering to Section 5.1 JSON contract.
    Returned with HTTP 200 OK.
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(..., description="Correlated request identifier")
    model_version: str = Field(
        ..., description="Model version tag generating the recommendations"
    )
    count: int = Field(..., ge=0, description="Number of items returned")
    items: list[RankedPostItem] = Field(
        default_factory=list, description="Ordered list of ranked posts"
    )
