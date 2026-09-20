from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.common import TopicTaxonomy


class PostRecord(BaseModel):
    """
    Internal representation of an educational post stored in catalog/vector cache.
    Includes text, derived topic predictions, confidence scores, and creator reputation.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(
        ..., min_length=1, description="Unique post identifier (e.g., pst_101)"
    )
    creator_id: str = Field(
        ..., min_length=1, description="Author identifier (e.g., usr_creator_01)"
    )
    title: str = Field(..., min_length=1, description="Title of the educational post")
    body: str = Field(..., min_length=1, description="Full text body of the post")
    created_at: datetime = Field(..., description="Timestamp of post creation")
    primary_topic: TopicTaxonomy = Field(
        ..., description="Single primary subject topic"
    )
    predicted_topics: list[TopicTaxonomy] = Field(
        default_factory=list,
        description="Predicted topics from 8-topic taxonomy",
    )
    topic_confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores per topic (e.g. {'Programming/Web': 0.92})",
    )
    creator_teaching_quality: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Creator peer teaching reputation score [0.0 to 1.0]",
    )
    ocr_extracted_text: str | None = Field(
        default=None,
        description="Optional text extracted via OCR from diagrams/slides/code screenshots",
    )
    difficulty_level: str | None = Field(
        default=None,
        description="Difficulty level from Content Analysis API (EASY, INTERMEDIATE, ADVANCED)",
    )
    difficulty_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for predicted difficulty",
    )
    safety_status: str | None = Field(
        default="SAFE",
        description="Safety status from Content Analysis API (SAFE, REVIEW_REQUIRED)",
    )
    recommendation_signal: str | None = Field(
        default="ALLOW",
        description="Moderation recommendation signal (ALLOW, DOWNRANK_OR_HOLD)",
    )
