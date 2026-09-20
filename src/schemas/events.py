from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.common import TopicTaxonomy


class PostUpsertEvent(BaseModel):
    """
    Asynchronous event payload received from the .NET Backend
    when an educational post is created, published, or updated.
    Supports optional content analysis metadata (topics, difficulty, safety).
    """

    model_config = ConfigDict(extra="ignore")

    post_id: str = Field(
        ..., min_length=1, description="Unique identifier for the post"
    )
    creator_id: str = Field(..., min_length=1, description="Author identifier")
    title: str = Field(
        ..., min_length=1, max_length=300, description="Title of the educational post"
    )
    body: str = Field(
        ..., min_length=1, description="Text body content of the educational post"
    )
    created_at: datetime = Field(
        ..., description="Creation/publication timestamp in ISO 8601 format"
    )
    primary_topic: TopicTaxonomy | None = Field(
        default=None, description="Primary taxonomy topic"
    )
    difficulty: dict[str, Any] | None = Field(
        default=None, description="Difficulty object from Content Analysis API"
    )
    safety: dict[str, Any] | None = Field(
        default=None, description="Safety object from Content Analysis API"
    )
    topics: dict[str, Any] | None = Field(
        default=None, description="Topics container object from Content Analysis API"
    )
