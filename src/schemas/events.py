from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PostUpsertEvent(BaseModel):
    """
    Asynchronous event payload received from the .NET Backend
    when an educational post is created, published, or updated.
    """

    model_config = ConfigDict(extra="forbid")

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
