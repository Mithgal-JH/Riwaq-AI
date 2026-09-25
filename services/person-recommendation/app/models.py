"""
Pydantic request/response models — field names and types match
BinX_PersonToPerson_Recommendation_API_Contract.docx exactly.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- Data Sync

class SyncProfile(BaseModel):
    profile_id: str
    user_id: str = ""
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    learning_direction: Optional[str] = None
    bio: str = ""


class SyncRequest(BaseModel):
    sync_type: Literal["full", "upsert", "delete"]
    profiles: Optional[List[SyncProfile]] = None   # required for full / upsert
    profile_ids: Optional[List[str]] = None        # required for delete


class SyncResponse(BaseModel):
    sync_status: str
    profiles_indexed: int
    index_version: str


# ---------------------------------------------------------------- Recommendations

class RecommendationItem(BaseModel):
    candidate_profile_id: str
    similarity_score: float
    shared_skills: List[str]
    shared_interests: List[str]
    same_learning_direction: bool


class RecommendationResponse(BaseModel):
    request_id: str
    profile_id: str
    processing_status: str
    processed_at: str
    recommendations: List[RecommendationItem]
    recommendation_count: int
    low_confidence: bool
    model_version: str
    preprocessing_version: str
