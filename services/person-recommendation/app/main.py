"""
BinX AI — Person-to-Person (Profile) Recommendation service.

Implements exactly the two endpoints defined in
BinX_PersonToPerson_Recommendation_API_Contract.docx:

  POST /api/v1/ai/recommendations/people/sync
  GET  /api/v1/ai/recommendations/people/{profile_id}

Owner: Zayan Shawareb — Team3-AI, BinX Tech.
"""
from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .models import RecommendationResponse, SyncRequest, SyncResponse
from .recommender import MODEL_VERSION, PREPROCESSING_VERSION, RecommendationIndex

app = FastAPI(
    title="BinX AI — Person-to-Person Recommendation",
    version="1.0.0",
    description="Content-based (TF-IDF + Cosine Similarity) profile recommendation service.",
)

index = RecommendationIndex()
logger = logging.getLogger("person-recommendation")


@app.exception_handler(RequestValidationError)
async def log_sync_validation_error(request, exc: RequestValidationError):
    if request.url.path == "/api/v1/ai/recommendations/people/sync":
        logger.error(
            "TEMPORARY sync validation failure: path=%s errors=%s",
            request.url.path,
            exc.errors(),
        )

    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/health")
def health():
    return {"status": "ok", "profiles_indexed": len(index.profiles)}


@app.post("/api/v1/ai/recommendations/people/sync", response_model=SyncResponse)
def sync_profiles(payload: SyncRequest):
    if payload.sync_type in ("full", "upsert"):
        if not payload.profiles:
            raise HTTPException(status_code=400, detail="'profiles' is required for sync_type=full/upsert.")
        profiles_as_dicts = [p.model_dump() for p in payload.profiles]
        count = index.full(profiles_as_dicts) if payload.sync_type == "full" else index.upsert(profiles_as_dicts)

    elif payload.sync_type == "delete":
        if not payload.profile_ids:
            raise HTTPException(status_code=400, detail="'profile_ids' is required for sync_type=delete.")
        count = index.delete(payload.profile_ids)

    else:  # unreachable — Literal already restricts this, kept for safety
        raise HTTPException(status_code=400, detail="sync_type must be one of: full, upsert, delete.")

    return SyncResponse(sync_status="completed", profiles_indexed=count, index_version=_now_iso())


@app.get("/api/v1/ai/recommendations/people/{profile_id}", response_model=RecommendationResponse)
def get_recommendations(
    profile_id: str = Path(..., description="The profile requesting recommendations."),
    top_n: int = Query(5, description="Number of recommendations to return. Default 5, max 20."),
    request_id: str = Query(None, description="Unique ID for tracing and retries."),
):
    # Validated manually (not via Query(ge=.., le=..)) so violations return
    # 400 per the contract's HTTP Errors table, not FastAPI's default 422.
    if not profile_id:
        raise HTTPException(status_code=400, detail="profile_id is required.")
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required.")
    if top_n < 1 or top_n > 20:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 20.")

    result = index.recommend(profile_id, top_n)
    if result is None:
        # Never a live fetch back to the Backend — see Integration Rules.
        raise HTTPException(
            status_code=404,
            detail=f"profile_id '{profile_id}' not found in the AI service's synced index.",
        )

    recommendations, low_confidence = result
    return RecommendationResponse(
        request_id=request_id,
        profile_id=profile_id,
        processing_status="completed",
        processed_at=_now_iso(),
        recommendations=recommendations,
        recommendation_count=len(recommendations),
        low_confidence=low_confidence,
        model_version=MODEL_VERSION,
        preprocessing_version=PREPROCESSING_VERSION,
    )
