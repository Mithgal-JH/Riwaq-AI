import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

CONTENT_ANALYSIS_URL = os.getenv("CONTENT_ANALYSIS_URL", "http://127.0.0.1:8001")
POST_RECOMMENDATION_URL = os.getenv("POST_RECOMMENDATION_URL", "http://127.0.0.1:8002")
PERSON_RECOMMENDATION_URL = os.getenv("PERSON_RECOMMENDATION_URL", "http://127.0.0.1:8003")
BLOCKED_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "content-encoding"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=180.0)
    yield
    await app.state.client.aclose()

app = FastAPI(
    title="Riwaq Unified AI API",
    version="1.0.0",
    description="Unified gateway for all Riwaq AI services.",
    lifespan=lifespan,
)

async def forward_request(request: Request, service_url: str, target_path: str) -> Response:
    url = f"{service_url.rstrip('/')}/{target_path.lstrip('/')}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in BLOCKED_HEADERS}
    try:
        upstream = await request.app.state.client.request(
            method=request.method,
            url=url,
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AI service unavailable: {exc}") from exc
    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in BLOCKED_HEADERS}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=response_headers)

@app.get("/")
def root():
    return {
        "service": "Riwaq Unified AI API",
        "status": "running",
        "docs": "/docs",
        "services": {
            "content_analysis": "/api/v1/ai/content",
            "post_recommendation": "/api/v1/recommendations",
            "person_recommendation": "/api/v1/ai/recommendations/people",
        },
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "riwaq-ai-gateway"}

@app.post("/api/v1/ai/content/analyze")
async def content_adapter(request: Request):
    b = await request.json()
    rid = b.get("requestId") or b.get("RequestId") or b.get("request_id")
    cid = b.get("contentId") or b.get("ContentId") or b.get("content_id")
    ver = b.get("contentVersion") or b.get("ContentVersion") or b.get("content_version") or 1
    text = b.get("text") or b.get("Text")
    if not rid or not cid or not text:
        raise HTTPException(422, "requestId, contentId and text are required")
    try:
        u = await request.app.state.client.post(
            f"{CONTENT_ANALYSIS_URL.rstrip('/')}/analyze",
            json={"post_id": str(cid), "text": str(text)},
        )
    except httpx.RequestError as exc:
        raise HTTPException(502, f"AI service unavailable: {exc}") from exc
    if u.status_code >= 400:
        return Response(content=u.content, status_code=u.status_code, media_type="application/json")
    d = u.json()
    t, df, s, mv = (d.get("topics") or {}, d.get("difficulty") or {},
                     d.get("safety") or {}, d.get("model_versions") or {})
    risks = s.get("risk_categories") or []
    names = [x.get("category", "") if isinstance(x, dict) else str(x) for x in risks]
    objects = [x for x in risks if isinstance(x, dict)]
    if objects:
        top = max(objects, key=lambda x: float(x.get("confidence", 0) or 0))
        confidence, threshold = float(top.get("confidence", 0) or 0), top.get("threshold")
    else:
        scores = s.get("all_scores") or {}
        confidence, threshold = max((float(x) for x in scores.values()), default=0.0), None
    return {
        "request_id": str(rid), "content_id": str(cid), "content_version": int(ver),
        "processing_status": "completed", "processed_at": datetime.now(timezone.utc).isoformat(),
        "topics": {"classification_status": t.get("classification_status", "unclassified"),
                   "reason_code": t.get("reason_code"), "primary_topics": t.get("primary_topics", []),
                   "secondary_topics": t.get("secondary_topics", []), "topic_count": int(t.get("topic_count", 0) or 0)},
        "difficulty": {"level": df.get("level", ""), "confidence": float(df.get("confidence", 0) or 0)},
        "safety": {"status": s.get("safety_status", ""), "confidence": confidence,
                   "threshold": threshold, "risk_categories": [x for x in names if x],
                   "review_required": bool(s.get("review_required", False)),
                   "recommendation_signal": s.get("recommendation_signal", "")},
        "needs_review": bool(d.get("needs_review", False)),
        "model_versions": {"topic_model": mv.get("topic", ""), "difficulty_model": mv.get("difficulty", ""),
                          "safety_model": mv.get("safety", "")},
        "preprocessing_version": "binx-content-preprocessing-v1",
    }


@app.api_route("/api/v1/ai/content/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def content_proxy(path: str, request: Request):
    return await forward_request(request, CONTENT_ANALYSIS_URL, path)

@app.api_route("/api/v1/recommendations/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def post_proxy(path: str, request: Request):
    return await forward_request(request, POST_RECOMMENDATION_URL, f"api/v1/recommendations/{path}")

@app.api_route("/api/v1/events/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def event_proxy(path: str, request: Request):
    return await forward_request(request, POST_RECOMMENDATION_URL, f"api/v1/events/{path}")

@app.api_route("/api/v1/ai/recommendations/people/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def person_proxy(path: str, request: Request):
    return await forward_request(request, PERSON_RECOMMENDATION_URL, f"api/v1/ai/recommendations/people/{path}")
