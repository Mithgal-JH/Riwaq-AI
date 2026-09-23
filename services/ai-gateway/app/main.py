import os
from contextlib import asynccontextmanager

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
