import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.config import (
    MAX_CONSECUTIVE_SAME_TOPIC,
    MOCK_POSTS_FILE,
    MODEL_NAME,
    MODEL_VERSION,
    POST_EMBEDDINGS_FILE,
    POST_METADATA_FILE,
)
from src.schemas.post import PostRecord
from src.services.diversity_service import DiversityService
from src.services.embedding_service import EmbeddingService
from src.services.profile_service import ProfileService
from src.services.ranking_engine import RankingEngine
from src.services.reason_service import ReasonService
from src.services.similarity_engine import SimilarityEngine
from src.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("binx-rec-service")


def _load_catalog() -> dict[str, PostRecord]:
    catalog: dict[str, PostRecord] = {}
    if MOCK_POSTS_FILE.exists():
        with open(MOCK_POSTS_FILE, encoding="utf-8") as f:
            raw_data = json.load(f)
        for item in raw_data:
            rec = PostRecord.model_validate(item)
            catalog[rec.id] = rec
        logger.info("Loaded %d catalog posts into memory", len(catalog))
    else:
        logger.warning("Mock posts catalog not found at %s", MOCK_POSTS_FILE)
    return catalog


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    Pre-warms embedding model, loads vectors and catalogs into memory on startup
    to minimize initial request latency.
    """
    logger.info("Initializing Post Recommendation Service components...")

    # 1. Embedding Service & Warmup
    embedding_service = EmbeddingService(model_name=MODEL_NAME)
    # Warm up CPU caches
    _ = embedding_service.embed_text("BinX Educational Recommendation Warmup")
    logger.info("Embedding service pre-warmed successfully (%s)", MODEL_NAME)

    # 2. Catalog Ingestion
    catalog = _load_catalog()

    # 3. Vector Store Ingestion
    vector_store = VectorStore()
    if POST_EMBEDDINGS_FILE.exists() and POST_METADATA_FILE.exists():
        success = vector_store.load_from_disk(
            embeddings_path=POST_EMBEDDINGS_FILE,
            metadata_path=POST_METADATA_FILE,
        )
        if success:
            logger.info(
                "Loaded %d vectors from disk cache", len(vector_store)
            )
        else:
            logger.warning("Failed to load vectors from disk cache, building from catalog")
            vector_store.build_from_catalog(
                catalog_path=MOCK_POSTS_FILE,
                embedding_service=embedding_service,
                persist=True,
            )
    else:
        logger.info("Building vector store from catalog and persisting...")
        vector_store.build_from_catalog(
            catalog_path=MOCK_POSTS_FILE,
            embedding_service=embedding_service,
            persist=True,
        )

    # 4. Initialize Core Pipeline Services
    profile_service = ProfileService(embedding_service=embedding_service)
    similarity_engine = SimilarityEngine()
    ranking_engine = RankingEngine(post_catalog=catalog)
    diversity_service = DiversityService(max_consecutive_same_topic=MAX_CONSECUTIVE_SAME_TOPIC)
    reason_service = ReasonService()

    # 5. Store instances in app.state for route dependency access
    app.state.embedding_service = embedding_service
    app.state.catalog = catalog
    app.state.vector_store = vector_store
    app.state.profile_service = profile_service
    app.state.similarity_engine = similarity_engine
    app.state.ranking_engine = ranking_engine
    app.state.diversity_service = diversity_service
    app.state.reason_service = reason_service

    logger.info("Recommendation Microservice ready for traffic.")
    yield
    logger.info("Shutting down Recommendation Microservice.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="BinX Educational Post Recommendation Microservice",
        version=MODEL_VERSION,
        description="Semantic & multi-factor educational post recommendation engine.",
        lifespan=lifespan,
    )

    # CORS Middleware (credentials disallowed for wildcard origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Latency Timing Middleware (X-Process-Time-Ms)
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Global Exception Handler (Opaque error_id, no leaked internals)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        error_id = f"err_{uuid.uuid4().hex[:12]}"
        logger.exception("Unhandled server exception [%s]: %s", error_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error occurred.",
                "error_id": error_id,
            },
        )

    # Mount Routes
    app.include_router(api_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
