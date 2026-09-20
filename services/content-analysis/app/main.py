from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.model_service import (
    get_model_service,
    models_are_loaded,
)


app = FastAPI(
    title="BinX AI Content Analysis API",
    description=(
        "Topic classification, difficulty classification "
        "and content safety analysis."
    ),
    version="1.0.0",
)

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="English post textual content.",
    )
    post_id: Optional[str] = None


@app.get("/")
def root():
    return {
        "service": "BinX AI Content Analysis API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": models_are_loaded(),
        "models": {
            "topic": (
                MODELS_DIR / "topic_model"
            ).exists(),
            "safety": (
                MODELS_DIR / "safety_model"
            ).exists(),
            "difficulty": (
                MODELS_DIR / "difficulty_model"
            ).exists(),
        },
        "endpoints": {
            "documentation": "/docs",
            "analysis": "/analyze",
            "preload_models": "/models/load",
        },
    }


@app.post("/models/load")
def load_models():
    try:
        service = get_model_service()

        return {
            "status": "loaded",
            "device": service.device,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {error}",
        )


@app.post("/analyze")
def analyze_post(request: AnalyzeRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty.",
        )

    try:
        service = get_model_service()
        result = service.analyze(text)

        return {
            "post_id": request.post_id,
            "text_length": len(text),
            **result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {error}",
        )