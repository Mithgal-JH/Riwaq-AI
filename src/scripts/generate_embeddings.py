"""
Precomputation script:
Loads all 40 educational posts from data/mock_posts.json, generates
dense 384-dimensional embeddings via SentenceTransformers ('all-MiniLM-L6-v2'),
and persists both embeddings and metadata to disk for instant retrieval.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    MOCK_POSTS_FILE,
    POST_EMBEDDINGS_FILE,
    POST_METADATA_FILE,
)
from src.services.embedding_service import EmbeddingService
from src.services.vector_store import VectorStore


def main() -> None:
    print(f"Loading catalog from: {MOCK_POSTS_FILE}")
    start_time = time.perf_counter()

    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    count = vector_store.build_from_catalog(
        catalog_path=MOCK_POSTS_FILE,
        embedding_service=embedding_service,
        persist=True,
    )

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"Successfully embedded and indexed {count} posts in {elapsed:.1f} ms.")
    print(f"Saved vectors to: {POST_EMBEDDINGS_FILE}")
    print(f"Saved metadata to: {POST_METADATA_FILE}")


if __name__ == "__main__":
    main()
