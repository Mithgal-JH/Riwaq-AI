import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from app.config import (
    EMBEDDING_DIM,
    MOCK_POSTS_FILE,
    POST_EMBEDDINGS_FILE,
    POST_METADATA_FILE,
)
from app.schemas.post import PostRecord
from app.services.embedding_service import EmbeddingService


class VectorStore:
    """
    In-memory vector store for educational post embeddings with persistence capabilities.
    Supports sub-millisecond candidate lookup and matrix operations.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        self._records: dict[str, PostRecord] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def contains(self, post_id: str) -> bool:
        return post_id in self._vectors

    def upsert(
        self,
        post_id: str,
        vector: np.ndarray,
        record: PostRecord | None = None,
    ) -> None:
        """
        Add or update a post vector and optional metadata record.
        Vector is converted to float32 and normalized if needed.
        """
        vec = np.asarray(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._vectors[post_id] = vec
        if record is not None:
            self._records[post_id] = record

    def get_vector(self, post_id: str) -> np.ndarray | None:
        """Retrieve single post vector by ID."""
        return self._vectors.get(post_id)

    def get_record(self, post_id: str) -> PostRecord | None:
        """Retrieve single post metadata record by ID."""
        return self._records.get(post_id)

    def get_vectors_for_candidates(
        self, candidate_ids: list[str]
    ) -> tuple[list[str], np.ndarray]:
        """
        Retrieve existing candidate vectors preserving order of available candidates.
        Returns:
            found_ids: list of candidate IDs that exist in the vector store
            vectors_matrix: 2D numpy array of shape (len(found_ids), 384)
        """
        found_ids: list[str] = []
        vectors_list: list[np.ndarray] = []

        for cid in candidate_ids:
            vec = self._vectors.get(cid)
            if vec is not None:
                found_ids.append(cid)
                vectors_list.append(vec)

        if not vectors_list:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        return found_ids, np.stack(vectors_list, axis=0)

    def save_to_disk(
        self,
        embeddings_path: Path = POST_EMBEDDINGS_FILE,
        metadata_path: Path = POST_METADATA_FILE,
    ) -> None:
        """
        Persist in-memory vectors to a numpy file and metadata records to a JSON file.
        """
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        ordered_ids = list(self._vectors.keys())
        if ordered_ids:
            matrix = np.stack([self._vectors[pid] for pid in ordered_ids], axis=0)
        else:
            matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        np.save(embeddings_path, matrix)

        metadata: dict[str, Any] = {
            "post_ids": ordered_ids,
            "records": {
                pid: self._records[pid].model_dump(mode="json")
                for pid in ordered_ids
                if pid in self._records
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_from_disk(
        self,
        embeddings_path: Path = POST_EMBEDDINGS_FILE,
        metadata_path: Path = POST_METADATA_FILE,
    ) -> bool:
        """
        Load persisted vectors and records from disk. Returns True if successful.
        """
        if not embeddings_path.exists() or not metadata_path.exists():
            return False

        matrix = np.load(embeddings_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        post_ids = metadata.get("post_ids", [])
        records_data = metadata.get("records", {})

        self._vectors.clear()
        self._records.clear()

        for idx, pid in enumerate(post_ids):
            if idx < len(matrix):
                self._vectors[pid] = np.asarray(matrix[idx], dtype=np.float32)
            if pid in records_data:
                try:
                    self._records[pid] = PostRecord.model_validate(records_data[pid])
                except ValidationError:
                    continue

        return True

    def build_from_catalog(
        self,
        catalog_path: Path = MOCK_POSTS_FILE,
        embedding_service: EmbeddingService | None = None,
        persist: bool = True,
    ) -> int:
        """
        Load posts from JSON catalog, batch embed them, and index in memory.
        """
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found at: {catalog_path}")

        with open(catalog_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        records = [PostRecord.model_validate(item) for item in raw_data]
        if embedding_service is None:
            embedding_service = EmbeddingService()

        texts = [embedding_service.prepare_post_text(rec) for rec in records]
        vectors = embedding_service.embed_batch(texts)

        for rec, vec in zip(records, vectors):
            self.upsert(rec.id, vec, record=rec)

        if persist:
            self.save_to_disk()

        return len(records)
