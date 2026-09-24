import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np

from app.config import (
    DATA_DIR,
    MOCK_POSTS_FILE,
    POST_EMBEDDINGS_FILE,
    POST_METADATA_FILE,
)
from app.schemas.common import TopicTaxonomy
from app.schemas.post import PostRecord
from app.services.embedding_service import EmbeddingService
from app.services.profile_service import ProfileService
from app.services.ranking_engine import RankingEngine, ScoreBreakdown
from app.services.similarity_engine import SimilarityEngine
from app.services.vector_store import VectorStore


class EvalProfile(TypedDict):
    name: str
    declared_topics: list[TopicTaxonomy]
    direction: str


SAMPLE_EVAL_PROFILES: list[EvalProfile] = [
    {
        "name": "Backend Python Learner",
        "declared_topics": [TopicTaxonomy.PROGRAMMING_WEB],
        "direction": "Full Stack Web Development, FastAPI, and Python Asyncio Concurrency",
    },
    {
        "name": "Machine Learning Engineer",
        "declared_topics": [TopicTaxonomy.AI_DATA],
        "direction": "Machine Learning Models, Deep Learning, and PyTorch Neural Networks",
    },
    {
        "name": "Cybersecurity Specialist",
        "declared_topics": [TopicTaxonomy.CYBERSECURITY],
        "direction": "Network Security, Ethical Hacking, and Cryptography Protocols",
    },
    {
        "name": "Robotics & Hardware Explorer",
        "declared_topics": [TopicTaxonomy.ROBOTICS, TopicTaxonomy.ELECTRONICS_EMBEDDED],
        "direction": "ROS2, Arduino, Microcontrollers, and Autonomous Robotics",
    },
    {
        "name": "Digital Product Designer",
        "declared_topics": [TopicTaxonomy.DESIGN],
        "direction": "UI/UX Design Systems, Wireframing, Figma, and User Psychology",
    },
    {
        "name": "Applied Mathematician",
        "declared_topics": [TopicTaxonomy.MATHEMATICS],
        "direction": "Linear Algebra, Multivariable Calculus, and Convex Optimization",
    },
    {
        "name": "Natural Sciences Researcher",
        "declared_topics": [TopicTaxonomy.NATURAL_SCIENCES],
        "direction": "Quantum Physics, Biophysics, and Molecular Biology Systems",
    },
    {
        "name": "Full-Stack AI Developer (Broad Interest)",
        "declared_topics": [TopicTaxonomy.PROGRAMMING_WEB, TopicTaxonomy.AI_DATA],
        "direction": "AI-Powered Web Applications, LLM Integrations, and Distributed Backends",
    },
]


def calculate_distribution_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "count": float(len(arr)),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def run_distribution_study(
    output_path: Path = DATA_DIR / "score_distribution.json",
    reference_time: datetime | None = None,
) -> dict[str, dict[str, float]]:
    if reference_time is None:
        reference_time = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Load mock posts catalog
    with open(MOCK_POSTS_FILE, encoding="utf-8") as f:
        raw_catalog = json.load(f)
    catalog = {item["id"]: PostRecord.model_validate(item) for item in raw_catalog}

    # 2. Load vector store
    vector_store = VectorStore()
    vector_store.load_from_disk(POST_EMBEDDINGS_FILE, POST_METADATA_FILE)

    embedding_service = EmbeddingService()
    profile_service = ProfileService(embedding_service)
    similarity_engine = SimilarityEngine()
    ranking_engine = RankingEngine(catalog)

    all_candidate_ids = list(catalog.keys())

    all_semantic: list[float] = []
    all_topic: list[float] = []
    all_creator: list[float] = []
    all_freshness: list[float] = []
    all_composite: list[float] = []

    print(
        f"Running Score Distribution Study over {len(catalog)} posts x {len(SAMPLE_EVAL_PROFILES)} profiles..."
    )

    for prof in SAMPLE_EVAL_PROFILES:
        user_vec = profile_service.build_declared_profile(
            declared_topics=prof["declared_topics"],
            learning_direction=prof["direction"],
        )

        ranked_similarities = similarity_engine.retrieve_and_score(
            user_vector=user_vec,
            candidate_ids=all_candidate_ids,
            vector_store=vector_store,
        )
        sim_dict = dict(ranked_similarities)

        scored: list[ScoreBreakdown] = ranking_engine.rank_candidates(
            candidate_ids=all_candidate_ids,
            semantic_scores=sim_dict,
            declared_topics=prof["declared_topics"],
            reference_time=reference_time,
        )

        for s in scored:
            all_semantic.append(s.semantic_score)
            all_topic.append(s.topic_score)
            all_creator.append(s.creator_score)
            all_freshness.append(s.freshness_score)
            all_composite.append(s.final_score)

    signals: dict[str, dict[str, float]] = {
        "semantic": calculate_distribution_stats(all_semantic),
        "topic": calculate_distribution_stats(all_topic),
        "creator": calculate_distribution_stats(all_creator),
        "freshness": calculate_distribution_stats(all_freshness),
        "composite": calculate_distribution_stats(all_composite),
    }

    distribution_data = {
        "timestamp": reference_time.isoformat(),
        "profiles_count": len(SAMPLE_EVAL_PROFILES),
        "posts_count": len(catalog),
        "total_evaluations": len(all_composite),
        "signals": signals,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(distribution_data, f, indent=2)

    print(f"Saved distribution stats to {output_path}")
    print("\n--- Empirical Score Percentiles ---")
    for signal, stats in signals.items():
        print(
            f"{signal.upper():10s} | min: {stats['min']:.3f} | median: {stats['median']:.3f} | "
            f"p75: {stats['p75']:.3f} | p90: {stats['p90']:.3f} | max: {stats['max']:.3f}"
        )

    return signals


if __name__ == "__main__":
    run_distribution_study()
