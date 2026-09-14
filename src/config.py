from pathlib import Path
from typing import Final

# Base Directory Paths
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MOCK_POSTS_FILE: Final[Path] = DATA_DIR / "mock_posts.json"

# Model Identification
MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
MODEL_VERSION: Final[str] = f"sentence-transformer-v1.0-{MODEL_NAME}"
EMBEDDING_DIM: Final[int] = 384

# The 8 Agreed Taxonomy Topics
TAXONOMY_TOPICS: Final[list[str]] = [
    "Programming/Web",
    "AI/Data",
    "Electronics/Embedded",
    "Robotics",
    "Cybersecurity",
    "Design",
    "Mathematics",
    "Natural Sciences",
]

# Composite Ranking Weights (0.70 + 0.15 + 0.10 + 0.05 = 1.00)
WEIGHT_SEMANTIC: Final[float] = 0.70
WEIGHT_TOPIC: Final[float] = 0.15
WEIGHT_CREATOR: Final[float] = 0.10
WEIGHT_FRESHNESS: Final[float] = 0.05

# Freshness Decay Half-life in Days
FRESHNESS_HALF_LIFE_DAYS: Final[float] = 14.0

# Creator Teaching Quality Defaults & Thresholds
DEFAULT_CREATOR_QUALITY: Final[float] = 0.50
HIGH_RATED_CREATOR_THRESHOLD: Final[float] = 0.80

# Request Batch Sizing and Limits
MAX_CANDIDATES_PER_REQUEST: Final[int] = 100
DEFAULT_RECOMMENDATION_LIMIT: Final[int] = 10
MAX_RECOMMENDATION_LIMIT: Final[int] = 50

# Profile Modeling Weights (Warm Learner Profile EMA)
WARM_LEARNER_DECLARED_WEIGHT: Final[float] = 0.60
WARM_LEARNER_INTERACTION_WEIGHT: Final[float] = 0.40

# Feed Diversity Rules
MAX_CONSECUTIVE_SAME_TOPIC: Final[int] = 3
