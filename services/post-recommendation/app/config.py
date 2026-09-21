from pathlib import Path
from typing import Final

# Base Directory Paths
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MOCK_POSTS_FILE: Final[Path] = DATA_DIR / "mock_posts.json"
POST_EMBEDDINGS_FILE: Final[Path] = DATA_DIR / "post_embeddings.npy"
POST_METADATA_FILE: Final[Path] = DATA_DIR / "post_embeddings_metadata.json"

# Model Identification
MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
MODEL_VERSION: Final[str] = f"sentence-transformer-v1.0-{MODEL_NAME}"
EMBEDDING_DIM: Final[int] = 384

# The 12-Class Taxonomy Topics (aligned with Haitham's Content Analysis API)
TAXONOMY_TOPICS: Final[list[str]] = [
    "Programming/Web",
    "AI/Data",
    "Electronics/Embedded",
    "Robotics",
    "Cybersecurity",
    "Design",
    "Mathematics",
    "Natural Sciences",
    "Health/Medicine",
    "Business/Economics",
    "Language/Communication",
    "Humanities/Social",
]

# Composite Ranking Weights (0.70 + 0.15 + 0.10 + 0.05 = 1.00)
# UNVALIDATED DEFAULT. Not a trained coefficient and not measured on any test set.
# Pending offline evaluation (see docs/EVALUATION.md).
WEIGHT_SEMANTIC: Final[float] = 0.70
WEIGHT_TOPIC: Final[float] = 0.15

# WARNING ON CREATOR SCORES:
# The creator_teaching_quality values in mock_posts.json are self-authored synthetic
# placeholders. They do not come from Zayan's model, do not represent real peer-review data,
# and correlate with nothing outside this catalog.
# Because these values vary (0.50 to 0.95), this 0.10 weight actively steers ranking based
# on hand-invented numbers. To prevent self-confirmation bias during Day 6 evaluation,
# an ablation variant with W_CREATOR = 0.0 must be tested and reported.
WEIGHT_CREATOR: Final[float] = 0.10
WEIGHT_FRESHNESS: Final[float] = 0.05

# Aliases per plan specification
W_SEMANTIC: Final[float] = WEIGHT_SEMANTIC
W_TOPIC: Final[float] = WEIGHT_TOPIC
W_CREATOR: Final[float] = WEIGHT_CREATOR
W_FRESH: Final[float] = WEIGHT_FRESHNESS

# Freshness Decay: true half-life formula where S_fresh(14d) == 0.50
FRESHNESS_HALF_LIFE_DAYS: Final[float] = 14.0
FRESH_CONTENT_MAX_DAYS: Final[float] = 3.0

# Topic containment credit for non-primary matching topics
TOPIC_SECONDARY_CREDIT: Final[float] = 0.60

# Creator Teaching Quality Defaults & Thresholds
DEFAULT_CREATOR_QUALITY: Final[float] = 0.50
CREATOR_SCORE_FALLBACK: Final[float] = DEFAULT_CREATOR_QUALITY
HIGH_RATED_CREATOR_THRESHOLD: Final[float] = 0.80

# Dynamic threshold populated on Day 3 from data/score_distribution.json
# Empirically measured p90 from data/score_distribution.json (Sep 17, 2026 run)
SEMANTIC_HIGH_THRESHOLD: Final[float] = 0.285

# Reason Code Activation Thresholds
# Note: SIMILAR_TO_INTERESTS uses SEMANTIC_HIGH_THRESHOLD (0.285, empirically measured p90)
# Note: FRESH_CONTENT strictly uses FRESH_CONTENT_MAX_DAYS (3.0 days), single source of truth


# Request Batch Sizing and Limits
MAX_CANDIDATES_PER_REQUEST: Final[int] = 100
DEFAULT_RECOMMENDATION_LIMIT: Final[int] = 10
MAX_RECOMMENDATION_LIMIT: Final[int] = 50

# Profile Modeling Weights (Declared-Interest Blend)
WARM_LEARNER_DECLARED_WEIGHT: Final[float] = 0.60
WARM_LEARNER_INTERACTION_WEIGHT: Final[float] = 0.40
PROFILE_DECLARED_WEIGHT: Final[float] = WARM_LEARNER_DECLARED_WEIGHT
PROFILE_INTERACTION_WEIGHT: Final[float] = WARM_LEARNER_INTERACTION_WEIGHT

# Feed Diversity Rules
MAX_CONSECUTIVE_SAME_TOPIC: Final[int] = 3
MAX_CONSECUTIVE_TOPIC: Final[int] = MAX_CONSECUTIVE_SAME_TOPIC
