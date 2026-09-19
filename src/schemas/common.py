from enum import Enum


class TopicTaxonomy(str, Enum):
    """The 8 agreed subject taxonomy classes for BinX educational posts."""

    PROGRAMMING_WEB = "Programming/Web"
    AI_DATA = "AI/Data"
    ELECTRONICS_EMBEDDED = "Electronics/Embedded"
    ROBOTICS = "Robotics"
    CYBERSECURITY = "Cybersecurity"
    DESIGN = "Design"
    MATHEMATICS = "Mathematics"
    NATURAL_SCIENCES = "Natural Sciences"


class ReasonCode(str, Enum):
    """Explainable reason codes mapped to frontend post card badges."""

    SIMILAR_TO_INTERESTS = "SIMILAR_TO_INTERESTS"
    TOPIC_MATCH = "TOPIC_MATCH"
    FRESH_CONTENT = "FRESH_CONTENT"
    HIGH_RATED_CREATOR = "HIGH_RATED_CREATOR"


class InteractionType(str, Enum):
    """Types of positive user interactions tracked for warm profile updating."""

    LIKE = "like"
    SAVE = "save"
    REPOST = "repost"
