"""
Core recommendation engine.

This is the notebook's validated logic (person_to_person_recommendation.ipynb,
Sections 2-5) wrapped as a live, syncable index instead of a static DataFrame —
same content-profile builder, same is_missing() fix, same TF-IDF + Cosine
Similarity pipeline.
"""
import math
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_VERSION = "binx-recsys-content-v1"
PREPROCESSING_VERSION = "content-profile-v1"


def is_missing(value) -> bool:
    """True for None, NaN (pandas' coercion of None), or an empty string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value == ""


def build_content_profile(profile: dict) -> str:
    parts = [
        " ".join(profile.get("skills") or []),
        " ".join(profile.get("interests") or []),
        "" if is_missing(profile.get("learning_direction")) else profile["learning_direction"],
        profile.get("bio") or "",
    ]
    return " ".join(p for p in parts if p).strip()


class RecommendationIndex:
    """
    In-memory, synced, read-only index of profile content — built and
    refreshed only through Data Sync (full / upsert / delete), never by
    fetching from the Backend on a recommendation request. See "Data Sync
    (Backend -> AI)" in the API contract.
    """

    def __init__(self):
        self.profiles: Dict[str, dict] = {}
        self._order: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._similarity = None  # np.ndarray or None

    # ---------------------------------------------------------- sync ops

    def full(self, profiles: List[dict]) -> int:
        self.profiles = {p["profile_id"]: p for p in profiles}
        self._rebuild()
        return len(profiles)

    def upsert(self, profiles: List[dict]) -> int:
        for p in profiles:
            self.profiles[p["profile_id"]] = p
        self._rebuild()
        return len(profiles)

    def delete(self, profile_ids: List[str]) -> int:
        removed = 0
        for pid in profile_ids:
            if pid in self.profiles:
                del self.profiles[pid]
                removed += 1
        self._rebuild()
        return removed

    # ---------------------------------------------------------- internal

    def _rebuild(self) -> None:
        self._order = list(self.profiles.keys())
        if not self._order:
            self._vectorizer = None
            self._similarity = None
            return

        content_profiles = [build_content_profile(self.profiles[pid]) for pid in self._order]

        try:
            vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(content_profiles)
            self._vectorizer = vectorizer
            self._similarity = cosine_similarity(tfidf_matrix)
        except ValueError:
            # Empty vocabulary — every synced profile is empty/near-empty
            # (e.g. cold start with only placeholder profiles). Every
            # profile is then simply low_confidence with no recommendations,
            # rather than a 500.
            self._vectorizer = None
            self._similarity = None

    # ---------------------------------------------------------- read path

    def recommend(self, profile_id: str, top_n: int) -> Optional[Tuple[List[dict], bool]]:
        """
        Returns (recommendations, low_confidence), or None if profile_id is
        not in the synced index (caller returns 404 — never a live fetch
        back to the Backend, per the Integration Rules).
        """
        if profile_id not in self.profiles:
            return None

        own = self.profiles[profile_id]
        own_content = build_content_profile(own)
        low_confidence = own_content == ""

        if self._similarity is None or len(self._order) <= 1:
            return [], low_confidence

        idx = self._order.index(profile_id)
        scores = self._similarity[idx].copy()
        scores[idx] = -1  # exclude self
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_idx = [i for i in ranked if scores[i] > -1][:top_n]

        own_skills = set(own.get("skills") or [])
        own_interests = set(own.get("interests") or [])
        own_ld = own.get("learning_direction")

        recommendations = []
        for i in top_idx:
            candidate_id = self._order[i]
            candidate = self.profiles[candidate_id]
            candidate_ld = candidate.get("learning_direction")
            same_ld = (own_ld is not None) and (candidate_ld is not None) and (own_ld == candidate_ld)

            recommendations.append({
                "candidate_profile_id": candidate_id,
                "similarity_score": round(float(scores[i]), 4),
                "shared_skills": sorted(own_skills & set(candidate.get("skills") or [])),
                "shared_interests": sorted(own_interests & set(candidate.get("interests") or [])),
                "same_learning_direction": same_ld,
            })

        return recommendations, low_confidence
