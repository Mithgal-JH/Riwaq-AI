"""
Smoke tests mirroring the notebook's own validation section (6.1-6.4):
sanity check, empty/minimal profile, identical ("twin") profiles, plus the
HTTP-contract specifics (404 / 400 / sync lifecycle) that only exist once
the logic is wrapped as a live API.

Run with:  pytest test_api.py -v
"""
from fastapi.testclient import TestClient

from app.main import app, index

client = TestClient(app)


def _reset():
    index.full([])


def test_health():
    _reset()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["profiles_indexed"] == 0


def test_full_sync_and_recommend_sanity():
    _reset()
    profiles = [
        {"profile_id": "p1", "skills": ["Python", "SQL"], "interests": ["Machine Learning"],
         "learning_direction": "AI & Machine Learning", "bio": "Loves Python and ML."},
        {"profile_id": "p2", "skills": ["Python", "React"], "interests": ["Machine Learning", "Web Development"],
         "learning_direction": "AI & Machine Learning", "bio": "Python developer into ML and web."},
        {"profile_id": "p3", "skills": ["UI Design"], "interests": ["Digital Marketing"],
         "learning_direction": None, "bio": "Designer exploring marketing."},
    ]
    r = client.post("/api/v1/ai/recommendations/people/sync", json={"sync_type": "full", "profiles": profiles})
    assert r.status_code == 200
    body = r.json()
    assert body["sync_status"] == "completed"
    assert body["profiles_indexed"] == 3

    r = client.get("/api/v1/ai/recommendations/people/p1", params={"top_n": 2, "request_id": "req_1"})
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == "p1"
    assert body["request_id"] == "req_1"
    assert body["model_version"] == "binx-recsys-content-v1"
    assert body["preprocessing_version"] == "content-profile-v1"
    # p1 never recommends itself
    assert all(rec["candidate_profile_id"] != "p1" for rec in body["recommendations"])
    # p1 and p2 share Python + Machine Learning -> p2 should rank above p3
    ranked_ids = [rec["candidate_profile_id"] for rec in body["recommendations"]]
    assert ranked_ids[0] == "p2"
    assert "Python" in body["recommendations"][0]["shared_skills"]
    assert body["low_confidence"] is False


def test_learning_direction_null_never_crashes_and_is_false_when_null():
    _reset()
    profiles = [
        {"profile_id": "a", "skills": ["Java"], "interests": [], "learning_direction": None, "bio": "Backend dev."},
        {"profile_id": "b", "skills": ["Java"], "interests": [], "learning_direction": None, "bio": "Also backend."},
    ]
    r = client.post("/api/v1/ai/recommendations/people/sync", json={"sync_type": "full", "profiles": profiles})
    assert r.status_code == 200

    r = client.get("/api/v1/ai/recommendations/people/a", params={"request_id": "req_2"})
    assert r.status_code == 200
    rec = r.json()["recommendations"][0]
    assert rec["same_learning_direction"] is False  # both null -> always false per contract


def test_minimal_empty_profile_is_low_confidence_not_error():
    _reset()
    profiles = [
        {"profile_id": "normal", "skills": ["Python"], "interests": ["Machine Learning"],
         "learning_direction": "AI & Machine Learning", "bio": "ML enthusiast."},
        {"profile_id": "empty", "skills": [], "interests": [], "learning_direction": None, "bio": ""},
    ]
    r = client.post("/api/v1/ai/recommendations/people/sync", json={"sync_type": "full", "profiles": profiles})
    assert r.status_code == 200

    r = client.get("/api/v1/ai/recommendations/people/empty", params={"request_id": "req_3"})
    assert r.status_code == 200
    body = r.json()
    assert body["low_confidence"] is True


def test_identical_twin_profiles_score_near_one():
    _reset()
    twin = {"skills": ["Python", "SQL"], "interests": ["Machine Learning"],
            "learning_direction": "AI & Machine Learning", "bio": "Passionate about Python and Machine Learning."}
    profiles = [
        {"profile_id": "twin_a", **twin},
        {"profile_id": "twin_b", **twin},
        {"profile_id": "other", "skills": ["UI Design"], "interests": ["Digital Marketing"],
         "learning_direction": None, "bio": "Designer."},
    ]
    r = client.post("/api/v1/ai/recommendations/people/sync", json={"sync_type": "full", "profiles": profiles})
    assert r.status_code == 200

    r = client.get("/api/v1/ai/recommendations/people/twin_a", params={"request_id": "req_4"})
    body = r.json()
    assert body["recommendations"][0]["candidate_profile_id"] == "twin_b"
    assert body["recommendations"][0]["similarity_score"] >= 0.999


def test_upsert_then_delete_lifecycle():
    _reset()
    r = client.post("/api/v1/ai/recommendations/people/sync", json={
        "sync_type": "upsert",
        "profiles": [{"profile_id": "x", "skills": ["Python"], "interests": [], "learning_direction": None, "bio": ""}],
    })
    assert r.status_code == 200
    assert r.json()["profiles_indexed"] == 1

    r = client.get("/api/v1/ai/recommendations/people/x", params={"request_id": "req_5"})
    assert r.status_code == 200

    r = client.post("/api/v1/ai/recommendations/people/sync", json={"sync_type": "delete", "profile_ids": ["x"]})
    assert r.status_code == 200
    assert r.json()["profiles_indexed"] == 1  # one profile removed

    r = client.get("/api/v1/ai/recommendations/people/x", params={"request_id": "req_6"})
    assert r.status_code == 404  # never a live fetch back to Backend


def test_unknown_profile_returns_404_not_500():
    _reset()
    r = client.get("/api/v1/ai/recommendations/people/does_not_exist", params={"request_id": "req_7"})
    assert r.status_code == 404


def test_missing_request_id_returns_400():
    _reset()
    client.post("/api/v1/ai/recommendations/people/sync", json={
        "sync_type": "full",
        "profiles": [{"profile_id": "p1", "skills": [], "interests": [], "learning_direction": None, "bio": ""}],
    })
    r = client.get("/api/v1/ai/recommendations/people/p1")
    assert r.status_code == 400


def test_top_n_out_of_range_returns_400():
    _reset()
    client.post("/api/v1/ai/recommendations/people/sync", json={
        "sync_type": "full",
        "profiles": [{"profile_id": "p1", "skills": [], "interests": [], "learning_direction": None, "bio": ""}],
    })
    r = client.get("/api/v1/ai/recommendations/people/p1", params={"request_id": "r", "top_n": 21})
    assert r.status_code == 400
    r = client.get("/api/v1/ai/recommendations/people/p1", params={"request_id": "r", "top_n": 0})
    assert r.status_code == 400


def test_invalid_sync_type_returns_400():
    _reset()
    r = client.post("/api/v1/ai/recommendations/people/sync", json={"sync_type": "upsert", "profiles": None})
    assert r.status_code == 400
