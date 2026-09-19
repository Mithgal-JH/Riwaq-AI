<div align="center">

# 🤝 Person-to-Person (Profile) Recommendation
### Team3-AI · BinX Tech — Content-Based Recommendation Workstream

![Team](https://img.shields.io/badge/TEAM3--AI-0B2E59?style=for-the-badge)
![Task](https://img.shields.io/badge/TASK-PROFILE_↔_PROFILE-00A6C0?style=for-the-badge)
![Phase](https://img.shields.io/badge/PHASE-2--3_CORE_BUILD_%26_VALIDATION-FFC857?style=for-the-badge)
![Status](https://img.shields.io/badge/STATUS-IN_PROGRESS-1BAF7A?style=for-the-badge)
![Platform](https://img.shields.io/badge/PLATFORM-EDUCATIONAL_KNOWLEDGE_SHARING-8B5CF6?style=for-the-badge)

</div>

---

## 📖 Overview

This notebook implements the **AI logic behind the platform's "people you may want to connect with" feature** — recommending other users based on how similar their profiles are, using **Skills**, **Interests**, **Learning Direction**, and **bio**.

It builds directly on the finalized design (`Zayan_PersonToPerson_Recommendation_Task`) and the schema confirmed with the Backend team on Discord (Sept 17, 2026):

| Table / Relation | Fields Used | Cardinality |
|---|---|---|
| `Profile` | `first_name`, `last_name`, `bio`, `university` | — |
| `Skill` (via `UserSkill`) | tag list | one or more per profile |
| `Interest` (via `UserInterest`) | tag list | one or more per profile |
| `LearningDirection` | single tag | **one or none** per profile (nullable) |

## 🧠 Method

1. **Content Profile** — combine each user's Skills, Interests, Learning Direction, and bio into one text blob.
2. **TF-IDF** — vectorize every content profile (term frequency, weighted down for terms common across all profiles).
3. **Cosine Similarity** — score every pair of profiles from 0 to 1.
4. **Top-N Ranking** — return the N most similar profiles to a given user, excluding themselves.

This is the simplest approach that works end-to-end with no training data, and it's the same TF-IDF + Cosine Similarity technique from Sprint 3 (Week 8, Day 2), applied here to people instead of text reviews.

## 🐛 A Real Bug, Found and Fixed

The Backend team confirmed `learning_direction` returns as an explicit `null`. The obvious guard — `value or ""` — works for a raw Python `None`, but **pandas silently turns a missing value in a DataFrame column into `NaN` (a float)**, and `NaN` is truthy in Python. The naive guard let a `TypeError` through when building content profiles from the DataFrame. Fixed with an explicit `is_missing()` check that catches `None`, `NaN`, and `""` alike. Full before/after is in Section 2 of the notebook — including the actual error message it threw.

## ✅ Validation

| Check | Result |
|---|---|
| Sanity check on Top-N | Recommended profiles share real skills/interests with the target |
| Empty / minimal profile | Runs cleanly, similarity ≈ 0.0 against everyone (no text to match on) |
| Identical profiles | Similarity ≈ 1.0000, correctly rank each other as the #1 match |
| Scale (10 → 500 profiles) | 1.6 ms → 10.0 ms for the full pipeline — no changes needed before real data arrives |

## 📂 Files

| File | Purpose |
|---|---|
| `person_to_person_recommendation.ipynb` | Full, executed notebook — synthetic data, pipeline, validation |
| `README.md` | This file |

## 🔜 Next (Phase 4)

- Lock in the `GET /recommendations` response contract with Backend (ranked IDs + similarity scores vs. full profile summaries).
- Receive the fixed taxonomy lists for Skills, Interests, and Learning Direction so the model validates against real categories.
- Swap synthetic input for real seed data — pipeline logic is expected to run unchanged.

---

<div align="center">
<sub>BinX Tech · AI &amp; ML Internship — Team3-AI — Content-Based Recommendation Workstream</sub>
</div>
