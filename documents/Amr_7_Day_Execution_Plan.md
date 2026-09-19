# BinX Tech — 7-Day Sprint Execution Plan (Day 1 to Day 7)
## Feature: Recommend Educational Posts for People

**Feature Owner:** Amr Sheqwara (AI/ML Engineer)  
**Assigned Scope:** Educational Post Recommendation Engine (Exclusively)  
**Git Branch:** `Amr-Sheqwara`  
**Sprint Window:** Tuesday, September 15, 2026 – Monday, September 21, 2026  
**Status:** In Progress (Days 1, 2, 3, 4, and 5 Completed; Day 6 ready for execution)  
**Target Stakeholders:** Backend Team (.NET), Frontend Team, AI/ML Colleagues (Haitham, Zayan)

---

## Executive Overview

This document outlines the detailed day-by-day technical execution plan for building, testing, and delivering the Post Recommendation Engine for the BinX educational platform. For each day from Day 1 to Day 7, it details:
1. **What I Am Going to Do:** Concrete architectural tasks, code components, algorithms, schemas, and endpoints.
2. **Why I Am Doing It:** Engineering rationale, performance requirements, decoupling strategies, mathematical formulations, and risk mitigation.
3. **Deliverables & Verification:** The tangible artifacts produced and the objective tests validating them.

---

## Sprint Schedule Summary

| Day | Date | Primary Focus | Status |
|---|---|---|---|
| **Day 1** | Tue, Sep 15, 2026 | Project Scaffolding, Pydantic Schemas & Synthetic Catalog | Completed |
| **Day 2** | Wed, Sep 16, 2026 | Dense Semantic Embeddings (`all-MiniLM-L6-v2`) & Vector Retrieval | Completed |
| **Day 3** | Thu, Sep 17, 2026 | Composite Multi-Factor Ranking Engine & Score Calibration | Completed |
| **Day 4** | Fri, Sep 18, 2026 | Feed Diversity, Freshness Decay, Deduplication & Reason Codes | Completed |
| **Day 5** | Sat, Sep 19, 2026 | FastAPI Microservice Endpoints & Decoupling Fallbacks | Completed |
| **Day 6** | Sun, Sep 20, 2026 | End-to-End Stress Testing, Latency Benchmarking & Docker Build | Scheduled |
| **Day 7** | Mon, Sep 21, 2026 | Final Smoke Verification, OpenAPI Handoff & Production Delivery | Scheduled |

---

## Detailed Day-by-Day Execution Plan

### Day 1 (Tuesday, September 15, 2026): Scaffolding, Contracts, and Synthetic Catalog

#### 1. What I Did / Am Going to Do
* **Directory Scaffolding:** Initialized modular repository structure (`src/config.py`, `src/schemas/`, `data/`, `tests/`).
* **Data Contracts Definition:** Built strict Pydantic v2 schemas for all service inputs and outputs:
  * `PostRecommendationRequest`: Validates `user_id`, `declared_topics`, `learning_direction`, `eligible_candidate_ids` (1 to 100 items), `exclude_post_ids`, `limit`, and `recent_interactions`.
  * `PostRecommendationResponse`: Validates output `request_id`, `model_version`, `count`, and `items` with `id`, `rank`, `score`, `reason_codes`, and `primary_topic`.
  * `PostUpsertEvent`: Validates post ingestion payload (`post_id`, `creator_id`, `title`, `body`, `created_at`).
* **Centralized Configuration:** Created `src/config.py` defining model names, weight vectors, candidate limits, half-life decay constants, and file paths.
* **Synthetic Educational Catalog:** Curated a realistic 40-post synthetic educational catalog across the platform's 8 core domains (`Programming/Web`, `AI/Data`, `Electronics/Embedded`, `Robotics`, `Cybersecurity`, `Design`, `Mathematics`, `Natural Sciences`).
* **Validation Suite:** Implemented unit tests (`tests/test_schemas.py`) covering contract serialization, boundary constraints, and catalog integrity.

#### 2. Why I Did It (Rationale & Technical Trade-offs)
* **Contract-First Development:** Establishing strict Pydantic schemas upfront allows the .NET Backend team and AI engineers to work in parallel without blocking each other on interface ambiguity.
* **Decoupling from Upstream Databases:** Developing against a realistic synthetic dataset ensures recommendation engine development proceeds immediately, without waiting for database schema migration or seed data from the backend.
* **Zero Magic Numbers:** Hardcoded constants inside ranking logic lead to silent calibration drift. Centralizing all hyperparameters in `src/config.py` ensures transparent adjustments.
* **Resilient Schema Parsing:** Configured `ConfigDict(extra="ignore")` across models to prevent production breaking errors if upstream backend services add extra diagnostic fields.

#### 3. Key Deliverables
* [src/config.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/config.py)
* [src/schemas/post_recommendation.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/schemas/post_recommendation.py)
* [src/schemas/post_event.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/schemas/post_event.py)
* [data/mock_posts.json](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/data/mock_posts.json)
* [tests/test_schemas.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/tests/test_schemas.py)

---

### Day 2 (Wednesday, September 16, 2026): Semantic Embeddings and Vector Retrieval

#### 1. What I Did / Am Going to Do
* **Embedding Model Integration:** Wrapped SentenceTransformers with `all-MiniLM-L6-v2` in `src/services/embedding_service.py` to convert educational text into 384-dimensional dense vectors.
* **L2 Vector Normalization:** Enforced strict unit normalization ($\|v\|_2 = 1.0$) on all generated vectors so cosine similarity reduces to a fast dot product operation.
* **Persistent Vector Store:** Built `src/services/vector_store.py` with in-memory lookup maps and disk persistence (`data/post_embeddings.npy` and `data/post_embeddings_metadata.json`).
* **Offline Embedding Pipeline:** Created `src/scripts/generate_embeddings.py` to index the 40-post synthetic catalog offline.
* **Learner Profile Modeling:** Developed `src/services/profile_service.py` handling two learner states:
  * **Cold-Start Learner:** Centroid vector derived from declared onboarding topics and learning direction.
  * **Warm Learner:** Declared-interest blend combining declared interests (weight 0.60) and recent interacted post vectors (weight 0.40), strictly re-normalized:
    $$u_{\text{blend}} = 0.60 \cdot u_{\text{declared}} + 0.40 \cdot \left(\frac{1}{N} \sum_{i=1}^{N} v_i\right)$$
    $$u_{\text{warm}} = \frac{u_{\text{blend}}}{\|u_{\text{blend}}\|_2} \quad (\text{with } \|u_{\text{blend}}\|_2 < 10^{-8} \text{ degenerate fallback})$$
* **Vectorized Matrix Similarity Engine:** Created `src/services/similarity_engine.py` using NumPy matrix multiplication ($V \cdot u$) bounded strictly in $[0.0, 1.0]$.
* **Automated Verification:** Added unit tests (`tests/test_embeddings.py`) confirming vector shapes, L2 norms, warm profile weighting, and sub-millisecond retrieval.

#### 2. Why I Did It (Rationale & Technical Trade-offs)
* **Semantic Understanding Over Keyword Matching:** Traditional BM25 or TF-IDF fails when a learner searches for "web styling" but a post is titled "CSS Flexbox layout". Dense embeddings capture conceptual similarity.
* **Why `all-MiniLM-L6-v2`:** Produces high semantic quality while being ultra-lightweight (~80MB footprint, ~15ms inference per item on CPU). It requires zero GPU infrastructure, minimizing server operational costs.
* **Computational Efficiency of L2 Normalization:** Normalizing vectors at ingestion time allows candidate retrieval to execute via pure matrix dot products ($O(N \cdot D)$ BLAS routines) rather than computing square roots and norms on every online request.
* **Cold-Start vs. Warm Profile Balance:** A pure interaction-based profile overfits to the last clicked post, while a pure declared-topics profile never adapts. The 0.60/0.40 blend maintains learner intent stability while adapting to recent engagement.

#### 3. Key Deliverables
* [src/services/embedding_service.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/services/embedding_service.py)
* [src/services/vector_store.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/services/vector_store.py)
* [src/services/profile_service.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/services/profile_service.py)
* [src/services/similarity_engine.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/services/similarity_engine.py)
* [src/scripts/generate_embeddings.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/src/scripts/generate_embeddings.py)
* [data/post_embeddings.npy](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/data/post_embeddings.npy)
* [tests/test_embeddings.py](file:///c:/Users/Amr%20Sheqwara/Desktop/Team3-AI/tests/test_embeddings.py)

---

### Day 3 (Thursday, September 17, 2026): Composite Ranking Formula and Score Calibration

#### 1. What I Am Going to Do
* **Composite Ranking Engine:** Build `src/services/ranking_engine.py` to synthesize four distinct signals into a single calibrated score:
  $$\text{PostScore} = 0.70 \cdot S_{\text{semantic}} + 0.15 \cdot S_{\text{topic}} + 0.10 \cdot S_{\text{creator}} + 0.05 \cdot S_{\text{freshness}}$$
* **Signal Implementations & Normalization:**
  * **$S_{\text{semantic}}$ (0.70):** Cosine similarity between learner profile vector and post vector, clipped to $[0.0, 1.0]$.
  * **$S_{\text{topic}}$ (0.15):** Containment metric over post topics: $1.00$ for primary match, $0.60 \times \text{overlap}$ for secondary/predicted match, and $0.00$ for zero overlap.
  * **$S_{\text{creator}}$ (0.10):** Creator teaching quality rating (bounded in $[0.0, 1.0]$, falling back to 0.50 baseline; self-authored mock values during staging).
  * **$S_{\text{freshness}}$ (0.05):** Exponential true half-life decay function: $S_{\text{freshness}}(\Delta t) = \exp(-\ln(2) \cdot \Delta t / 14)$ ensuring content aged 14 days evaluates to 0.50.
* **Candidate Scoring Pipeline:** Filter incoming candidate IDs, retrieve cached vectors and metadata, compute sub-scores, and sort candidates in descending order.
* **Component Score Breakdown:** Maintain intermediate score components for auditing and explainability generation.
* **Automated Unit Tests:** Construct `tests/test_ranking.py` testing edge cases: zero topic match, unrated creators, extreme timestamps, and weight boundary validation.

#### 2. Why I Am Doing It (Rationale & Technical Trade-offs)
* **Avoiding Semantic Monoculture:** Relying solely on semantic embedding similarity can favor older popular posts and ignore explicit topic alignment or verified creator quality.
* **Dominant Weight Distribution:** Setting semantic fit at 0.70 guarantees that recommendations remain strictly relevant to the learner's educational focus, while the remaining 0.30 modulates rankings based on quality, freshness, and topic precision.
* **Graceful Degradation for External Signals:** The ranking formula must never crash if Zayan's creator model is unavailable. The fallback default of 0.50 represents a neutral quality baseline, ensuring seamless operation.
* **Calibrated Output Range:** Enforcing all individual sub-scores into $[0.0, 1.0]$ ensures the composite score is mathematically bounded within $[0.0, 1.0]$, giving predictable score interpretation to downstream consumers.

#### 3. Key Deliverables
* `src/services/ranking_engine.py`
* `tests/test_ranking.py`
* Calibrated multi-factor scoring verification report.

---

### Day 4 (Friday, September 18, 2026): Diversity, Freshness Decay, Deduplication & Reason Codes

#### 1. What I Am Going to Do
* **Freshness Decay Engine:** Implement the mathematical half-life decay function:
  $$\text{Freshness}(\Delta t) = \exp\left(-\frac{\Delta t}{14\text{ days}}\right)$$
  Where $\Delta t$ is the time elapsed since post creation.
* **Topic Frequency Capping:** Implement a feed diversity filter that inspects consecutive ranked posts and caps identical primary topics at a maximum of 3 consecutive items. If violated, subsequent posts of that topic are shifted down.
* **Exclusion & Deduplication Engine:** Filter out posts that the learner has already viewed, saved, or authored before ranking commences.
* **Explainable Reason Code Generator:** Assign human-readable reason tags to each recommended post based on clear quantitative criteria:
  * `SIMILAR_TO_INTERESTS`: Triggered if $S_{\text{semantic}} \ge 0.75$ or user has matching interaction history.
  * `TOPIC_MATCH`: Triggered if candidate post primary topic is in the learner's declared topics.
  * `FRESH_CONTENT`: Triggered if post was published within the last 72 hours ($\text{Freshness} \ge 0.85$).
  * `HIGH_RATED_CREATOR`: Triggered if $S_{\text{creator}} \ge 0.80$.
* **Automated Unit Tests:** Build `tests/test_diversity.py` verifying topic cap enforcement, deduplication correctness, and reason code threshold behavior.

#### 2. Why I Am Doing It (Rationale & Technical Trade-offs)
* **Preventing Feed Fatigue:** Without frequency capping, a learner with "Programming/Web" interest might receive 10 consecutive JavaScript posts, causing cognitive boredom. Capping ensures healthy topic exploration.
* **Exponential vs. Linear Freshness Decay:** Educational material remains relevant longer than social media updates. A 14-day exponential half-life ensures a 3-week-old high-quality tutorial is not abruptly discarded, while keeping feed content dynamic.
* **Explainability Builds Learner Trust:** Presenting opaque recommendation lists leads to user skepticism. Providing concrete reason codes allows the Frontend team to render clear badges (e.g., "Matches your learning direction", "From a top-rated mentor"), significantly improving user engagement.
* **Computational Safeguard:** Performing deduplication and candidate exclusion *before* heavy vector retrieval saves unnecessary similarity calculations.

#### 3. Key Deliverables
* `src/services/diversity_service.py`
* `src/services/reason_service.py`
* `tests/test_diversity.py`

---

### Day 5 (Saturday, September 19, 2026): FastAPI Microservice Serving and Decoupling Handlers

#### 1. What I Am Going to Do
* **FastAPI Application Setup:** Build `src/main.py` hosting the production REST API with lifespan events for embedding model warmup.
* **Online Recommendation Route:** Implement `POST /api/v1/recommendations/posts`:
  * Parses and validates `PostRecommendationRequest`.
  * Generates learner profile vector (Cold-start or Warm).
  * Executes similarity retrieval and multi-factor ranking.
  * Applies diversity rules, deduplication, and reason code tagging.
  * Returns calibrated `PostRecommendationResponse` with latency header.
* **Asynchronous Event Route:** Implement `POST /api/v1/events/post-upserted`:
  * Ingests newly created or updated post content.
  * Computes embedding vector on the fly and updates the in-memory vector store and metadata cache.
* **Health & Diagnostics Route:** Implement `GET /health` exposing model status, vector count, and memory metrics.
* **Decoupling Fallback Handlers:** Implement robust internal defaults for external dependencies:
  * Fallback to synthetic topic labels if Haitham's classifier is offline.
  * Fallback to 0.50 creator score if Zayan's model is offline.
* **Error Handling & Middleware:** Add global exception handlers, request timing middleware, and CORS configuration.

#### 2. Why I Am Doing It (Rationale & Technical Trade-offs)
* **Sub-Millisecond Routing & Async I/O:** FastAPI running on Uvicorn provides high throughput with minimal overhead, essential for fulfilling the .NET Backend SLA (< 300ms round-trip).
* **Model Warmup at Startup:** Loading the SentenceTransformer model on the first user request creates a severe 2-second latency spike ("cold start"). Preloading and warming up the model during FastAPI lifespan initialization guarantees immediate sub-50ms responses for all incoming traffic.
* **Dynamic Vector Cache Synchronization:** As new educational posts are published, the engine must incorporate them without restarting the microservice. The `/events/post-upserted` endpoint ensures zero-downtime vector cache updates.
* **Architectural Decoupling:** In a multi-team project, upstream delays must never halt downstream delivery. Building explicit fallback handlers guarantees my service operates independently and reliably under all circumstances.

#### 3. Key Deliverables
* `src/main.py`
* `src/api/routes.py`
* Interactive OpenAPI documentation (`/docs`).
* `tests/test_api.py` validating HTTP status codes, request validation errors, and response formatting.

---

### Day 6 (Sunday, September 20, 2026): Testing, Latency Benchmarking, and Docker Packaging

#### 1. What I Am Going to Do
* **End-to-End Test Suite:** Write integration tests simulating realistic backend payloads, multi-page pagination requests, and edge-case user profiles.
* **Latency Benchmarking:** Measure end-to-end processing time for candidate batches of 20, 50, and 100 items on CPU to verify compliance with the 150ms SLA.
* **Containerization (`Dockerfile`):**
  * Create an optimized, multi-stage Docker build using `python:3.11-slim`.
  * Pre-download SentenceTransformer model weights into the container image to prevent runtime network dependencies.
  * Configure non-root user execution for container security.
* **Dependency Freezing:** Lock production dependencies in `requirements.txt` with pinned versions to ensure reproducible builds.
* **Static Code Analysis:** Run `ruff` linting and `mypy` strict type checking across the entire repository.

#### 2. Why I Am Doing It (Rationale & Technical Trade-offs)
* **Meeting Backend SLA Thresholds:** The .NET backend team configures a 300ms timeout. Benchmarking on commodity CPU hardware guarantees the AI engine delivers responses in < 150ms, leaving ample margin for network transit and JSON serialization.
* **Self-Contained Container Image:** Pre-baking model weights (`all-MiniLM-L6-v2`) inside the Docker image ensures that container deployment in staging/production never fails due to Hugging Face network timeouts or external rate limits.
* **Security & Production Best Practices:** Running containers as non-root prevents container breakout risks. Strict type hints (`mypy`) eliminate runtime `TypeError` and `AttributeError` exceptions.
* **Deterministic Deployment:** Pinned dependencies prevent sudden pipeline breakages caused by upstream library updates.

#### 3. Key Deliverables
* `Dockerfile`
* `requirements.txt` (pinned)
* `benchmarks/latency_benchmark.py` with latency report.
* 100% test pass confirmation across all test files.

---

### Day 7 (Monday, September 21, 2026): Verification, OpenAPI Handoff, and Final Delivery

#### 1. What I Am Going to Do
* **Final End-to-End Smoke Test:** Run automated smoke tests against the containerized service verifying cold-start profiles, warm profiles, deduplication, and reason codes.
* **API Documentation Handoff:**
  * Export clean OpenAPI JSON/YAML specifications for the .NET Backend team.
  * Provide sample `curl` commands and expected response payloads for backend and frontend engineers.
* **Stakeholder Coordination:**
  * Confirm payload contracts with Backend Lead.
  * Validate reason code badge mappings with Frontend Lead.
  * Verify taxonomy alignment with Haitham and rating normalization with Zayan.
* **Branch Finalization:** Ensure all commits, documentation, reports, and code on branch `Amr-Sheqwara` are clean, reviewed, and ready for pull request merge.

#### 2. Why I Am Doing It (Rationale & Technical Trade-offs)
* **Frictionless Handoff:** Engineering handoff fails when documentation is ambiguous. Providing tested OpenAPI specs and curl samples eliminates guesswork for the backend integration team.
* **Clean Code Review Process:** Keeping all work cleanly structured on `Amr-Sheqwara` ensures compliance with team Git hygiene (no direct commits to `main`), facilitating a seamless pull request review.
* **Verifiable Acceptance Criteria:** Executing the final smoke test against the containerized image proves that the service runs identically in staging as it did in local development.

#### 3. Key Deliverables
* Final End-to-End Smoke Test Report.
* OpenAPI Specification export (`openapi.json`).
* Final Handoff Documentation for Backend and Frontend teams.
* Clean, verified feature branch `Amr-Sheqwara`.

---

## Technical Summary of Key Formulas & Constants

```
+-----------------------------------------------------------------------------------------+
|                                    FORMULA SUMMARY                                      |
+-----------------------------------------------------------------------------------------+
| 1. Warm Learner Profile (EMA):                                                          |
|    u_warm = 0.60 * u_declared + 0.40 * (1/N * sum(v_i))                                 |
|                                                                                         |
| 2. Candidate Semantic Similarity:                                                       |
|    SemanticFit(u, v) = (u . v) / (||u||_2 * ||v||_2)  [= u . v when normalized]         |
|                                                                                         |
| 3. Composite Post Score:                                                                |
|    PostScore = 0.70*SemanticFit + 0.15*TopicFit + 0.10*CreatorQuality + 0.05*Freshness   |
|                                                                                         |
| 4. Freshness Exponential Half-Life Decay:                                               |
|    Freshness(dt) = exp( -dt / 14 days )                                                 |
|                                                                                         |
| 5. Diversity Constraint:                                                                |
|    Max consecutive posts with identical primary topic = 3                               |
+-----------------------------------------------------------------------------------------+
```

---

## Stakeholder Interface Agreements

| Stakeholder | Direction | Contract / Protocol | Key SLA / Expectation |
|---|---|---|---|
| **Backend (.NET)** | Inbound | `POST /api/v1/recommendations/posts` | Timeout 300ms; max 100 candidate IDs per request; backend handles auth and pre-filtering. |
| **Backend (.NET)** | Inbound | `POST /api/v1/events/post-upserted` | Asynchronous notification upon post publication/update to refresh vector cache. |
| **Frontend** | Outbound | Reason codes (`SIMILAR_TO_INTERESTS`, `TOPIC_MATCH`, etc.) | Render corresponding UI badges on feed cards; send `exclude_post_ids` on infinite scroll. |
| **Haitham (AI)** | Shared | 8-Class Content Taxonomy | Fallback to mock tags if classifier is in training; zero blocking dependency. |
| **Zayan (AI)** | Shared | Creator Quality Score `[0.0, 1.0]` | Fallback to 0.50 neutral baseline if model is calibrating; zero blocking dependency. |
