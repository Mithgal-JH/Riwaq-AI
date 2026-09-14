# BinX Tech — Feature Specification & Execution Plan
## Feature: Recommend Educational Posts for People

**Feature Owner:** Amr Sheqwara (AI/ML Engineer)  
**Assigned Scope:** Recommend Educational Posts for People (Exclusively)  
**Git Branch:** `Amr-Sheqwara`
**Status:** Approved Working Plan  
**Target Stakeholders:** Backend Team (.NET), Frontend Team, AI/ML Colleagues (Haitham, Zayan)  

---

## 1. Feature Objective & Scope

The objective of this feature is to build and deploy a dedicated, high-speed, content-based recommendation engine that delivers personalized educational posts to a learner's feed.

### Core Boundaries:
* **In-Scope:** Learner profile modeling, semantic post embeddings, candidate retrieval via cosine similarity, multi-factor ranking, feed diversity rules (freshness decay, topic frequency capping), explainable reason codes, and the FastAPI recommendation endpoint.
* **Out-of-Scope (Owned by Colleagues):** 
  * Recommending People for People (Mentors/Peers) -> Owned by Zayan.
  * Content Topic Classification & OCR Vision Pipeline -> Owned by Haitham.
  * User Authentication, Authorization, Database Persistence, and Feed Card Hydration -> Owned by Backend Team.
  * User Interface Rendering & Feed Display -> Owned by Frontend Team.

---

## 2. Technology Stack & Rationale

| Tool / Technology | Version / Model | Role in Feature | Justification |
|---|---|---|---|
| **Python** | 3.11+ | Primary Development Language | Native ecosystem for modern NLP, vector math, and async web services. |
| **FastAPI** | Latest stable | Microservice REST API Framework | High-speed async request handling, auto-generated OpenAPI documentation, native Pydantic validation. |
| **Pydantic** | v2 | Request/Response Data Validation | Strict data contract enforcement between Backend and AI microservice. |
| **Uvicorn** | Latest stable | ASGI Web Server | Lightweight, production-grade server for async Python execution. |
| **Sentence Transformers** | `all-MiniLM-L6-v2` | Dense Text Semantic Embeddings | Produces 384-dimensional dense vectors; understands concepts rather than exact keywords; fast CPU inference (~15ms per chunk); lightweight (~80MB model size). |
| **scikit-learn** | Latest stable | Baselines & Evaluation Metrics | TF-IDF feature extraction baseline; NDCG@10, Precision@5, and Recall@10 evaluation. |
| **NumPy** | Latest stable | Vector Mathematics | High-speed dot product and cosine similarity matrix operations over candidate vectors. |
| **pandas** | Latest stable | Data Manipulation | Offline evaluation processing, test-set batch handling, and interaction logging. |
| **Docker** | Debian-slim | Containerization | Self-contained, reproducible container environment requiring no host system GPU. |
| **pytest, ruff, mypy** | Latest | Code Quality & Testing | Static type safety, automated formatting, and unit testing of ranking logic. |

---

## 3. What I Am Going to Build

### 3.1 Learner Profile Modeling
* **Cold-Start Learner:** Mean embedding vector generated from declared onboarding topics and current learning direction.
* **Active (Warm) Learner:** Dynamic profile vector updated incrementally using an Exponential Moving Average of declared interests and recent positive interactions (liked, saved, and reposted posts):
  $$u_{\text{warm}} = 0.6 \cdot u_{\text{declared}} + 0.4 \cdot \left(\frac{1}{N} \sum_{i=1}^{N} v_{\text{interacted\_post}_i}\right)$$

### 3.2 Post Semantic Embedding
* When a post is published or updated, combine `title + " " + description/body` and pass it through `all-MiniLM-L6-v2` to produce a persistent 384-dimensional vector $v_{\text{post}}$.

### 3.3 Candidate Retrieval
* Compute cosine similarity between the learner vector $u$ and all pre-filtered candidate post vectors $v$:
  $$\text{semantic\_fit}(u, v) = \frac{u \cdot v}{\|u\|_2 \cdot \|v\|_2}$$

### 3.4 Composite Ranking Pipeline
Calculate the final score for each eligible candidate post:
$$\text{post\_score} = 0.70 \times \text{semantic\_fit} + 0.15 \times \text{topic\_fit} + 0.10 \times \text{creator\_teaching\_quality} + 0.05 \times \text{freshness}$$

* **`semantic_fit` (0.70):** Cosine similarity between user profile vector and post vector.
* **`topic_fit` (0.15):** Match between candidate post topics and user's declared topics.
* **`creator_teaching_quality` (0.10):** Author's peer teaching reputation score provided by Zayan (defaults to 0.50 neutral baseline).
* **`freshness` (0.05):** Exponential half-life decay function with a 14-day half-life:
  $$\text{freshness}(\Delta t) = \exp\left(-\frac{\Delta t}{14 \text{ days}}\right)$$

### 3.5 Feed Diversity & Exclusion Rules
* **Exclusion Filter:** Remove posts already seen, saved, or authored by the learner.
* **Topic Frequency Capping:** Prevent feed saturation by allowing a maximum of 3 consecutive posts from the same primary topic.
* **Explainability Codes:** Tag each ranked post with transparent reason codes (`SIMILAR_TO_INTERESTS`, `TOPIC_MATCH`, `FRESH_CONTENT`, `HIGH_RATED_CREATOR`).

---

## 4. INPUTS: What I Need and From Whom

```
┌─────────────────────────────────────────────────────────────┐
│ 1. BACKEND TEAM (.NET)                                      │
│    - User context (ID, declared topics, learning direction) │
│    - Eligible candidate post IDs (pre-filtered)             │
│    - Exclude post IDs (viewed/saved/self)                   │
│    - Recent interactions (likes, saves)                     │
│    - Post content upserts (title, body, creator_id, time)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│ 2. HAITHAM (Content Classification)                         │
│    - Post topic predictions (8-topic taxonomy)              │
│    - Topic confidence scores                                │
│    - Primary topic tag                                      │
│    - OCR extracted text (from diagrams/screenshots)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│ 3. ZAYAN (People & Feedback Analysis)                       │
│    - Creator teaching quality score [0.0 to 1.0]            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
            ┌────────────────────────────────────┐
            │ AMR: Post Recommendation Engine    │
            └────────────────────────────────────┘
```

### 4.1 From the Backend Team (.NET)

#### A. Recommendation Request (Online, per feed request):
* **Endpoint:** `POST /api/v1/recommendations/posts`
* **Fields Received:**
  * `request_id` (string/UUID): Tracking identifier.
  * `user_id` (string/UUID): The learner requesting the feed.
  * `declared_topics` (List[string]): Onboarding topics chosen by user (e.g., `["Programming/Web", "AI/Data"]`).
  * `learning_direction` (string): Primary learning direction (e.g., `"Backend Development"`).
  * `eligible_candidate_ids` (List[string]): Post IDs already filtered by the backend as published, non-draft, and non-blocked.
  * `exclude_post_ids` (List[string]): Post IDs to exclude (already displayed, saved, or authored by the user).
  * `limit` (integer): Number of posts requested (e.g., 10).
  * `recent_interactions` (List[object], optional): Post IDs recently liked/saved with interaction timestamps.

#### B. Content Ingestion Event (Asynchronous, when a post is created/updated):
* **Endpoint:** `POST /api/v1/events/post-upserted`
* **Fields Received:**
  * `post_id` (string/UUID): Unique post identifier.
  * `creator_id` (string/UUID): Author's user ID.
  * `title` (string): Post title.
  * `body` (string): Text content of the post.
  * `created_at` (ISO timestamp): Publication timestamp.

### 4.2 From Haitham (Content Classification)
Metadata received per post (either directly or via shared database record):
* `post_id` (string): Post identifier.
* `predicted_topics` (List[string]): Topics from the 8-class taxonomy (`Programming/Web`, `AI/Data`, `Electronics/Embedded`, `Robotics`, `Cybersecurity`, `Design`, `Mathematics`, `Natural Sciences`).
* `topic_confidence_scores` (Dict[string, float]): Confidence scores per topic.
* `primary_topic` (string): The single highest-confidence topic.
* `ocr_extracted_text` (string, optional): Text extracted from slides, code snippets, or diagrams.

*Decoupling Fallback:* While Haitham is training his classifier, I will use author-declared tags and mock topic labels.

### 4.3 From Zayan (People & Feedback Analysis)
Reputation metric received per creator:
* `creator_id` (string): Author identifier.
* `teaching_quality_score` (float): Bounded score in `[0.0, 1.0]` representing the creator's verified peer-teaching rating and review sentiment.

*Decoupling Fallback:* While Zayan is calibrating his model, I will default all creators to a neutral baseline of `0.50`.

---

## 5. OUTPUTS: What I Will Give Out and To Whom

```
            ┌────────────────────────────────────┐
            │ AMR: Post Recommendation Engine    │
            └──────────────────┬─────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. TO BACKEND TEAM (.NET)                                   │
│    - HTTP 200 OK JSON payload                               │
│    - Ordered list of post IDs                               │
│    - Composite ranking score                                │
│    - Explainable reason codes                               │
│    - Primary topic tag                                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TO FRONTEND TEAM (Hydrated via Backend)                  │
│    - Feed card ordering                                     │
│    - Reason code UI badges on feed cards                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 To the Backend Team (.NET)
The service returns an HTTP `200 OK` JSON payload containing the ranked candidate IDs, composite scores, and explainable reason codes.

```json
{
  "request_id": "req_8849b2c1-d41a-4d2a-89a1-8e50b8ef1092",
  "model_version": "sentence-transformer-v1.0-all-MiniLM-L6-v2",
  "count": 3,
  "items": [
    {
      "id": "pst_101",
      "rank": 1,
      "score": 0.865,
      "reason_codes": [
        "SIMILAR_TO_INTERESTS",
        "TOPIC_MATCH"
      ],
      "primary_topic": "Programming/Web"
    },
    {
      "id": "pst_104",
      "rank": 2,
      "score": 0.782,
      "reason_codes": [
        "FRESH_CONTENT",
        "TOPIC_MATCH"
      ],
      "primary_topic": "Programming/Web"
    },
    {
      "id": "pst_103",
      "rank": 3,
      "score": 0.691,
      "reason_codes": [
        "HIGH_RATED_CREATOR"
      ],
      "primary_topic": "AI/Data"
    }
  ]
}
```

### 5.2 To the Frontend Team (Delivered via Backend Hydration)
The backend uses my output to order the feed, and the frontend uses `reason_codes` to display badges on the post cards:

| Output Reason Code | Frontend Badge Displayed on Post Card |
|---|---|
| `SIMILAR_TO_INTERESTS` | "Based on your saved & liked posts" |
| `TOPIC_MATCH` | "Matches your learning direction" |
| `FRESH_CONTENT` | "Recently published" |
| `HIGH_RATED_CREATOR` | "From a top-rated mentor" |

---

## 6. What Colleagues Need to Know

### 6.1 What the Backend Team (.NET) Needs to Know
1. **Authorization Responsibility:** My service only ranks the `eligible_candidate_ids` that the backend supplies. My service **never performs permission or privacy checks**. The backend must continue to pre-filter items (ensure published, non-draft, non-blocked) before calling the recommendation API.
2. **Endpoint & Timeout:** Call `POST /api/v1/recommendations/posts`. Configure your HTTP client with a **300ms timeout threshold**.
3. **Batch Sizing:** Pass a maximum of 100 candidate IDs per request for optimal latency.
4. **Content Synchronization:** Whenever a post is published, updated, or deleted, send an asynchronous notification (`POST /api/v1/events/post-upserted`) so the vector cache stays updated.

### 6.2 What the Frontend Team Needs to Know
1. **Explainability Badges:** Use the `reason_codes` array in the response to render the corresponding badge on each feed card.
2. **Infinite Scroll Deduplication:** On subsequent page loads or infinite scrolls, include the IDs of all posts already displayed in the `exclude_post_ids` array.
3. **Impression & Interaction Telemetry:** Log an **impression event** when a post card enters the viewport, and an **interaction event** when clicked, liked, saved, or reposted. These logs are needed for offline model evaluation.

### 6.3 What AI/ML Colleagues (Haitham & Zayan) Need to Know
1. **Haitham (Content Classification):** Provide post topic predictions conforming to the 8-class taxonomy. While your model is in training, I will use mock tags; you will not block my development.
2. **Zayan (People & Feedback Analysis):** Provide creator teaching quality scores normalized to `[0.0, 1.0]`. While your model is in training, I will default to `0.50`; you will not block my development.
3. **Git Hygiene:** No one pushes to `main`. All work stays on feature branches.

---

## 7. 7-Day Sprint Schedule & Milestones (Deadline: Next Monday, Sep 21)

```
Day 1 (Tue, Sep 15): Project Scaffolding, Pydantic Schemas & Mock Post Catalog
Day 2 (Wed, Sep 16): Dense Semantic Embeddings (MiniLM) & Cosine Retrieval Pipeline
Day 3 (Thu, Sep 17): Composite Ranking Formula Implementation & Score Calibration
Day 4 (Fri, Sep 18): Feed Diversity, Freshness Decay, Deduplication & Reason Codes
Day 5 (Sat, Sep 19): FastAPI Microservice Endpoint (POST /api/v1/recommendations/posts)
Day 6 (Sun, Sep 20): End-to-End Testing with Mock Backend Payloads & Docker Build
Day 7 (Mon, Sep 21): Final Verification, API Documentation Handoff & Delivery
```

### Day-by-Day Deliverables:

| Day | Focus Area | Concrete Deliverable |
|---|---|---|
| **Day 1 (Tue)** | Setup & Contracts | Create `src/` structure, define Pydantic request/response models, construct synthetic post dataset (30–50 representative educational posts). |
| **Day 2 (Wed)** | Semantic Retrieval | Load `all-MiniLM-L6-v2`, generate post embeddings, implement user interest vectorization and exact cosine similarity matrix search. |
| **Day 3 (Thu)** | Ranking Engine | Implement the 4-part scoring formula (`0.70 semantic + 0.15 topic + 0.10 creator + 0.05 freshness`). Calibrate components to `[0.0, 1.0]`. |
| **Day 4 (Fri)** | Diversity & Rules | Implement 14-day exponential freshness decay, topic frequency capping (max 3 consecutive per topic), exclusion filtering, and reason code tagging. |
| **Day 5 (Sat)** | FastAPI Serving | Wire ranking engine into `POST /api/v1/recommendations/posts`, add error handlers, and implement mock fallbacks for Haitham's topics and Zayan's scores. |
| **Day 6 (Sun)** | Testing & Packaging | Write `pytest` test suite verifying scoring edge cases, benchmark CPU latency (< 150ms for 100 items), write `Dockerfile` and `requirements.txt`. |
| **Day 7 (Mon)** | Handoff (Deadline) | Final end-to-end smoke test, export API documentation for Backend team, package deliverables on branch `Amr-Sheqwara`. |

### Acceptance & Success Criteria for Monday:
* **Working Service:** A runnable FastAPI microservice returning ranked posts with scores and reason codes in under 150ms on CPU.
* **Independence:** Feature runs end-to-end on mock inputs without depending on whether Haitham or Zayan finished their modules.
* **Safety:** Zero private post bodies exposed; 100% test pass rate on ranking formulas.
