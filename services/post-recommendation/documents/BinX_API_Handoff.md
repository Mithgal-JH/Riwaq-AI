# BinX Post Recommendation API - Backend Handoff

Welcome to the **Post Recommendation Engine** API documentation! This guide provides the .NET backend team with everything they need to integrate our AI recommendation service into the core BinX platform.

## 🚀 Getting Started

The Post Recommendation API is built with FastAPI. It handles complex semantic vector search, topic similarity, and creator affinity to deliver personalized timelines for users.

**Base URL (Local):** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`  
**OpenAPI Spec:** `http://localhost:8000/openapi.json`

*(Note: In production on Railway/Render, replace `localhost:8000` with the deployed domain).*

---

## 📡 Core Endpoints

### 1. `GET /health`
Validates that the microservice is alive and all machine learning models are loaded into memory.
- **Expected Response:** `{"status": "healthy", "model_version": "1.0.0", "vectors_count": 50, "catalog_count": 50}`
- **Usage:** Use this for your Docker / Kubernetes health checks.

### 2. `POST /api/v1/recommendations/posts`
The primary endpoint for generating a personalized timeline.
- **Payload:**
```json
{
  "request_id": "req_12345",
  "user_id": "usr_12345",
  "limit": 10,
  "declared_topics": ["Programming/Web", "AI/Data"],
  "learning_direction": "Frontend Development",
  "recent_interactions": [
    {
      "post_id": "post_789",
      "event_type": "like",
      "timestamp": "2026-09-21T18:00:00Z"
    }
  ]
}
```
- **Response:** Returns an array of `items` sorted by relevance (Highest Score First).
- **Notes:** You must pass the returned `id` array from the `items` list to your database (e.g., PostgreSQL) to fetch the actual post metadata/images to render on the client side. Note that the `.NET` backend is responsible for tracking user interactions and passing them in the `recent_interactions` list.

### 3. `POST /api/v1/events/post-upserted`
When a new post is created in the main .NET system, hit this endpoint so the AI service can generate text embeddings and store them in the Vector Store.
- **Payload:**
```json
{
  "post_id": "post_999",
  "creator_id": "creator_42",
  "title": "Learning FastAPI",
  "body": "Just learned how to use FastAPI with .NET!",
  "created_at": "2026-09-21T18:00:00Z",
  "primary_topic": "Programming/Web"
}
}
```

---

## 🛑 Known Limitations (Development Context)
While this service is functional, the backend and frontend teams must be aware of the following development simplifications used during the current sprint:
1. **Unvalidated Weights:** Calibration thresholds (e.g., `SEMANTIC_HIGH_THRESHOLD = 0.285`) are mathematically sound based on a 320-pair ground-truth dataset, but are completely unvalidated against real user traffic. These are Day-3 placeholders.
2. **Self-authored Creator Scores:** The creator reputation scores are currently mocked in the dataset rather than dynamically pulled from a real user-graph.
3. **Mock Topic Labels:** Topic classifications on posts are static string properties on the mock data, pending real-time injection from Haitham's Content Analysis API.
4. **Single-annotator Eval:** The 320-pair ground-truth benchmark was built by a single annotator (me). It has structural biases and lacks peer validation.
5. **Non-replicated In-memory Store:** Vector similarity search currently runs in memory (NumPy) inside the FastAPI process. This will not scale horizontally across multiple instances without state loss.

---

## 🏅 Reason Codes & Badges

The backend uses this API's output to order the feed, and the frontend uses `reason_codes` to display UI badges on the post cards.

| Output Reason Code | Frontend Badge Displayed on Post Card |
|---|---|
| `SIMILAR_TO_INTERESTS` | "Based on your saved & liked posts" |
| `TOPIC_MATCH` | "Matches your learning direction" |
| `FRESH_CONTENT` | "Recently published" |
| `HIGH_RATED_CREATOR` | "From a top-rated mentor" |

---

## 🔒 Authentication & Headers
*(To be finalized based on platform architecture)*
Currently, the API endpoints do not require an `idtoken` for internal microservice-to-microservice communication. If the deployment sits behind a public gateway, you will need to forward the Authorization header.

## 💾 Deployment Notes for DevOps
- **Memory (RAM):** The minimum recommended RAM for this service is **1GB - 2GB** due to the PyTorch NLP models loaded into memory.
- **Build Context:** When deploying to Railway, set the Root Directory to `/services/post-recommendation`.
