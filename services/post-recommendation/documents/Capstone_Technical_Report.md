# Post Recommendation Engine: Capstone Technical Report

**Author:** Amr Sheqwara  
**Role:** AI/ML Engineer Intern  
**Program:** BinX Tech AI & ML Internship (Phase 3 Capstone)

---

## Part 1: Technical Implementation & Methodology

### 1. Problem Statement
In modern social and professional platforms, users are often overwhelmed by the volume of content. Keyword-based search and chronological feeds fail to capture user intent and contextual relevance. The objective of this capstone project was to design, build, and deploy an end-to-end Machine Learning **Post Recommendation Engine** capable of serving highly relevant, safe, and diverse content to users in real-time, integrating seamlessly with upstream APIs.

### 2. Architecture & Technology Stack
- **Backend Framework:** FastAPI (Asynchronous Python for high-throughput API endpoints)
- **Frontend Dashboard:** Streamlit (For interactive visualization and manual testing)
- **Machine Learning / NLP:** Hugging Face `SentenceTransformers` (`all-MiniLM-L6-v2`), NumPy
- **Containerization & Deployment:** Docker (Multi-stage builds, non-root execution)

### 3. Methodology: How It Works & Why
#### A. Semantic Embeddings & Vector Storage
**How:** We use `all-MiniLM-L6-v2` to map raw post text into 384-dimensional dense vectors. These vectors are stored in memory using NumPy arrays.  
**Why:** Traditional TF-IDF or keyword matching fails on synonyms. Dense embeddings allow the system to calculate "conceptual" similarity (cosine distance) in constant time, ensuring rapid retrieval. NumPy was chosen for the MVP over a dedicated Vector DB to minimize infrastructure overhead while achieving sub-50ms latency.

#### B. Composite Ranking Engine
**How:** Recommendations are not purely similarity-based. The `RankingEngine` calculates a final score based on:
$Score = (Cosine\_Similarity \times 0.6) + (Creator\_Quality \times 0.4) - Diversity\_Penalty$  
**Why:** Optimizing solely for similarity creates a "filter bubble". Injecting quality scores (from upstream models) ensures high-tier content surfaces, while the diversity penalty forces the engine to explore secondary topics, improving user retention.

#### C. Safety Moderation & Upstream Integration
**How:** The system intercepts a 12-class `TopicTaxonomy` and a `safety_status` flag from the Content Analysis API. If a post fails the safety gate, it is strictly filtered out *before* the ranking vector math begins.  
**Why:** Safety is a hard constraint. Filtering early reduces computational waste on the embedding layer and prevents brand-damaging content from reaching end users.

#### D. Explainability (Reason Service)
**How:** For every recommended post, the system outputs an opaque `reason_id` alongside human-readable text (e.g., "Matches your declared interest in Cyber Security").  
**Why:** Transparency builds user trust. By correlating the user's `declared_topics` with the post's dominant taxonomy, we provide clear, SHAP-like explanations for the algorithm's behavior.

### 4. Results & Performance
- **Latency:** Achieved a P95 latency of ~20.6ms for 100-candidate batches, well below the 150ms SLA.
- **Robustness:** Achieved a 100% pass rate across 116 integration tests, gracefully handling dirty data, missing fields, and infinite-scroll pagination boundaries.

### 5. Limitations & Future Work
- **Limitation:** In-memory NumPy arrays are extremely fast but do not horizontally scale well beyond hundreds of thousands of vectors.
- **Future Work:** Migrate the vector index to a dedicated Vector Database (e.g., Pinecone, Qdrant, or PostgreSQL with `pgvector`) and introduce continuous online learning to update user profiles in real-time.

---

## Part 2: Non-Technical Executive Summary

### What did we build?
We built the **"Brain"** of a personalized content feed. Whenever a user opens our application, this system instantly decides the absolute best posts to show them out of thousands of possibilities.

### How does it work (in simple terms)?
1. **Understanding Meaning:** Instead of just looking for matching words, our AI actually reads the content and understands the *meaning* behind it. If a user likes "finance," it knows to show them "investing" or "markets" even if the word "finance" is never mentioned.
2. **Quality over Quantity:** It doesn't just show the most relevant posts; it actively boosts posts created by high-quality authors so users get the best possible experience.
3. **Keeping it Fresh:** To prevent users from getting bored by seeing the exact same type of content over and over, the system purposefully mixes in new and different topics.
4. **Safety First:** Before a user ever sees a post, the system double-checks it against a safety filter. Unsafe content is blocked instantly.

### Why is this important?
By showing users exactly what they want to see, while keeping them safe and introducing them to fresh ideas, we keep them engaged with the platform longer. Furthermore, because we built this to be lightning-fast (taking less than a fraction of a second to think), the user never experiences any loading delays.
