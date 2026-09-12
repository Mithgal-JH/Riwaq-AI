<p align="center">
  <img src="https://img.shields.io/badge/BINX%20TECH-AI%20%26%20ML%20INTERNSHIP-0B2E59?style=for-the-badge" alt="BinX Tech"/>
</p>

<h1 align="center">🔎 AI Research — Content-Based Recommendation System</h1>
<h3 align="center">Technical Documentation for Backend Integration Planning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/SOURCE-TRELLO%20BACKLOG-0B2E59?style=for-the-badge" alt="Trello Backlog"/>
  <img src="https://img.shields.io/badge/TYPE-RESEARCH%20TASK-00A6C0?style=for-the-badge" alt="Research Task"/>
  <img src="https://img.shields.io/badge/AUTHOR-ZAYAN%20SHAWAREB-1BAF7A?style=for-the-badge" alt="Author"/>
</p>

<p align="center"><i>Prepared to help the team decide how to build and integrate the recommendation AI system with the Backend.</i></p>

---

## What a Content-Based Recommendation System Is

A **content-based recommender** suggests items to a user by comparing the *attributes of the items themselves* — text, genres, tags, ingredients, keywords, and so on — rather than looking at what other, similar users liked. If a user engages with Item A, the system finds other items whose content profile is most similar to Item A's and recommends those.

This is the key difference from the other major family, **collaborative filtering**, which instead relies on patterns across many users' behavior (`users who liked X also liked Y`) and does not need to understand what the item actually is. Content-based systems have three practical advantages that matter for a new product: they work from day one for brand-new items with no interaction history yet, recommendations are explainable, and they don't require a large existing user base.

---

## ✅ 1–2. Similar Projects & Reliable Resources (with links)

| # | Project | Link |
|---|---|---|
| 1 | **Recommenders** — Microsoft/Linux-Foundation-hosted best-practices repository (21.9k★) with production-grade notebooks for content-based approaches (TF-IDF similarity, LightGBM, DKN, NAML) alongside collaborative methods | [github.com/recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) |
| 2 | **Content-Based Movie Recommendation System** — a clean, minimal reference implementation using TF-IDF + cosine similarity on movie metadata | [github.com/sanchitbhasin/Content-Based-Movie-Recommendation-System](https://github.com/sanchitbhasin/Content-Based-Movie-Recommendation-System) |
| 3 | **recommendation-system (tri3amy)** — product recommender built explicitly to demonstrate the linear-algebra core of content-based filtering: TF-IDF, cosine similarity, k-nearest neighbors | [github.com/tri3amy/recommendation-system](https://github.com/tri3amy/recommendation-system) |
| 4 | **TMDB 5000 Movie Recommendation System (hybrid)** — combines content-based filtering with user-preference weighting, a good next step once collaborative signals exist | [github.com/spoluan/TMDB_5000_Movie_recommendation_system](https://github.com/spoluan/TMDB_5000_Movie_recommendation_system) |
| 5 | **How TF-IDF Scoring in Content-Based Recommenders Works** — written walkthrough of the exact math (term frequency, inverse document frequency, cosine similarity) | [medium.com/@shengyuchen](https://medium.com/@shengyuchen/how-tfidf-scoring-in-content-based-recommender-works-5791e36ee8da) |
| 6 | **Content-Based Recommender System** — end-to-end walkthrough (data cleaning → feature combination → vectorization → similarity → ranking) | [medium.com/@bindhubalu](https://medium.com/@bindhubalu/content-based-recommender-system-4db1b3de03e7) |

## ✅ 3. Main Idea of Each Project

- **recommenders-team/recommenders**: not a single demo but a library of vetted reference notebooks; useful as the team's "menu" of algorithms (from simple TF-IDF similarity up to deep knowledge-graph models like DKN) with shared evaluation utilities, so the team can benchmark options before committing to one.
- **sanchitbhasin/Content-Based-Movie-Recommendation-System** and **tri3amy/recommendation-system**: both show the minimal, "from scratch" version of the pipeline — turn item text into TF-IDF vectors, score similarity with cosine distance, return the top-N nearest items. Good as a first prototype template.
- **spoluan/TMDB_5000_Movie_recommendation_system**: shows the natural next step after a pure content-based v1 — blending in user rating signals for a hybrid system — relevant once the product collects user interaction data.
- The two Medium write-ups explain *why* the TF-IDF/cosine-similarity math works — useful to link internally so backend engineers who didn't build the model can still reason about it.

## ✅ 4. The Recommendation Workflow

1. **Collect item content** — gather the text/attributes describing each item (title, description, category, tags, genre list, specifications, etc.) and combine relevant fields into one "content profile" per item.
2. **Clean & normalize** — lowercase text, strip punctuation/stopwords, handle missing fields, standardize categorical values.
3. **Extract features** — decide which fields actually go into the model (see Section 6 below).
4. **Vectorize** — convert each item's content profile into a numeric vector (TF-IDF or embeddings — see Section 7).
5. **Compute similarity & rank** — score how close item vectors are and sort candidates (see Section 7).
6. **Serve via API** — wrap the last steps behind a Backend-callable endpoint (see Sections 8–9).
7. **Evaluate & iterate** — since there's no ground-truth "correct" recommendation initially, rely on qualitative review, then offline metrics (precision@k) once interaction data exists.

```
Item Content  →  Clean/Normalize  →  Feature Extraction  →  Vectorize (TF-IDF / Embeddings)
                                                                       │
                                                                       ▼
                                                       Cosine Similarity + Ranking
                                                                       │
                                                                       ▼
                                                    Recommendation API  →  Backend
```

## ✅ 5. Dataset / Required Data

| Dataset | Domain | Size | Key fields | Link |
|---|---|---|---|---|
| **TMDB Movie Metadata** | Movies | ~5,000 movies | title, overview, genres, keywords, cast, crew | [kaggle.com/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| **MovieLens (25M / Tag Genome)** | Movies | 25M ratings · 62,423 movies · 162,541 users | `movies.csv` (genres); `tags.csv` (user tags); `genome-scores.csv` + `genome-tags.csv` (computed tag-relevance per movie) | [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/) |
| **Book Recommendation Dataset** | Books | ~1.1M ratings · ~270K books · ~278K users | book metadata (title, author, publisher, year) plus user ratings | [kaggle.com/arashnic/book-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) |

**For our own product**, the real requirement is: every item needs at minimum a unique ID, a short text description (or combinable text fields), and category/tag labels. If our current item catalog lacks rich text fields, that's the first gap to close before this system can go beyond a prototype.

## ✅ 6. Features Used by the System

| Feature type | Examples | Role |
|---|---|---|
| Free text | description/overview, review snippets | Main signal for TF-IDF/embeddings — the richest source of similarity |
| Categorical (single or multi-label) | genres, categories, tags | Combined into the text profile, or one-hot encoded and weighted separately |
| Structured metadata | brand, author, release year | Usually used as **filters** (e.g., same category only) rather than similarity inputs |
| Numerical | price, popularity, rating average | Not part of similarity itself — used afterward to re-rank or break ties |

A common practice is building a single combined "content soup" string per item (e.g., `genres + keywords + description`) and vectorizing that one field, which is what most of the reference projects above do.

## ✅ 7. How Similarity / Ranking Is Calculated

- **Vectorization**: TF-IDF is the standard starting point — it weights a word by how distinctive it is to one item versus the whole catalog (`term frequency × inverse document frequency`). A more modern option is **embeddings** (sentence-transformer or similar models), which capture meaning rather than exact word overlap — better quality, but needs a model to run and more compute.
- **Similarity metric**: **cosine similarity** between two item vectors is used almost universally (measures the angle between vectors, so item length/verbosity doesn't skew the score). Score ranges from -1 to 1 (in practice 0 to 1 for TF-IDF/embeddings, since values are non-negative).
- **Ranking**: for a seed item, compute similarity against all other items, sort descending, take the top-N, and exclude the seed item itself and anything the user has already seen.
- **At scale**: computing a full item-by-item similarity matrix works fine up to roughly tens of thousands of items; beyond that, an approximate-nearest-neighbor index (e.g., FAISS, Annoy) is used instead of a full matrix, since it answers "top-N most similar" queries without comparing against every item.

## ✅ 8. How the AI System Can Communicate with the Backend

The recommendation logic should run as an independent service, not be embedded inside the AI training code:

- **Offline step (batch, scheduled)**: a scheduled job (cron, or a simple periodic script) recomputes the vectorization + similarity index whenever the item catalog changes meaningfully (e.g., nightly, or triggered on new-item creation). This keeps request-time latency low, since nothing is computed live.
- **Online step (per request)**: the Backend calls a lightweight **REST API** exposed by the recommendation service, passing an item ID (or user ID) and getting back a ranked list of item IDs + scores. For this scale of project, a synchronous HTTP call is simpler and sufficient — no message queue needed unless request volume becomes very high.
- **Deployment shape**: recommendation service as its own small container/process (e.g., a FastAPI app) that the Backend calls over HTTP, either on the same server (internal network call) or as a separate microservice — whichever matches how the rest of the system is already deployed.

## ✅ 9. Suggested Request & Response Structure

**Request** — `GET /api/v1/recommendations/{item_id}?top_n=10`

| Parameter | Type | Notes |
|---|---|---|
| `item_id` (path) | string/int | The seed item to base recommendations on |
| `top_n` (query, optional) | int | Number of results to return (default e.g. 10) |
| `exclude_ids` (query, optional) | list | Item IDs to skip (e.g., already seen by the user) |

**Response** (JSON):
```json
{
  "item_id": "1234",
  "recommendations": [
    { "item_id": "5678", "title": "Example Item A", "score": 0.87 },
    { "item_id": "9012", "title": "Example Item B", "score": 0.81 }
  ],
  "generated_at": "2026-09-12T10:00:00Z"
}
```

**Error case** — return `404` with `{"error": "item_id not found"}` if the seed item isn't in the index, so the Backend can fail gracefully instead of showing an empty list silently.

## ✅ 10. Required Technologies & Tools

| Purpose | Tool options |
|---|---|
| Data handling | Python, pandas, numpy |
| Vectorization | scikit-learn (`TfidfVectorizer`) for the baseline; `sentence-transformers` for embeddings later |
| Similarity search | scikit-learn cosine_similarity for small catalogs; FAISS or Annoy for large ones |
| Serving the model | FastAPI (lightweight, async, easy Backend integration) or Flask |
| Scheduling recomputation | cron job, or a simple scheduled script; Airflow if the pipeline grows more complex |
| Deployment | Docker container for the recommendation service |
| Optional caching | Redis, to cache frequent lookups and reduce repeated computation |

## ✅ 11. Advantages & Limitations

**Advantages**
- No cold-start problem for new items — a new item is recommendable the moment its content is indexed.
- Explainable: "recommended because it shares these tags/features."
- Doesn't require an existing large user base or interaction history to function.

**Limitations**
- Cold-start still exists **for new users** with no history to seed recommendations from.
- Limited serendipity — tends to recommend items very similar to what's already known, rarely something surprising.
- Quality depends entirely on how rich and clean the item content is; sparse descriptions produce weak recommendations.
- Doesn't capture cross-user "taste" patterns the way collaborative filtering does (e.g., two items with different content that happen to be liked by the same kind of people).
- Similarity search needs a scalable index (FAISS/Annoy) once the catalog grows large — a naive full similarity matrix won't scale indefinitely.

## ✅ 12. Useful Ideas for Our Project

- Start with the **simplest working version** (TF-IDF + cosine similarity) before investing in embeddings — matches the "start simple" approach the team has used elsewhere this sprint, and gives the Backend something to integrate against quickly.
- Precompute the similarity index **offline on a schedule**, never per-request, so the API stays fast regardless of catalog size.
- Design the API response (Section 9) to include a similarity **score**, not just item IDs — the frontend/Backend can use it to decide a relevance cutoff or display a "why recommended" hint later.
- Keep the door open for a **hybrid model**: once real user interaction data exists, the content-based system can be blended with collaborative signals (like the TMDB hybrid example in Section 3) without discarding the work done here.

## ✅ 13. Recommendation for the Best Approach

Given there's no user interaction data yet, the pragmatic path is:

1. Build a **TF-IDF + cosine-similarity MVP** on whatever item text/category data currently exists.
2. Serve it behind a small **FastAPI** service with the endpoint shape in Section 9, precomputing the similarity matrix on a schedule.
3. Ship that to the Backend for integration first — it validates the API contract end-to-end with minimal complexity.
4. Only move to embeddings or a FAISS index **after** confirming the catalog size and content richness justify the added complexity — don't over-build before those numbers are known.

## ✅ 14. Sharing This Document

This document is ready to be attached to the Trello card (as a file or a link) and shared with the AI team for review before implementation begins, per the last checklist item.

---

<p align="center"><sub>BinX Tech · AI &amp; ML Internship — Research prepared for Trello backlog card: <b>AI Research – Content-Based Recommendation System</b></sub></p>
