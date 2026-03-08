# fuzzy-cluster-cache

A lightweight semantic search system built from first principles using the 20 Newsgroups dataset, sentence-transformer embeddings, Gaussian Mixture Model fuzzy clustering, and a custom cluster-partitioned semantic cache — all served via a FastAPI backend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                   │
│   POST /query     GET /cache/stats     DELETE /cache         │
│   GET  /health                                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │      SemanticCache         │
              │  cluster-partitioned       │
              │  O(N/K) cosine lookup      │
              └─────────────┬──────────────┘
              ┌─────────────▼──────────────┐
              │     FuzzyClusterer (GMM)   │
              │  BIC-selected K, diag cov  │
              │  predict_proba → soft P(k) │
              └─────────────┬──────────────┘
              ┌─────────────▼──────────────┐
              │   EmbeddingModel           │
              │   all-MiniLM-L6-v2 (384d) │
              │   L2-normalised vecs       │
              └─────────────┬──────────────┘
              ┌─────────────▼──────────────┐
              │   FAISSVectorStore         │
              │   IndexFlatIP (cosine)     │
              │   ~18k 20-Newsgroups docs  │
              └────────────────────────────┘
```

---

## Project Structure

```
fuzzy-cluster-cache/
├── app/
│   ├── main.py               # FastAPI app with lifespan state management
│   ├── models/
│   │   └── schemas.py        # Pydantic v2 request / response models
│   ├── core/
│   │   ├── embeddings.py     # EmbeddingModel + FAISSVectorStore
│   │   ├── clustering.py     # FuzzyClusterer (GMM + PCA + BIC sweep)
│   │   └── cache.py          # SemanticCache (cluster-partitioned, O(N/K))
│   └── pipeline/
│       └── data.py           # NewsgroupsPipeline (fetch, clean, filter)
├── notebooks/
│   └── analysis.ipynb        # End-to-end analysis with BIC curves &
│                             # boundary document visualisation
├── Dockerfile                # Multi-stage build (python:3.10-slim)
├── docker-compose.yml        # Service definition with health-check
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### 1. Data Preprocessing — `remove=('headers', 'footers', 'quotes')`

Without this, sentence-transformer embeddings cluster by email address rather
than topic. This is the single highest-leverage data-hygiene step.

### 2. Fuzzy Clustering — GMM instead of K-Means

K-Means performs _hard_ assignment. GMM computes `P(cluster_k | document_i)`
via Bayes' theorem, giving every document a probability distribution over clusters.
A post about gun legislation can be 70% "politics" and 30% "firearms" simultaneously.

### 3. BIC-Selected Cluster Count

We sweep K = 5…30, compute BIC for each, and choose K at the BIC minimum.
BIC penalises free parameters as `K × log(N)`, preventing over-segmentation of
the cache into uselessly small partitions.

### 4. Semantic Cache — O(N/K) Lookup

A naïve cache scans all N entries per query — O(N). By routing each query to its
dominant GMM cluster partition, we reduce the expected scan to O(N/K). With K=20
and N=1000 cached entries: ~50 comparisons instead of 1000.

### 5. Similarity Threshold — 0.85 Default

| Threshold | Behaviour                                                  |
| --------- | ---------------------------------------------------------- |
| 0.98      | Near-verbatim only → very low hit rate                     |
| **0.85**  | Captures paraphrases, rejects topically different queries  |
| 0.60      | "Apple fruit" and "Apple computers" would collide → bad UX |

---

## Quick Start

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API (startup takes ~2–5 min for embedding + GMM fit)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info

# 3. Query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks in robotics", "top_k": 5}' | python -m json.tool

# 4. Warm cache hit (same query again — should be faster)
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks in robotics", "top_k": 5}' | python -m json.tool

# 5. Statistics
curl -s http://localhost:8000/cache/stats | python -m json.tool

# 6. Clear cache
curl -s -X DELETE http://localhost:8000/cache | python -m json.tool
```

### Docker

```bash
# Build & start
docker compose up --build

# Override defaults
SIMILARITY_THRESHOLD=0.90 GMM_K_MAX=25 docker compose up --build
```

---

## API Endpoints

| Method   | Path           | Description                             |
| -------- | -------------- | --------------------------------------- |
| `POST`   | `/query`       | Semantic search (cache-first)           |
| `GET`    | `/cache/stats` | Hit rate, bucket distribution, counters |
| `DELETE` | `/cache`       | Flush cache + reset counters            |
| `GET`    | `/health`      | Liveness probe                          |

Interactive docs: `http://localhost:8000/docs`

---

## Environment Variables

| Variable               | Default                                  | Description                              |
| ---------------------- | ---------------------------------------- | ---------------------------------------- |
| `SIMILARITY_THRESHOLD` | `0.85`                                   | Cosine similarity cut-off for cache hits |
| `CACHE_MAX_SIZE`       | `1024`                                   | Max entries before LRU eviction          |
| `EMBEDDING_MODEL`      | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID                     |
| `GMM_K_MIN`            | `5`                                      | Lower bound of BIC sweep                 |
| `GMM_K_MAX`            | `30`                                     | Upper bound of BIC sweep                 |
| `NEWSGROUPS_SUBSET`    | `all`                                    | `"train"`, `"test"`, or `"all"`          |

---

## Tech Stack

| Component        | Library               | Version      |
| ---------------- | --------------------- | ------------ |
| Web framework    | FastAPI + Uvicorn     | 0.111 / 0.29 |
| Schemas          | Pydantic v2           | 2.7          |
| Embeddings       | sentence-transformers | 3.0          |
| Vector store     | FAISS (CPU)           | 1.8          |
| Fuzzy clustering | scikit-learn GMM      | 1.5          |
| Data & math      | NumPy                 | 1.26         |

No Redis, Memcached, or external caching middleware.
