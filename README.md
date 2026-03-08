# Lightweight Semantic Search & Partitioned Cache

A custom-built, lightweight semantic search API built over the 20 Newsgroups dataset. This system features soft document clustering, an `O(N/K)` partitioned semantic cache built from scratch (no Redis/Memcached), and a FastAPI backend optimized for single-worker, stateful memory management.

Author: Kanad Bhattacharya

---

# Core Architecture and Justifications

## 1. Data Hygiene and Vector Space

The 20 Newsgroups dataset is inherently noisy.

### Data Leakage Prevention

Raw headers, footers, and quoted replies were strictly stripped during ingestion. Without this preprocessing step, the embedding model tends to group documents based on sender email domains rather than semantic meaning.

### Sparsity Filtering

Documents with fewer than 50 tokens were removed, leaving approximately 12.8k cleaned documents. This prevents polluting the embedding space with extremely low-information samples.

### Embeddings and Vector Database

The corpus is encoded using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Properties:

- 384 dimensional embeddings  
- L2-normalised vectors  
- Stored in an in-memory FAISS `IndexFlatIP` index  
- Enables very fast cosine similarity search

---

## 2. Fuzzy Clustering (GMM + PCA)

Hard clustering approaches such as K-Means force documents into rigid categories. This is unsuitable for real-world text where documents often belong to multiple semantic domains (for example military encryption overlaps with both politics and cryptography).

### Dimensionality Reduction

To mitigate the curse of dimensionality and stabilize covariance matrices:

- 384 dimensional embeddings are reduced to 64 dimensions using PCA
- This retains more than 56 percent of the variance while improving numerical stability for the Gaussian Mixture Model.

### Gaussian Mixture Models

GMM produces soft probabilistic cluster assignments:

```
P(cluster_k | document)
```

This allows documents to partially belong to multiple clusters.

### Mathematical Justification for K

A Bayesian Information Criterion (BIC) sweep was performed.

Observations:

- BIC continues to decrease slightly past K = 30
- However higher values excessively fragment cache partitions

Therefore:

```
K = 30
```

was selected as the best engineering trade-off between semantic separation and cache efficiency.

The full BIC analysis can be found in:

```
notebooks/analysis.ipynb
```

---

## 3. The O(N/K) Semantic Cache

A naive semantic cache compares incoming queries against every cached query:

```
O(N)
```

As the cache grows, lookup eventually becomes slower than a direct FAISS search.

### Cluster Partitioning

The cache is implemented as a cluster-partitioned dictionary.

Workflow:

1. Incoming query -> embedding  
2. Embedding -> GMM cluster prediction  
3. Query routed to the dominant cluster bucket  
4. Similarity comparison only within that bucket  

This reduces lookup complexity to roughly:

```
O(N / K)
```

### Tunable Similarity Threshold

The cache uses a configurable cosine similarity threshold.

Threshold behavior:

0.98 -> almost exact match cache (very low hit rate)  
0.60 -> semantic collisions and false positives  
0.85 -> captures paraphrases while maintaining semantic boundaries

---

# Local Setup and Installation

## Prerequisites

Python 3.10 or higher.  
Using a Conda environment is recommended.

## Installation

```bash
# Clone repository
git clone https://github.com/kanadb004/fuzzy-cluster-cache.git

cd fuzzy-cluster-cache

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Startup Behavior

The FastAPI lifespan manager performs the following steps on startup:

1. Dataset download  
2. Document preprocessing  
3. Embedding generation  
4. PCA fitting  
5. GMM training  
6. FAISS index construction  

Startup time is typically:

```
3 - 8 minutes
```

The server opens on port 8000 once initialization completes.

---

# Docker Deployment

The system is containerized using a multi-stage Dockerfile to keep the runtime image lightweight.

Important design choice:

```
Single Uvicorn worker
```

Multiple workers would fragment the in-memory cache across processes and break cache consistency.

### Build and Run

```bash
docker compose up --build
```

The `docker-compose.yml` includes an extended health-check start period to accommodate the heavy GMM training during boot.

---

# API Endpoints and Usage

## POST /query

Performs semantic search.

If a semantically similar query exists in the cache, the request bypasses FAISS and returns immediately.

### Request

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "deep learning applications in computer vision",
           "top_k": 3,
           "similarity_threshold": 0.85
         }'
```

### Example Response

```json
{
  "query": "deep learning applications in computer vision",
  "cache_hit": false,
  "matched_query": null,
  "results": [
    {
      "doc_id": 16126,
      "text": "call for papers =============== progress in neural networks...",
      "target_name": "comp.graphics",
      "similarity": 0.4066,
      "cluster_id": 4
    }
  ],
  "latency_ms": 1450.12,
  "dominant_cluster": 4
}
```

---

## GET /cache/stats

Returns cache telemetry including hit rate and query distribution across the 30 cluster buckets.

### Request

```bash
curl http://localhost:8000/cache/stats
```

---

## DELETE /cache

Clears the in-memory cache and resets performance counters.

### Request

```bash
curl -X DELETE http://localhost:8000/cache
```

---

# Primary Project Structure

```
fuzzy-cluster-cache/
├── app/
│   ├── main.py                Phase 4: FastAPI lifespan + 4 endpoints
│   ├── models/schemas.py      Phase 4: Pydantic v2 request/response schemas
│   ├── core/
│   │   ├── embeddings.py      Phase 1: EmbeddingModel + FAISSVectorStore
│   │   ├── clustering.py      Phase 2: FuzzyClusterer (GMM + PCA + BIC sweep)
│   │   └── cache.py           Phase 3: SemanticCache (cluster-partitioned)
│   └── pipeline/data.py       Phase 1: NewsgroupsPipeline (fetch + clean)
├── notebooks/analysis.ipynb   11-section end-to-end analysis
├── Dockerfile                 Phase 5: multi-stage build
├── docker-compose.yml         Phase 5: service + health-check
├── .dockerignore
└── requirements.txt
```

---

# Notes

This system intentionally avoids Redis or Memcached to demonstrate a first-principles semantic caching architecture.

The project combines:

- vector search
- probabilistic clustering
- cache complexity optimization
- production API deployment