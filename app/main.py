"""
app/main.py
────────────────────────────────────────────────────────────────────────────
Phase 4 – FastAPI Application & State Management
──────────────────────────────────────────────────
Startup / Shutdown: lifespan context manager
────────────────────────────────────────────
FastAPI ≥ 0.93 deprecated `@app.on_event("startup")` in favour of a
single ``lifespan`` async context manager.  We use this modern pattern to:

  1. Build the full ML pipeline (data → embeddings → FAISS → GMM → cache)
     exactly once when the ASGI server starts.
  2. Attach all stateful objects to ``app.state`` so every request handler
     can access them without module-level globals (which break under reload).
  3. Cleanly log shutdown without special teardown logic (our in-memory
     structures need no explicit cleanup).

State management
────────────────
All stateful objects live on ``app.state``:

  app.state.vector_store  : FAISSVectorStore  – 384-d indexed corpus
  app.state.embedder      : EmbeddingModel    – sentence-transformers wrapper
  app.state.clusterer     : FuzzyClusterer    – fitted GMM + PCA
  app.state.cache         : SemanticCache     – cluster-partitioned cache
  app.state.documents     : list[NewsDocument] – metadata for FAISS results

This avoids global singleton anti-patterns and makes unit testing
straightforward: construct a test client with a custom app.state for
dependency injection without patching global state.

Endpoints
─────────
  POST  /query        – semantic search with cache-first lookup
  GET   /cache/stats  – statistics snapshot
  DELETE /cache       – flush the cache (resets counters too)
  GET   /health       – liveness probe
"""

from __future__ import annotations
import os

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.cache import SemanticCache
from app.core.clustering import FuzzyClusterer
from app.core.embeddings import EmbeddingModel, FAISSVectorStore
from app.models.schemas import (
    CacheClearResponse,
    CacheStatsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from app.pipeline.data import NewsgroupsPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── environment / configuration ───────────────────────────────────────────────

SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1024"))
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GMM_K_MIN: int = int(os.getenv("GMM_K_MIN", "5"))
GMM_K_MAX: int = int(os.getenv("GMM_K_MAX", "30"))
NEWSGROUPS_SUBSET: str = os.getenv("NEWSGROUPS_SUBSET", "all")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan – builds the entire ML stack once at startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    ASGI lifespan context.  Everything before ``yield`` runs at startup;
    everything after runs at shutdown (currently nothing to clean up).

    We intentionally do NOT make the heavy ML work async (no ``await``)
    because scikit-learn and sentence-transformers are CPU-bound synchronous
    code that does not benefit from async I/O.  Running them synchronously in
    the lifespan prevents event-loop starvation that would occur if we
    incorrectly wrapped them in ``asyncio.to_thread`` here.
    """
    # ── Step 1 · Ingest & preprocess corpus ──────────────────────────────────
    logger.info("=== STARTUP: Loading 20 Newsgroups corpus ===")
    pipeline = NewsgroupsPipeline(subset=NEWSGROUPS_SUBSET)
    pipeline.load()
    documents = pipeline.documents
    logger.info("Corpus ready: %d documents.", len(documents))

    # ── Step 2 · Build embeddings ─────────────────────────────────────────────
    logger.info("=== STARTUP: Building sentence embeddings ===")
    embedder = EmbeddingModel(
        model_name=EMBEDDING_MODEL, show_progress_bar=True)
    vector_store = FAISSVectorStore()

    texts = [doc.text for doc in documents]
    embeddings = embedder.encode(texts, normalize=True)

    # Attach per-document metadata so FAISS search results are enriched.
    metadata = [
        {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "target": doc.target,
            "target_name": doc.target_name,
        }
        for doc in documents
    ]
    vector_store.add(embeddings, metadata)
    logger.info("FAISS index built: %d vectors.", vector_store.size)

    # ── Step 3 · Fit fuzzy GMM clustering ────────────────────────────────────
    logger.info(
        "=== STARTUP: Fitting fuzzy GMM (BIC sweep K=%d..%d) ===", GMM_K_MIN, GMM_K_MAX)
    clusterer = FuzzyClusterer(k_min=GMM_K_MIN, k_max=GMM_K_MAX)
    result = clusterer.fit(embeddings)

    # Annotate metadata with cluster assignments for enriched API responses.
    for i, meta in enumerate(metadata):
        meta["cluster_id"] = int(result.dominant_clusters[i])
        meta["cluster_posteriors"] = result.posteriors[i].tolist()

    logger.info("GMM fitted: K=%d components chosen by BIC.",
                result.n_components)
    logger.info(
        "Boundary documents (max_posterior < 0.4): %d",
        len(result.boundary_indices),
    )

    # ── Step 4 · Initialise semantic cache ───────────────────────────────────
    logger.info("=== STARTUP: Initialising semantic cache ===")
    cache = SemanticCache(
        clusterer=clusterer,
        embedding_fn=embedder.encode_single,
        similarity_threshold=SIMILARITY_THRESHOLD,
        max_size=CACHE_MAX_SIZE,
        multi_cluster=True,
    )

    # ── Attach to app.state ───────────────────────────────────────────────────
    app.state.documents = documents
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.clusterer = clusterer
    app.state.cache = cache
    app.state.metadata = metadata   # enriched with cluster assignments

    logger.info("=== STARTUP COMPLETE ===")

    yield  # ── server runs here ──────────────────────────────────────────────

    logger.info("=== SHUTDOWN: releasing resources ===")
    # In-memory structures are GC'd automatically; nothing explicit needed.


# ─────────────────────────────────────────────────────────────────────────────
# App construction
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fuzzy Cluster Semantic Search",
    description=(
        "Semantic search over the 20 Newsgroups corpus with GMM-partitioned "
        "in-memory cache.  Embeddings: all-MiniLM-L6-v2.  "
        "Clustering: Gaussian Mixture Models (BIC-selected K).  "
        "Cache: cluster-partitioned cosine similarity with configurable threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper to access app.state from a request
# ─────────────────────────────────────────────────────────────────────────────
def _state(request: Request):
    """Convenience accessor; avoids repeating ``request.app.state`` everywhere."""
    return request.app.state


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search with cache-first lookup",
    tags=["Search"],
)
async def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    """
    Execute a semantic search query.

    **Cache-first algorithm:**

    1. Embed the query with all-MiniLM-L6-v2.
    2. Route to GMM cluster partition(s).
    3. Scan cluster bucket for a cached answer above the similarity threshold.
    4. On HIT – return cached results immediately (no FAISS scan).
    5. On MISS – scan FAISS vector store for top-k similar documents,
       store the result in the cache, and return.

    The optional ``similarity_threshold`` field overrides the server default
    for this single request, enabling per-request A/B testing.
    """
    state = _state(request)
    t0 = time.perf_counter()

    # Temporarily override the threshold if the caller requested it.
    original_threshold = state.cache.similarity_threshold
    if payload.similarity_threshold is not None:
        state.cache.similarity_threshold = payload.similarity_threshold

    try:
        # ── 1. Embed query ────────────────────────────────────────────────
        query_vec: np.ndarray = state.embedder.encode_single(payload.query)

        # ── 2. Determine dominant cluster (for response metadata) ─────────
        posteriors = state.clusterer.predict_proba(query_vec)
        dominant_cluster = int(posteriors.argmax())

        # ── 3. Cache lookup ───────────────────────────────────────────────
        hit, cached_answer, matched_query = state.cache.lookup(payload.query)

        if hit:
            latency_ms = (time.perf_counter() - t0) * 1000
            return QueryResponse(
                query=payload.query,
                cache_hit=True,
                matched_query=matched_query,
                results=cached_answer,   # type: ignore[arg-type]
                latency_ms=round(latency_ms, 2),
                dominant_cluster=dominant_cluster,
            )

        # ── 4. FAISS vector store search (cache miss) ─────────────────────
        raw_results = state.vector_store.search(query_vec, k=payload.top_k)

        search_results = [
            SearchResult(
                doc_id=meta["doc_id"],
                text=meta["text"][:500],   # truncate for API response payload
                target_name=meta["target_name"],
                similarity=round(sim, 4),
                cluster_id=meta.get("cluster_id"),
                cluster_posteriors=meta.get("cluster_posteriors"),
            )
            for sim, meta in raw_results
        ]

        # ── 5. Store in cache ─────────────────────────────────────────────
        # We store the serialised list[SearchResult] so a future hit returns
        # the same structure that a miss would return.
        state.cache.store(
            query=payload.query,
            answer=search_results,
            query_vec=query_vec,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryResponse(
            query=payload.query,
            cache_hit=False,
            matched_query=None,
            results=search_results,
            latency_ms=round(latency_ms, 2),
            dominant_cluster=dominant_cluster,
        )

    finally:
        # Always restore the original threshold regardless of success/failure.
        state.cache.similarity_threshold = original_threshold


@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics snapshot",
    tags=["Cache"],
)
async def cache_stats(request: Request) -> CacheStatsResponse:
    """
    Return a real-time snapshot of the semantic cache statistics.

    Useful for monitoring hit/miss ratios, inspecting cluster load
    distribution, and tuning the similarity threshold.
    """
    stats = _state(request).cache.stats
    # Convert integer cluster keys to strings for JSON serialisation
    # (JSON object keys must be strings).
    stats["bucket_distribution"] = {
        str(k): v for k, v in stats["bucket_distribution"].items()
    }
    return CacheStatsResponse(**stats)


@app.delete(
    "/cache",
    response_model=CacheClearResponse,
    summary="Flush the semantic cache",
    tags=["Cache"],
)
async def clear_cache(request: Request) -> CacheClearResponse:
    """
    Flush all cached entries and reset hit/miss counters.

    This endpoint is idempotent: calling it on an already-empty cache
    returns ``entries_removed: 0`` without error.
    """
    cache = _state(request).cache
    entries_before = cache.stats["total_size"]
    cache.clear()
    return CacheClearResponse(
        message="Cache cleared successfully.",
        entries_removed=entries_before,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
    tags=["Health"],
)
async def health(request: Request) -> HealthResponse:
    """
    Confirms the service is alive and the ML stack is fully initialised.

    Returns corpus size, number of cluster partitions, and current cache
    occupancy so orchestration tools can validate readiness.
    """
    state = _state(request)
    return HealthResponse(
        status="ok",
        vector_store_size=state.vector_store.size,
        n_clusters=state.clusterer.n_components,
        cache_size=state.cache.stats["total_size"],
    )
