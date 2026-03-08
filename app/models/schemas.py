"""
app/models/schemas.py
────────────────────────────────────────────────────────────────────────────
Pydantic v2 schemas for all API request/response bodies.

Keeping schemas in a dedicated module:
  • Separates validation logic from routing logic.
  • Allows the same schema to be imported by tests, notebooks, and clients
    without pulling in FastAPI itself.
  • Makes OpenAPI auto-generated docs richer because Pydantic field metadata
    (description, examples) is reflected in the schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# /query endpoint
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Incoming semantic search request.

    Fields
    ------
    query : str
        The user's natural-language question.  Must be non-empty after
        stripping whitespace.
    top_k : int
        Number of semantically similar documents to retrieve from the
        FAISS vector store when the cache misses.  Range 1–20.
    similarity_threshold : float | None
        Override the server-default cache similarity threshold for this
        specific request.  Useful for A/B testing threshold values without
        redeploying the service.  If omitted, the server default is used.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Natural-language search query.",
        examples=["What are the applications of neural networks?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve from the vector store on a cache miss.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-request cosine similarity threshold override (0.0–1.0).",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must contain non-whitespace characters.")
        return v.strip()


class SearchResult(BaseModel):
    """A single document retrieved from the FAISS vector store."""

    doc_id: int = Field(..., description="Internal document identifier.")
    text: str = Field(..., description="Cleaned document body text.")
    target_name: str = Field(..., description="Newsgroup category label.")
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity to the query."
    )
    cluster_id: Optional[int] = Field(
        default=None,
        description="Dominant GMM cluster index assigned at ingestion time.",
    )
    cluster_posteriors: Optional[List[float]] = Field(
        default=None,
        description="Full posterior probability distribution over all clusters.",
    )


class QueryResponse(BaseModel):
    """
    Response from ``POST /query``.

    Fields
    ------
    query           : echoed back for client-side bookkeeping.
    cache_hit       : True if the answer was served from the semantic cache.
    matched_query   : the cached query whose answer was returned (on hit).
    results         : list of retrieved SearchResult objects (on miss).
    latency_ms      : server-side processing time in milliseconds.
    dominant_cluster: GMM cluster the query was routed to.
    """

    query: str
    cache_hit: bool
    matched_query: Optional[str] = Field(
        default=None,
        description="The cached query whose answer matched (on cache hit).",
    )
    results: List[SearchResult] = Field(default_factory=list)
    latency_ms: float = Field(...,
                              description="End-to-end server latency in ms.")
    dominant_cluster: int = Field(
        ..., description="GMM cluster index the query was routed to."
    )


# ─────────────────────────────────────────────────────────────────────────────
# /cache/stats endpoint
# ─────────────────────────────────────────────────────────────────────────────

class CacheStatsResponse(BaseModel):
    """Statistics snapshot for the semantic cache."""

    hits: int = Field(..., description="Total number of cache hits.")
    misses: int = Field(..., description="Total number of cache misses.")
    total_lookups: int = Field(..., description="hits + misses.")
    hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="hits / total_lookups."
    )
    total_size: int = Field(...,
                            description="Current number of cached entries.")
    max_size: int = Field(...,
                          description="Maximum capacity before LRU eviction.")
    similarity_threshold: float = Field(
        ..., description="Active cosine similarity threshold."
    )
    n_clusters: int = Field(
        ..., description="Number of GMM cluster partitions."
    )
    bucket_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-cluster entry counts (cluster_id → count).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# /cache DELETE endpoint
# ─────────────────────────────────────────────────────────────────────────────

class CacheClearResponse(BaseModel):
    """Confirmation payload returned after a cache flush."""

    message: str = Field(
        default="Cache cleared successfully.",
        description="Human-readable confirmation string.",
    )
    entries_removed: int = Field(
        ..., description="Number of entries that were present before clearing."
    )


# ─────────────────────────────────────────────────────────────────────────────
# /health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Basic liveness/readiness probe response."""

    status: str = Field(default="ok")
    vector_store_size: int = Field(
        ..., description="Number of documents indexed in FAISS."
    )
    n_clusters: int = Field(
        ..., description="Number of active GMM clusters."
    )
    cache_size: int = Field(
        ..., description="Number of entries currently in the semantic cache."
    )
