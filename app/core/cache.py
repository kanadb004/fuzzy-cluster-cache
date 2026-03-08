"""
app/core/cache.py
────────────────────────────────────────────────────────────────────────────
Phase 3 – Semantic Cache Layer
───────────────────────────────
Design rationale – optimising lookup time complexity
────────────────────────────────────────────────────
A naïve semantic cache compares an incoming query embedding against *every*
cached entry by computing cosine similarity one-by-one.  If N entries are
cached, this is O(N) per request and degrades linearly as usage grows.

We eliminate most of this work by exploiting the GMM cluster structure:

  Step 1.  Embed the incoming query             → shape (384,)
  Step 2.  Project through GMM                  → P(cluster_k | query)
  Step 3.  Select dominant cluster              → integer k*
  Step 4.  Scan only the bucket for cluster k*  → ≈ N/K entries

This takes lookup from O(N) to O(N/K) without sacrificing correctness,
because semantically similar queries will overwhelmingly share the same
dominant cluster.  With K=20 clusters and N=1000 cached entries, the
average bucket scan is ~50 comparisons instead of 1 000.

The trade-off is a small false-negative rate: a query at a cluster boundary
might be routed to a different bucket than its nearest neighbour.  We
mitigate this with the `multi_cluster_search` option (top-2 GMM buckets).

Similarity threshold tuning
────────────────────────────
`similarity_threshold` is the cosine similarity cut-off for a cache hit.

  • HIGH threshold (e.g. 0.98): only near-verbatim paraphrases hit.
    → Low hit rate, mostly misses.  Redundant computation.

  • LOW threshold (e.g. 0.60): semantically related but topically distinct
    queries collide.  "What fruit is an apple?" and "Tell me about
    Apple computers" both cluster under 'technology' yet deserve different
    answers.  Returning a stale answer here produces a bad user experience.

  • SWEET SPOT (0.82 – 0.92): captures genuine paraphrases and near-synonym
    reformulations while rejecting tangentially related queries.  0.85 is
    the default based on empirical evals on the MSMARCO paraphrase corpus.

Cache data structure
────────────────────
```
_buckets: Dict[int, List[CacheEntry]]
```

  • Keyed by dominant GMM cluster index k (integer, 0..K-1).
  • Each bucket is an unsorted Python list of CacheEntry objects.
  • Lookup within a bucket is a linear scan with early-exit on first hit.
  • This avoids external dependencies (no Redis, no Memcached) while staying
    well within O(N/K) per request for typical cache sizes.

For cache sizes > 10 000 entries per bucket, consider upgrading the bucket
to FAISS IndexFlatIP for sub-linear search, but this is out of scope here.

LRU-style eviction
───────────────────
When `max_size` entries are reached, the entry with the oldest
`last_accessed` timestamp is evicted from its bucket.  This keeps memory
bounded and favours recently used entries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default similarity threshold.  See module docstring for trade-off analysis.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.85

# Maximum number of cache entries across all buckets.  When exceeded, the
# globally least-recently-used entry is evicted.
DEFAULT_MAX_SIZE: int = 1024

# Whether to also search the 2nd-most-probable cluster to catch near-boundary
# queries.  Slightly increases lookup cost but reduces false-negative rate.
DEFAULT_MULTI_CLUSTER: bool = True


@dataclass
class CacheEntry:
    """
    A single cached (query, answer) pair.

    Fields
    ------
    query_text   : original query string
    query_vec    : L2-normalised embedding (384-d float32)
    answer       : the serialisable answer object stored for this query
    cluster_id   : the dominant GMM cluster at insertion time
    created_at   : Unix timestamp of first insertion
    last_accessed: Unix timestamp of most recent cache hit (updated on hit)
    hit_count    : number of times this entry has been served from cache
    """

    query_text: str
    query_vec: np.ndarray
    answer: Any
    cluster_id: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        """Update access metadata on a cache hit."""
        self.last_accessed = time.time()
        self.hit_count += 1


class SemanticCache:
    """
    Cluster-partitioned semantic query cache with configurable cosine
    similarity threshold.

    Parameters
    ----------
    clusterer : FuzzyClusterer
        A *fitted* clusterer instance.  Used to route queries to buckets.
    embedding_fn : callable
        A function ``(text: str) -> np.ndarray`` that returns an L2-normalised
        384-d embedding.  Typically ``EmbeddingModel.encode_single``.
    similarity_threshold : float
        Cosine similarity cut-off for a cache hit (see module docstring).
    max_size : int
        Maximum total entries before LRU eviction.
    multi_cluster : bool
        If True, search top-2 GMM clusters for boundary robustness.
    """

    def __init__(
        self,
        # FuzzyClusterer (typed as Any to avoid
        clusterer: Any,
        embedding_fn: Any,               # circular import at module load time)
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_size: int = DEFAULT_MAX_SIZE,
        multi_cluster: bool = DEFAULT_MULTI_CLUSTER,
    ) -> None:
        self._clusterer = clusterer
        self._embed = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.multi_cluster = multi_cluster

        # ── core data structure ───────────────────────────────────────────
        # _buckets[k] = list of CacheEntry objects whose dominant cluster is k.
        # We use a defaultdict-style approach but prefer explicit assignment to
        # make the structure visible.
        self._buckets: Dict[int, List[CacheEntry]] = {}

        # ── statistics ────────────────────────────────────────────────────
        self._hits: int = 0
        self._misses: int = 0
        self._total_size: int = 0

        # ── thread safety ─────────────────────────────────────────────────
        # A single global lock is simple and safe for the expected request
        # concurrency of a single-node FastAPI service (asyncio event loop
        # with sync cache operations).  For truly concurrent workloads,
        # consider per-bucket locks.
        self._lock: Lock = Lock()

    # ── public interface ──────────────────────────────────────────────────

    def lookup(self, query: str) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Attempt to answer ``query`` from the cache.

        Algorithm
        ---------
        1. Embed query                    → vec (384-d)
        2. GMM predict_proba              → posteriors (K-d)
        3. Select top cluster(s)          → candidate bucket indices
        4. For each bucket: compute cosine similarity with stored vecs
        5. If max(similarity) ≥ threshold → cache HIT, return stored answer
        6. Otherwise                      → cache MISS

        Returns
        -------
        (hit: bool, answer: Any | None, matched_query: str | None)
        """
        vec = self._embed(query)
        posteriors = self._clusterer.predict_proba(vec)

        # Determine buckets to search.
        clusters_to_search = self._top_clusters(posteriors)

        with self._lock:
            best_sim = -1.0
            best_entry: Optional[CacheEntry] = None

            for cluster_id in clusters_to_search:
                bucket = self._buckets.get(cluster_id, [])
                for entry in bucket:
                    # Cosine similarity: since both vecs are L2-normalised,
                    # this reduces to the inner product (dot product).
                    # Complexity per comparison: O(384) ≈ constant.
                    sim = float(np.dot(vec, entry.query_vec))
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_sim >= self.similarity_threshold and best_entry is not None:
                # ── CACHE HIT ────────────────────────────────────────────
                best_entry.touch()
                self._hits += 1
                logger.debug(
                    "Cache HIT (sim=%.4f, threshold=%.4f, cluster=%d).",
                    best_sim,
                    self.similarity_threshold,
                    best_entry.cluster_id,
                )
                return True, best_entry.answer, best_entry.query_text

            # ── CACHE MISS ───────────────────────────────────────────────
            self._misses += 1
            logger.debug(
                "Cache MISS (best_sim=%.4f, threshold=%.4f).",
                best_sim,
                self.similarity_threshold,
            )
            return False, None, None

    def store(self, query: str, answer: Any, query_vec: Optional[np.ndarray] = None) -> None:
        """
        Store a new (query, answer) pair in the appropriate cluster bucket.

        Parameters
        ----------
        query : str
            The original query text.
        answer : Any
            The answer/result to cache.  Must be JSON-serialisable if the
            cache is served via the API.
        query_vec : np.ndarray | None
            Pre-computed embedding.  If None, the query is re-embedded
            (wastes ~20 ms; pass the vec computed during lookup to avoid this).
        """
        if query_vec is None:
            query_vec = self._embed(query)

        posteriors = self._clusterer.predict_proba(query_vec)
        cluster_id = int(posteriors.argmax())

        entry = CacheEntry(
            query_text=query,
            query_vec=query_vec.astype(np.float32),
            answer=answer,
            cluster_id=cluster_id,
        )

        with self._lock:
            # ── eviction if at capacity ───────────────────────────────────
            if self._total_size >= self.max_size:
                self._evict_lru()

            if cluster_id not in self._buckets:
                self._buckets[cluster_id] = []
            self._buckets[cluster_id].append(entry)
            self._total_size += 1

        logger.debug(
            "Stored entry in cluster=%d (bucket_size=%d, total=%d).",
            cluster_id,
            len(self._buckets[cluster_id]),
            self._total_size,
        )

    def clear(self) -> None:
        """
        Flush all cache entries and reset statistics counters.
        Called by the DELETE /cache endpoint.
        """
        with self._lock:
            self._buckets.clear()
            self._hits = 0
            self._misses = 0
            self._total_size = 0
        logger.info("Semantic cache cleared.")

    # ── statistics ────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        """
        Return a statistics snapshot.

        hit_rate    : fraction of lookups that returned a cached answer.
        total_size  : current number of entries across all buckets.
        bucket_dist : per-cluster entry counts (shows how load is distributed).
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            bucket_dist = {k: len(v) for k, v in self._buckets.items()}
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_lookups": total,
                "hit_rate": round(hit_rate, 4),
                "total_size": self._total_size,
                "max_size": self.max_size,
                "similarity_threshold": self.similarity_threshold,
                "n_clusters": self._clusterer.n_components,
                "bucket_distribution": bucket_dist,
            }

    # ── private helpers ────────────────────────────────────────────────────

    def _top_clusters(self, posteriors: np.ndarray) -> List[int]:
        """
        Return the indices of the top-N clusters by posterior probability.

        When ``multi_cluster=True`` we search the top-2 clusters.  This
        catches queries that sit near a Gaussian boundary (e.g., posteriors
        [0.52, 0.43, 0.05, …]) and whose nearest cached neighbour lives in
        the second-ranked bucket.  The cost is at most doubling the bucket
        scan, which remains O(N/K).
        """
        n = 2 if self.multi_cluster else 1
        return posteriors.argsort()[::-1][:n].tolist()

    def _evict_lru(self) -> None:
        """
        Remove the globally least-recently-used cache entry.

        Scans all buckets to find the entry with the oldest ``last_accessed``
        timestamp.  O(N) in the number of cached entries, but called at most
        once per new insertion, amortising the cost.
        """
        oldest_time = float("inf")
        oldest_bucket: Optional[int] = None
        oldest_idx: Optional[int] = None

        for bucket_id, bucket in self._buckets.items():
            for i, entry in enumerate(bucket):
                if entry.last_accessed < oldest_time:
                    oldest_time = entry.last_accessed
                    oldest_bucket = bucket_id
                    oldest_idx = i

        if oldest_bucket is not None and oldest_idx is not None:
            evicted = self._buckets[oldest_bucket].pop(oldest_idx)
            self._total_size -= 1
            # Clean up empty buckets to keep the dictionary tidy.
            if not self._buckets[oldest_bucket]:
                del self._buckets[oldest_bucket]
            logger.debug(
                "Evicted LRU entry '%s…' from cluster=%d.",
                evicted.query_text[:40],
                evicted.cluster_id,
            )
