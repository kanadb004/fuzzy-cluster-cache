"""
app/core/embeddings.py
────────────────────────────────────────────────────────────────────────────
Phase 1 – Embedding Model & FAISS Vector Store
──────────────────────────────────────────────
Model choice: `sentence-transformers/all-MiniLM-L6-v2`
  • 384-dimensional output vectors  – small enough to keep FAISS memory
    footprint low (~1.5 MB per 1 000 documents) while preserving strong
    topical separability on English text.
  • Distilled from a larger MPNet teacher – runs on CPU in ~20 ms per
    sentence on modern hardware, satisfying the "lightweight" constraint.
  • Apache-2.0 licensed, no API key required.

FAISS choice: IndexFlatIP (inner-product / cosine after L2 normalisation)
  • IndexFlatIP performs exact nearest-neighbour search.  For a corpus of
    ≤20 000 documents the brute-force scan is sub-millisecond and avoids
    the approximation errors of IVF/HNSW indices that only pay off above
    ~100 k vectors.
  • After L2-normalising all vectors, inner product equals cosine similarity,
    so threshold comparisons in the cache layer are numerically identical
    to cosine similarity comparisons.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model name – centralised here so a single change propagates
# to both the API layer and any analysis notebooks.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding dimensionality for all-MiniLM-L6-v2.  Declared explicitly so
# the FAISS index can be pre-allocated before any documents are ingested
# and validated at runtime if necessary.
EMBEDDING_DIM: int = 384


class EmbeddingModel:
    """
    Thin wrapper around a ``SentenceTransformer`` that encapsulates batch
    encoding logic and ensures all vectors are L2-normalised before leaving
    this layer.

    L2 normalisation is performed here – not lazily at query time – so that
    (a) the FAISS index always stores unit vectors and (b) cosine similarity
    in the cache is simply ``np.dot(q, k)`` without a division.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    batch_size : int
        Number of sentences encoded per forward pass.  Higher values saturate
        CPU matrix-multiply throughput; lower values reduce peak RAM.
        128 is a safe default for machines with ≥4 GB RAM.
    show_progress_bar : bool
        Display tqdm progress during bulk encoding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 128,
        show_progress_bar: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        logger.info("Loading embedding model: %s …", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded (dim=%d).", EMBEDDING_DIM)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a list of strings to float32 embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Input sentences / documents.
        normalize : bool
            L2-normalise the output vectors.  Should always be ``True``
            when the result is used with ``IndexFlatIP`` or the cache's
            cosine similarity logic.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), EMBEDDING_DIM)``, dtype ``float32``.
        """
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convenience method for encoding a single query string at inference
        time.  Returns shape ``(EMBEDDING_DIM,)`` rather than ``(1, EMBEDDING_DIM)``
        so the caller doesn't need to squeeze.
        """
        vec = self.encode([text], normalize=True)
        return vec[0]


class FAISSVectorStore:
    """
    An in-memory FAISS vector store backed by ``IndexFlatIP``.

    Indexing strategy
    -----------------
    ``IndexFlatIP`` performs exhaustive inner-product search.  Because all
    stored vectors are L2-normalised by :class:`EmbeddingModel`, inner
    product equals cosine similarity, so ``search()`` effectively returns
    the *k* most semantically similar documents to a query.

    The store keeps a parallel list of metadata dictionaries (``_meta``) in
    Python, keyed by their integer FAISS position, to allow O(1) retrieval of
    document IDs, labels, and any cluster assignments added later.

    Parameters
    ----------
    dim : int
        Embedding dimensionality; must match the model's output.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        # IndexFlatIP: exact brute-force cosine search (inner product after
        # L2 normalisation).  Memory: dim × 4 bytes per vector ≈ 1.5 kB per doc.
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._meta: List[dict] = []

    # ── write path ───────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """
        Add a batch of L2-normalised embeddings and their associated metadata.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, dim)``, dtype ``float32``.  Must already be
            L2-normalised (EmbeddingModel guarantees this).
        metadata : list[dict]
            One dict per row of ``embeddings``.  Arbitrary keys are allowed;
            at minimum callers should include ``doc_id`` and ``target_name``.
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"embeddings batch size ({embeddings.shape[0]}) ≠ "
                f"metadata length ({len(metadata)})"
            )
        # FAISS mutates the array in-place when adding; we pass a C-contiguous
        # copy to avoid unexpected aliasing.
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._meta.extend(metadata)

    # ── read path ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[float, dict]]:
        """
        Return the *k* most similar stored documents to ``query_vec``.

        Parameters
        ----------
        query_vec : np.ndarray
            Shape ``(dim,)`` or ``(1, dim)``, L2-normalised.
        k : int
            Number of nearest neighbours to retrieve.

        Returns
        -------
        list[tuple[float, dict]]
            Each element is ``(cosine_similarity, metadata_dict)`` sorted
            from most to least similar.
        """
        query_vec = np.ascontiguousarray(
            query_vec.reshape(1, -1), dtype=np.float32
        )
        k = min(k, self._index.ntotal)
        if k == 0:
            return []

        distances, indices = self._index.search(query_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS sentinel for unfilled slots
                continue
            results.append((float(dist), self._meta[idx]))
        return results

    @property
    def size(self) -> int:
        """Total number of vectors currently stored in the index."""
        return self._index.ntotal

    def get_all_embeddings(self) -> Optional[np.ndarray]:
        """
        Reconstruct the full embedding matrix from the flat index.

        This is used by the GMM fitting step in Phase 2, which needs the raw
        vectors to estimate Gaussian parameters.  ``IndexFlatIP`` stores
        vectors verbatim (no compression), so this is a zero-copy view.

        Returns ``None`` if the index is empty.
        """
        if self._index.ntotal == 0:
            return None
        # xb is the flat float32 array backing the FAISS index; reshape to
        # (N, dim) without copying.
        return faiss.vector_to_array(self._index.xb).reshape(-1, self.dim)
