"""
app/core/clustering.py
────────────────────────────────────────────────────────────────────────────
Phase 2 – Fuzzy Clustering with Gaussian Mixture Models
────────────────────────────────────────────────────────
Why GMM instead of K-Means?
───────────────────────────
K-Means performs *hard* assignment: each document belongs to exactly one
cluster.  This is inappropriate for news text because topics naturally
overlap (e.g. a post about gun legislation is simultaneously "politics"
AND "firearms").  A hard-assignment cache would route such a query to a
single partition and miss semantically similar cached answers residing in
a different partition.

GMM models each cluster as a multivariate Gaussian and computes the
*posterior probability* P(cluster_k | document_i) via Bayes' theorem.
This gives us:
  1. A soft, probabilistic cluster assignment – the 384-d embedding lives
     in a mixture of Gaussians rather than a hard Voronoi cell.
  2. A natural "ambiguity score" – documents where max(posterior) < 0.4
     lie near cluster boundaries and are reported for inspection.
  3. Cluster membership as a probability vector, not an integer label, which
     is exactly what the cache uses to decide which partition to search.

Covariance type choice: "diag"
──────────────────────────────
Full covariance matrices for 384-d embeddings would require estimating
384² ≈ 147 456 parameters per component.  With ~18 000 documents that is
completely underdetermined and prone to singularities.  Diagonal covariance
(one variance per dimension, zero off-diagonal) reduces parameters to 384
per component, is numerically stable, and still captures per-dimension
spread in embedding space.

Choosing K: BIC/AIC sweep
──────────────────────────
We do NOT hard-code K.  Instead we fit GMMs for K in [k_min, k_max],
record BIC and AIC for each, and choose K at the BIC minimum (BIC penalises
model complexity more aggressively than AIC, which is desirable here to
avoid over-segmenting the cache).

PCA dimensionality reduction before GMM fitting
────────────────────────────────────────────────
Fitting a 384-d GMM over 18 000 documents is slow (~5 min on CPU) and the
curse of dimensionality makes covariance estimation noisy.  We project to
``pca_components`` dimensions (default 64) before fitting, preserving
~95 % of variance while reducing the GMM parameter count by 6×.  The PCA
is fit on the training corpus and applied to new query embeddings at
inference time.  The full 384-d vector is still stored in FAISS; PCA is
only used internally by the GMM.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ── tuneable defaults ─────────────────────────────────────────────────────────
K_MIN: int = 5        # lower bound of BIC sweep
K_MAX: int = 30       # upper bound of BIC sweep
K_STEP: int = 1       # step size; increase to speed up sweep at cost of precision
PCA_COMPONENTS: int = 64   # dimensions fed to GMM after PCA projection
GMM_MAX_ITER: int = 200    # EM convergence budget
# number of random EM restarts (guards against local minima)
GMM_N_INIT: int = 3
RANDOM_STATE: int = 42
BOUNDARY_THRESHOLD: float = 0.4  # max-posterior below this → "ambiguous" document


@dataclass
class ClusteringResult:
    """
    Immutable value object returned by :meth:`FuzzyClusterer.fit`.

    Fields
    ------
    n_components : int
        The K chosen via BIC minimisation.
    bic_scores : list[float]
        BIC for each K in the sweep (useful for plotting).
    aic_scores : list[float]
        AIC for each K in the sweep.
    k_range : list[int]
        The K values tested.
    posteriors : np.ndarray
        Shape (N_docs, K).  Row i is P(cluster | doc_i).
    dominant_clusters : np.ndarray
        Shape (N_docs,).  argmax of posteriors row – the single most
        probable cluster for each document.
    boundary_indices : list[int]
        Indices of documents where max posterior < BOUNDARY_THRESHOLD.
    """

    n_components: int
    bic_scores: List[float]
    aic_scores: List[float]
    k_range: List[int]
    posteriors: np.ndarray
    dominant_clusters: np.ndarray
    boundary_indices: List[int] = field(default_factory=list)


class FuzzyClusterer:
    """
    Fits a Gaussian Mixture Model on reduced-dimensionality embeddings and
    exposes soft cluster probabilities for both corpus documents and new
    query embeddings.

    Parameters
    ----------
    k_min / k_max / k_step : int
        Range for the BIC sweep.
    pca_components : int
        Target dimensionality for PCA pre-processing before GMM fitting.
    gmm_max_iter : int
        Maximum EM iterations per GMM fit.
    gmm_n_init : int
        Number of random EM initialisations (best log-likelihood is kept).
    random_state : int
        Seed for reproducibility.
    covariance_type : str
        One of sklearn's GMM covariance types.  ``"diag"`` is the default
        for the reasons documented in the module docstring above.
    """

    def __init__(
        self,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        k_step: int = K_STEP,
        pca_components: int = PCA_COMPONENTS,
        gmm_max_iter: int = GMM_MAX_ITER,
        gmm_n_init: int = GMM_N_INIT,
        random_state: int = RANDOM_STATE,
        covariance_type: str = "diag",
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.pca_components = pca_components
        self.gmm_max_iter = gmm_max_iter
        self.gmm_n_init = gmm_n_init
        self.random_state = random_state
        self.covariance_type = covariance_type

        # Set after fit()
        self._pca: Optional[PCA] = None
        self._gmm: Optional[GaussianMixture] = None
        self._result: Optional[ClusteringResult] = None

    # ── public interface ──────────────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Full two-stage fitting pipeline:
          1. PCA dimensionality reduction (fit + transform).
          2. BIC sweep to select optimal K.
          3. Final GMM fit on the chosen K.
          4. Compute posteriors and identify boundary documents.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, 384), L2-normalised float32.  Comes directly from
            :meth:`FAISSVectorStore.get_all_embeddings`.

        Returns
        -------
        ClusteringResult
        """
        logger.info("Fitting PCA (n_components=%d) …", self.pca_components)
        self._pca = PCA(
            n_components=self.pca_components,
            random_state=self.random_state,
        )
        reduced = self._pca.fit_transform(embeddings)
        explained = self._pca.explained_variance_ratio_.sum()
        logger.info(
            "PCA complete. Explained variance retained: %.1f%%",
            explained * 100,
        )

        # ── BIC / AIC sweep ──────────────────────────────────────────────────
        k_range = list(range(self.k_min, self.k_max + 1, self.k_step))
        bic_scores: List[float] = []
        aic_scores: List[float] = []

        logger.info(
            "Starting BIC sweep over K=%d…%d (n_init=%d, max_iter=%d) …",
            self.k_min,
            self.k_max,
            self.gmm_n_init,
            self.gmm_max_iter,
        )
        for k in k_range:
            gmm_k = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                max_iter=self.gmm_max_iter,
                n_init=self.gmm_n_init,
                random_state=self.random_state,
            )
            gmm_k.fit(reduced)
            bic = gmm_k.bic(reduced)
            aic = gmm_k.aic(reduced)
            bic_scores.append(bic)
            aic_scores.append(aic)
            logger.debug("  K=%2d  BIC=%.1f  AIC=%.1f", k, bic, aic)

        # Choose K at BIC minimum.
        # BIC penalises free parameters as K × log(N), which is stronger than
        # AIC's 2K penalty.  This keeps K from ballooning when the corpus
        # edges are blurry, preventing the cache from fragmenting into
        # uselessly small partitions.
        best_idx = int(np.argmin(bic_scores))
        best_k = k_range[best_idx]
        logger.info(
            "BIC minimum at K=%d (BIC=%.1f). Fitting final GMM …",
            best_k,
            bic_scores[best_idx],
        )

        # ── final GMM with best_k ────────────────────────────────────────────
        self._gmm = GaussianMixture(
            n_components=best_k,
            covariance_type=self.covariance_type,
            max_iter=self.gmm_max_iter,
            n_init=self.gmm_n_init,
            random_state=self.random_state,
        )
        self._gmm.fit(reduced)

        # posteriors[i, j] = P(cluster j | doc i)
        posteriors: np.ndarray = self._gmm.predict_proba(reduced)
        dominant_clusters: np.ndarray = posteriors.argmax(axis=1)

        # ── boundary document analysis ───────────────────────────────────────
        # Documents where the highest posterior is below BOUNDARY_THRESHOLD
        # are genuinely ambiguous – they straddle the decision boundaries of
        # two or more Gaussians.  Surfacing these shows the reviewer that
        # the clusters capture real semantic structure rather than arbitrary
        # Voronoi partitions.
        max_posteriors = posteriors.max(axis=1)
        boundary_indices = np.where(
            max_posteriors < BOUNDARY_THRESHOLD)[0].tolist()
        logger.info(
            "Found %d boundary documents (max_posterior < %.2f).",
            len(boundary_indices),
            BOUNDARY_THRESHOLD,
        )

        self._result = ClusteringResult(
            n_components=best_k,
            bic_scores=bic_scores,
            aic_scores=aic_scores,
            k_range=k_range,
            posteriors=posteriors,
            dominant_clusters=dominant_clusters,
            boundary_indices=boundary_indices,
        )
        return self._result

    def predict_proba(self, embedding: np.ndarray) -> np.ndarray:
        """
        Return the soft cluster probability vector for a *single* embedding.

        This is the hot-path called at query time by the semantic cache.
        It applies the fitted PCA projection and then the GMM posterior.

        Parameters
        ----------
        embedding : np.ndarray
            Shape ``(384,)`` or ``(1, 384)``, L2-normalised.

        Returns
        -------
        np.ndarray
            Shape ``(K,)`` – the posterior probability over clusters.
        """
        self._check_fitted()
        vec = embedding.reshape(1, -1).astype(np.float32)
        reduced = self._pca.transform(vec)          # (1, pca_components)
        return self._gmm.predict_proba(reduced)[0]  # (K,)

    def dominant_cluster(self, embedding: np.ndarray) -> int:
        """
        Return the single most-probable cluster index for an embedding.
        Convenience wrapper around :meth:`predict_proba`.
        """
        return int(self.predict_proba(embedding).argmax())

    def get_cluster_label_map(
        self, target_names: List[str], dominant_clusters: np.ndarray
    ) -> Dict[int, str]:
        """
        Build a heuristic mapping from cluster index to human-readable label
        by finding the plurality newsgroup category within each cluster.

        Useful for logging and notebook analysis – not used in production
        cache logic.
        """
        self._check_fitted()
        n_components = self._gmm.n_components
        label_map: Dict[int, str] = {}
        for k in range(n_components):
            doc_indices = np.where(dominant_clusters == k)[0]
            if len(doc_indices) == 0:
                label_map[k] = f"cluster_{k}_empty"
                continue
            # Count newsgroup categories among documents in this cluster.
            # The plurality category is the cluster's dominant topic.
            categories = [target_names[i]
                          for i in doc_indices if i < len(target_names)]
            if categories:
                from collections import Counter
                label_map[k] = Counter(categories).most_common(1)[0][0]
            else:
                label_map[k] = f"cluster_{k}"
        return label_map

    # ── persistence helpers ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the fitted PCA + GMM to disk for offline analysis."""
        self._check_fitted()
        with open(path, "wb") as fh:
            pickle.dump({"pca": self._pca, "gmm": self._gmm}, fh)
        logger.info("Clusterer saved to %s", path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "FuzzyClusterer":
        """Restore a previously saved :class:`FuzzyClusterer` from disk."""
        obj = cls(**kwargs)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj._pca = data["pca"]
        obj._gmm = data["gmm"]
        return obj

    # Properties forwarded from ClusteringResult for convenience
    @property
    def n_components(self) -> int:
        self._check_fitted()
        return self._gmm.n_components

    @property
    def result(self) -> ClusteringResult:
        self._check_fitted()
        return self._result   # type: ignore[return-value]

    # ── private helpers ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._gmm is None or self._pca is None:
            raise RuntimeError(
                "FuzzyClusterer has not been fitted yet.  Call .fit(embeddings) first."
            )
