"""
app/pipeline/data.py
────────────────────────────────────────────────────────────────────────────
Phase 1 – Data Ingestion & Preprocessing
─────────────────────────────────────────
The 20 Newsgroups dataset is notorious for metadata contamination.  Each
document arrives with:
  • Email headers  – From/To/Date/Newsgroups lines that expose the class
    purely through known email addresses, poisoning embeddings.
  • Footers        – Reply signatures, PGP blocks, legal disclaimers.
  • Quotes         – Inline quoted text copied from prior messages, causing
    semantic duplication between threads.

If these artefacts are left in, `all-MiniLM-L6-v2` will encode posting
patterns instead of topical semantics, making cluster separability an
artefact of email infrastructure rather than meaning.

sklearn exposes a `remove` parameter on `fetch_20newsgroups` that strips
all three noise sources *before* we ever touch the text, making it the
single most important preprocessing decision in the whole pipeline.

After denoising we apply a lightweight cleaning pass:
  1. Lower-case normalisation  – embeddings are case-insensitive in practice,
     but consistent casing avoids tokeniser surprises.
  2. Whitespace collapse       – multiple spaces / newlines → single space.
  3. Minimum-length guard      – discard documents with fewer than
     MIN_TOKENS words; they carry insufficient semantic signal and can
     distort cluster centroids.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)

# ── tuneable hyper-parameters ────────────────────────────────────────────────
# Documents shorter than this (after cleaning) are too sparse to embed
# meaningfully.  The MiniLM tokeniser has a max context of 256 word-pieces;
# documents well below this limit produce embedding vectors that cluster
# around the mean of the model's token distribution rather than encoding
# topical semantics.  50 words is a pragmatic minimum.
MIN_TOKENS: int = 50

# Subsets to load.  Using "all" gives the widest vocabulary coverage for
# cluster discovery.  In a production system you might restrict to "train".
DEFAULT_SUBSET: str = "all"


# ── data model ───────────────────────────────────────────────────────────────
@dataclass
class NewsDocument:
    """A single preprocessed newsgroup document."""

    doc_id: int
    text: str                          # cleaned body text
    target: int                        # original integer label (0–19)
    target_name: str                   # human-readable category string
    raw_text: str = field(repr=False)  # kept for audit / debugging only


# ── pipeline class ───────────────────────────────────────────────────────────
class NewsgroupsPipeline:
    """
    Fetches, denoises, and delivers the 20 Newsgroups corpus as a list of
    :class:`NewsDocument` objects ready for embedding.

    Parameters
    ----------
    subset : str
        One of "train", "test", or "all".  Defaults to "all" so the GMM
        sees the full vocabulary distribution when fitting.
    min_tokens : int
        Minimum document length (in whitespace-tokenised words) to retain.
    categories : list[str] | None
        Optionally restrict to a subset of the 20 categories.  ``None``
        loads all 20.
    """

    def __init__(
        self,
        subset: str = DEFAULT_SUBSET,
        min_tokens: int = MIN_TOKENS,
        categories: Optional[List[str]] = None,
    ) -> None:
        self.subset = subset
        self.min_tokens = min_tokens
        self.categories = categories
        self._documents: List[NewsDocument] = []

    # ── public interface ─────────────────────────────────────────────────────

    def load(self) -> "NewsgroupsPipeline":
        """
        Run the full ingestion + cleaning pipeline.  Returns *self* to allow
        chaining:  ``pipeline.load().documents``.
        """
        logger.info("Fetching 20 Newsgroups (subset=%s) …", self.subset)

        # The `remove` kwarg is the critical data-hygiene step:
        #   • 'headers'  strips the entire RFC-822 header block (From, Subject,
        #     Organization, X-* fields, etc.).
        #   • 'footers'  removes signature blocks that follow a "-- " separator.
        #   • 'quotes'   strips lines beginning with ">" (inline reply quotes).
        raw_dataset = fetch_20newsgroups(
            subset=self.subset,
            remove=("headers", "footers", "quotes"),
            categories=self.categories,
        )

        target_names: List[str] = raw_dataset.target_names
        dropped = 0

        for idx, (raw_text, target_int) in enumerate(
            zip(raw_dataset.data, raw_dataset.target)
        ):
            cleaned = self._clean(raw_text)

            # Length gate: discard documents too short to carry semantic content.
            # These become pseudo-outliers in embedding space and add noise to
            # the GMM covariance estimates.
            if len(cleaned.split()) < self.min_tokens:
                dropped += 1
                continue

            self._documents.append(
                NewsDocument(
                    doc_id=idx,
                    text=cleaned,
                    target=int(target_int),
                    target_name=target_names[int(target_int)],
                    raw_text=raw_text,
                )
            )

        logger.info(
            "Loaded %d documents (dropped %d below %d-token threshold).",
            len(self._documents),
            dropped,
            self.min_tokens,
        )
        return self

    @property
    def documents(self) -> List[NewsDocument]:
        """Return the processed document list (call :meth:`load` first)."""
        if not self._documents:
            raise RuntimeError("Call .load() before accessing .documents")
        return self._documents

    @property
    def texts(self) -> List[str]:
        """Convenience accessor – returns the cleaned text of every document."""
        return [doc.text for doc in self.documents]

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """
        Lightweight but effective text normalisation.

        Steps (order matters):
        1. Lower-case  – reduces vocabulary size without semantic loss for
           English text modelled by a cased tokeniser.
        2. Collapse repeated whitespace / line-breaks – the Usenet corpus is
           full of visual formatting that adds no semantic signal.
        3. Strip leading/trailing whitespace produced by step 2.

        We intentionally avoid stemming, stopword removal, or punctuation
        stripping because sentence-transformers are trained on full natural
        language sentences and degrade when those are removed.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()
