"""Retrieval metrics computed against evidence quotes in the ground truth dataset.

Recall is measured over evidence items (facts), not chunks. With chunk
overlap, one quote can appear in two chunks; counting facts avoids
double-counting, and matches what we care about: did the generator receive
each fact it needs?

All metrics are derived from one list per question: the rank (1-based) at
which each evidence item first appears in the retrieved chunks, or None if it
doesn't appear.
"""

import re
import unicodedata

from langchain_core.documents import Document
from rag_pipeline.ground_truth_data_set import Evidence

_QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_DASHES = re.compile(r"[‐‑‒–—−]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Normalise text so cosmetic PDF-parsing differences don't block a match.

    All whitespace is removed, because the PDF parser sometimes drops spaces
    ("are breast-feeding" becomes "arebreast-feeding") or adds them
    ("painkillers , you"). The result is for matching, not for display.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphens
    text = text.translate(_QUOTE_MARKS)
    text = _DASHES.sub("-", text)
    text = _WHITESPACE.sub("", text)
    return text.lower()


def contains_evidence(chunk: Document, evidence: Evidence) -> bool:
    """True if the chunk comes from the evidence's source and contains its quote
    (ignoring case, whitespace and cosmetic punctuation differences)."""
    if chunk.metadata.get("source") != evidence.source:
        return False
    return normalise(evidence.quote) in normalise(chunk.page_content)


def evidence_ranks(
    retrieved: list[Document], evidence: list[Evidence]
) -> list[int | None]:
    """For each evidence item, the 1-based rank of the first retrieved chunk
    containing it, or None if no retrieved chunk does."""
    ranks = []
    for item in evidence:
        rank = next(
            (
                i
                for i, chunk in enumerate(retrieved, start=1)
                if contains_evidence(chunk, item)
            ),
            None,
        )
        ranks.append(rank)
    return ranks


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of evidence items found in the top k."""
    if not ranks:
        raise ValueError("recall is undefined for a question with no evidence")
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def all_found_at_k(ranks: list[int | None], k: int) -> bool:
    """True if every evidence item was found in the top k."""
    return all(r is not None and r <= k for r in ranks)


def reciprocal_rank(ranks: list[int | None]) -> float:
    """1 / rank of the first retrieved chunk containing any evidence; 0 if none."""
    found = [r for r in ranks if r is not None]
    return 1 / min(found) if found else 0.0
