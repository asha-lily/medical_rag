"""Retrieval metrics computed against evidence quotes in the ground truth dataset.

Recall is measured over evidence items (facts), not chunks. With chunk
overlap, one quote can appear in two chunks; counting facts avoids
double-counting, and matches what we care about: did the generator receive
each fact it needs?

All metrics are derived from one list per question: the rank (1-based) at
which each evidence item first appears in the retrieved chunks, or None if it
doesn't appear.
"""

import logging
import re
import unicodedata

from langchain_core.documents import Document

from rag_pipeline.ground_truth_data_set import Evidence, GroundTruthQuestion

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_DASHES = re.compile(r"[‐‑‒–—−]")
_WHITESPACE = re.compile(r"\s+")

# Matching ignores whitespace, so a short quote can match across word
# boundaries ("is not" matches inside "this not"). Longer quotes make a
# false match very unlikely.
MIN_QUOTE_CHARS = 20  # excluding whitespace
MAX_EXPECTED_MATCHES = 2


def normalise(text: str) -> str:
    """Clean up text extracted from PDFs so it can be compared reliably.

    PDF text often contains characters that look the same as ordinary ones
    but aren't, such as styled Unicode variants, invisible soft hyphens,
    curly quotes and different kinds of dashes. This replaces them with
    plain equivalents, removes all whitespace and converts the text to
    lowercase, so strings that look the same will compare equal.

    Whitespace is removed entirely, not just tidied, because PDF extraction
    often adds or drops spaces unpredictably (e.g. "inform ation").
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.translate(_QUOTE_MARKS)
    text = _DASHES.sub("-", text)
    text = _WHITESPACE.sub("", text)
    return text.lower()


def contains_evidence(chunk: Document, evidence: Evidence) -> bool:
    """True if the chunk comes from the evidence's source and contains its quote."""
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


def reciprocal_rank(ranks: list[int | None]) -> float:
    """1 / rank of the first retrieved chunk containing any evidence; 0 if none."""
    found = [r for r in ranks if r is not None]
    return 1 / min(found) if found else 0.0


def check_evidence_findable(
    questions: list[GroundTruthQuestion], chunks: list[Document]
) -> set[str]:
    """Log any evidence quote that isn't in any chunk; return the affected question IDs.

    Also logs quotes that are short or match many chunks, which may be
    matching in the wrong place. These are warnings only; the question is
    still evaluated.
    """
    unfindable = set()
    for q in questions:
        for ev in q.evidence:
            quote = ev.quote.strip()
            if len(normalise(quote)) < MIN_QUOTE_CHARS:
                log.warning(
                    "%s: quote is shorter than %d characters, so it may match "
                    "in the wrong place; consider a longer quote: %r",
                    q.id,
                    MIN_QUOTE_CHARS,
                    quote,
                )
            n_matches = sum(contains_evidence(chunk, ev) for chunk in chunks)
            if n_matches == 0:
                unfindable.add(q.id)
                log.warning(
                    "%s: quote not found in any chunk of %s: %r",
                    q.id,
                    ev.source,
                    quote,
                )
            elif n_matches > MAX_EXPECTED_MATCHES:
                log.warning(
                    "%s: quote found in %d chunks of %s (expected at most %d); "
                    "check it isn't repeated text or a false match: %r",
                    q.id,
                    n_matches,
                    ev.source,
                    MAX_EXPECTED_MATCHES,
                    quote,
                )
    return unfindable


def evaluate_question(
    q: GroundTruthQuestion,
    retrieved: list[Document],
    k_values: tuple,
) -> dict:
    ranks = evidence_ranks(retrieved, q.evidence)

    row = {
        "id": q.id,
        "category": q.category,
        "n_evidence": len(q.evidence),
        "evidence_ranks": ranks,
        "reciprocal_rank": reciprocal_rank(ranks),
    }
    for k in k_values:
        row[f"recall@{k}"] = recall_at_k(ranks, k)
    return row
