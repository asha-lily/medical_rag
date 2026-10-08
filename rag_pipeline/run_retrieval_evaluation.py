"""Evaluate retrieval against the ground truth dataset.

uv run python -m rag_pipeline.run_retrieval_evaluation --check-only
uv run python -m rag_pipeline.run_retrieval_evaluation --label baseline

--check-only confirms every evidence quote can be found in the index, without
running retrieval. Run it whenever you add questions or change parsing or
chunking: a quote that can't be found is a labelling typo, a parsing problem,
or a quote split across two chunks, and that question would otherwise show up
as a retrieval failure.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from rag_pipeline.ground_truth_data_set import (
    GroundTruthQuestion,
    load_ground_truth_set,
)
from rag_pipeline.retrieval import retrieve
from rag_pipeline.retrieval_metrics import (
    all_found_at_k,
    contains_evidence,
    evidence_ranks,
    normalise,
    recall_at_k,
    reciprocal_rank,
)
from rag_pipeline.vector_store import (
    create_embedding_model,
    load_vector_store,
    vector_store_exists,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

K_VALUES = (1, 3, 5, 10)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Matching ignores whitespace, so a short quote can match across word
# boundaries ("is not" matches inside "this not"). Longer quotes make a
# false match very unlikely.
MIN_QUOTE_CHARS = 20  # excluding whitespace
MAX_EXPECTED_MATCHES = 2


def _all_chunks(store) -> list[Document]:
    """Every chunk in the vector store, as Documents."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


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


def evaluate_question(q: GroundTruthQuestion, retrieved: list[Document]) -> dict:
    ranks = evidence_ranks(retrieved, q.evidence)

    row = {
        "id": q.id,
        "category": q.category,
        "n_evidence": len(q.evidence),
        "evidence_ranks": ranks,
        "reciprocal_rank": reciprocal_rank(ranks),
    }
    for k in K_VALUES:
        row[f"recall@{k}"] = recall_at_k(ranks, k)
    return row


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric"""
    metric_cols = ["reciprocal_rank"] + [f"recall@{k}" for k in K_VALUES]
    numeric = results[metric_cols].astype(float)
    overall = numeric.mean().to_frame("all").T
    overall.insert(
        0,
        "n_questions",
        [len(results)],
    )
    return overall.rename(columns={"reciprocal_rank": "MRR"})


def run_retrieval_evaluation(label: str, check_only: bool) -> None:
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first."
        )

    questions = load_ground_truth_set()
    answerable = [q for q in questions if q.answerable]
    log.info(
        "Loaded %d questions (%d answerable, %d unanswerable).",
        len(questions),
        len(answerable),
        len(questions) - len(answerable),
    )

    embedding_model = create_embedding_model()
    store = load_vector_store(embedding_model)
    unfindable = check_evidence_findable(answerable, _all_chunks(store))
    if check_only:
        log.info(
            "Check complete: %d question(s) with unfindable evidence.", len(unfindable)
        )
        return
    if unfindable:
        log.warning("Excluding from metrics: %s", ", ".join(sorted(unfindable)))

    to_evaluate = [q for q in answerable if q.id not in unfindable]
    max_k = max(K_VALUES)
    rows = [
        evaluate_question(q, retrieve(q.question, store, embedding_model, k=max_k))
        for q in to_evaluate
    ]
    results = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_{label}.csv"
    results.to_csv(out_path, index=False)

    print(f"\n=== Retrieval metrics ({len(results)} questions, MRR@{max_k}) ===")
    print(summarise(results).round(2).to_string())
    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--label",
        default=datetime.now().strftime("%Y%m%d_%H%M"),
        help="Name for this run, used in the results file name.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check that every evidence quote can be found in the index.",
    )
    args = parser.parse_args()
    run_retrieval_evaluation(args.label, args.check_only)
