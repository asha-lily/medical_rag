"""Evaluate retrieval against the golden dataset.

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

from rag_pipeline.golden_set import GoldenQuestion, load_golden_set
from rag_pipeline.retrieval import retrieve
from rag_pipeline.retrieval_metrics import (
    all_found_at_k,
    contains_evidence,
    evidence_ranks,
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


def _all_chunks(store) -> list[Document]:
    """Every chunk in the vector store, as Documents."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def check_evidence_findable(
    questions: list[GoldenQuestion], chunks: list[Document]
) -> set[str]:
    """Log any evidence quote that isn't in any chunk; return the affected question IDs."""
    unfindable = set()
    for q in questions:
        for ev in q.evidence:
            n_matches = sum(contains_evidence(chunk, ev) for chunk in chunks)
            if n_matches == 0:
                unfindable.add(q.id)
                log.warning(
                    "%s: quote not found in any chunk of %s: %r",
                    q.id,
                    ev.source,
                    ev.quote,
                )
    return unfindable


def evaluate_question(q: GoldenQuestion, retrieved: list[Document]) -> dict:
    ranks = evidence_ranks(retrieved, q.evidence)
    essential_ranks = [r for r, ev in zip(ranks, q.evidence) if ev.essential]

    row = {
        "id": q.id,
        "risk": q.risk,
        "category": q.category,
        "n_evidence": len(q.evidence),
        "n_essential": len(essential_ranks),
        "evidence_ranks": ranks,
        "reciprocal_rank": reciprocal_rank(ranks),
    }
    for k in K_VALUES:
        row[f"recall@{k}"] = recall_at_k(ranks, k)
        # None when a question has no essential evidence, so it's left out of averages.
        row[f"all_essential@{k}"] = (
            all_found_at_k(essential_ranks, k) if essential_ranks else None
        )
    return row


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric, overall and by risk level."""
    metric_cols = ["reciprocal_rank"] + [
        f"{name}@{k}" for name in ("recall", "all_essential") for k in K_VALUES
    ]
    numeric = results[metric_cols].astype(float)
    overall = numeric.mean().to_frame("all").T
    by_risk = numeric.groupby(results["risk"]).mean()
    summary = pd.concat([overall, by_risk])
    summary.insert(
        0,
        "n_questions",
        [len(results)] + results["risk"].value_counts().reindex(by_risk.index).tolist(),
    )
    return summary.rename(columns={"reciprocal_rank": "MRR"})


def run_retrieval_evaluation(label: str, check_only: bool) -> None:
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first."
        )

    questions = load_golden_set()
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
