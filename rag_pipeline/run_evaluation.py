"""Run retrieval and generation once over the ground truth set, then score both.

uv run python -m rag_pipeline.run_evaluation --label baseline --collect-only
# ... manually label each record's "refusal" as true/false in results/run_baseline.json ...

uv run python -m rag_pipeline.run_evaluation --from-run baseline   # score a saved run
uv run python -m rag_pipeline.run_evaluation --from-run baseline --skip-ragas

Collecting a run (retrieval + generation for every question) writes
results/run_<label>.json, with "refusal": null on every record. Retrieval,
refusal and RAGAS metrics are then computed from that file, so they all describe
exactly the same retrieved chunks and answers, and RAGAS (the slow part) can be
re-run without regenerating answers. Retrieval metrics don't need the refusal
labels; refusal and RAGAS metrics do.
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import pandas as pd
from langchain_core.documents import Document

from rag_pipeline.config import (
    ChunkingConfig,
    EmbeddingModelConfig,
    GenerationConfig,
    RAGASConfig,
    RetrievalConfig,
)
from rag_pipeline.generation import create_answer_chain
from rag_pipeline.ground_truth_data_set import (
    GroundTruthQuestion,
    load_ground_truth_set,
)
from rag_pipeline.ragas_evaluation import evaluate_rag_samples
from rag_pipeline.retrieval import retrieve
from rag_pipeline.retrieval_metrics import (
    evidence_ranks,
    recall_at_k,
    reciprocal_rank,
    evaluate_question,
    summarise,
    check_evidence_findable,
)

from rag_pipeline.vector_store import (
    create_embedding_model,
    load_vector_store,
    vector_store_exists,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Retrieval is scored on the chunks the generator sees, so the largest k is
# the generator's k.
GENERATION_K = RetrievalConfig().k_chunks_to_retrieve
K_VALUES = tuple(k for k in (1, 3) if k < GENERATION_K) + (GENERATION_K,)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAGAS_METRICS = ("faithfulness", "answer_relevancy")


def _all_chunks(store) -> list[Document]:
    """Every chunk in the vector store, as Documents."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def _doc_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def _dict_to_doc(d: dict) -> Document:
    return Document(page_content=d["page_content"], metadata=d["metadata"])


def _run_path(label: str):
    return RESULTS_DIR / f"run_{label}.json"


def collect_run(label: str) -> dict:
    """Retrieve and generate for every ground truth question; save and return the run."""
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first."
        )

    questions = load_ground_truth_set()
    embedding_model = create_embedding_model()
    store = load_vector_store(embedding_model)
    answer_chain = create_answer_chain()

    # Checked against the index this run used, so a later re-score of this
    # run isn't affected by re-indexing.
    unfindable = check_evidence_findable(
        [q for q in questions if q.answerable], _all_chunks(store)
    )

    records = []
    for i, q in enumerate(questions, start=1):
        log.info("[%d/%d] %s: %s", i, len(questions), q.id, q.question)
        docs = retrieve(q.question, store, embedding_model, k=GENERATION_K)
        answer = answer_chain.invoke({"question": q.question, "docs": docs})
        records.append(
            {
                "id": q.id,
                "question": q.question,
                "category": q.category,
                "answerable": q.answerable,
                "reference_answer": q.reference_answer,
                "answer": answer,
                # Labelled by hand as true/false before scoring.
                "refusal": None,
                "docs": [_doc_to_dict(d) for d in docs],
            }
        )

    run = {
        "label": label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "chunking": asdict(ChunkingConfig()),
            "embedding": asdict(EmbeddingModelConfig()),
            "retrieval": asdict(RetrievalConfig()),
            "generation": asdict(GenerationConfig()),
        },
        "unfindable_evidence": sorted(unfindable),
        "records": records,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = _run_path(label)
    with open(path, "w") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    log.info("Saved run to %s", path)
    return run


def load_run(label: str) -> dict:
    with open(_run_path(label)) as f:
        return json.load(f)


def _summarise_retrieval(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric."""
    metric_cols = ["reciprocal_rank"] + [f"recall@{k}" for k in K_VALUES]
    overall = results[metric_cols].astype(float).mean().to_frame("all").T
    overall.insert(0, "n_questions", [len(results)])
    return overall.rename(columns={"reciprocal_rank": "MRR"})


def score_retrieval(run: dict) -> pd.DataFrame:
    """Retrieval metrics for answerable questions whose evidence is findable."""
    questions = {q.id: q for q in load_ground_truth_set()}
    excluded = set(run["unfindable_evidence"])
    if excluded:
        log.warning("Excluding from retrieval metrics: %s", ", ".join(sorted(excluded)))

    rows = []
    for rec in run["records"]:
        if not rec["answerable"] or rec["id"] in excluded:
            continue
        q = questions.get(rec["id"])
        if q is None:
            log.warning("%s: no longer in the ground truth set; skipping", rec["id"])
            continue
        rows.append(
            evaluate_question(q, [_dict_to_doc(d) for d in rec["docs"]], K_VALUES)
        )
    return pd.DataFrame(rows)


def validate_refusal_labels(run: dict) -> None:
    """Raise if any record's `refusal` label is missing or not true/false."""
    bad = [
        rec["id"] for rec in run["records"] if not isinstance(rec.get("refusal"), bool)
    ]
    if bad:
        raise ValueError(
            f"`refusal` must be true or false; missing or invalid for: {', '.join(bad)}"
        )


def _outcome(answerable: bool, refusal: bool) -> str:
    if answerable:
        return "false_refusal" if refusal else "answered"
    return "correct_refusal" if refusal else "missed_refusal"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_refusals(run: dict) -> tuple[pd.DataFrame, dict]:
    """Per-question refusal outcomes, plus refusal / missed / false refusal rates.

    Expects validated labels (see `validate_refusal_labels`).
    """
    rows = [
        {
            "id": rec["id"],
            "category": rec["category"],
            "answerable": rec["answerable"],
            "refusal": rec["refusal"],
            "outcome": _outcome(rec["answerable"], rec["refusal"]),
        }
        for rec in run["records"]
    ]
    df = pd.DataFrame(
        rows, columns=["id", "category", "answerable", "refusal", "outcome"]
    )

    unanswerable = df[~df["answerable"].astype(bool)]
    answerable = df[df["answerable"].astype(bool)]
    n_refused = int(unanswerable["refusal"].sum())
    n_false = int(answerable["refusal"].sum())
    refusal_rate = _rate(n_refused, len(unanswerable))
    summary = {
        "n_unanswerable": len(unanswerable),
        "n_refused_unanswerable": n_refused,
        "refusal_rate": refusal_rate,
        "missed_refusal_rate": None if refusal_rate is None else 1 - refusal_rate,
        "n_answerable": len(answerable),
        "n_refused_answerable": n_false,
        "false_refusal_rate": _rate(n_false, len(answerable)),
    }
    return df, summary


def _format_rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a (no questions)"
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def score_ragas(run: dict) -> pd.DataFrame:
    """RAGAS metrics for questions that are answerable and were answered.

    AnswerRelevancy scores a refusal as 0, so refusals are left out: correct
    refusals are covered by the refusal rate, and refusals of answerable
    questions by the false refusal rate. This means the number of questions
    scored depends on the model's behaviour, so read these means alongside the
    false refusal rate.
    """
    records = [
        rec for rec in run["records"] if rec["answerable"] and not rec["refusal"]
    ]
    samples = [
        {
            "question": rec["question"],
            "answer": rec["answer"],
            "contexts": [d["page_content"] for d in rec["docs"]],
            "ground_truth": rec["reference_answer"],
        }
        for rec in records
    ]
    result = evaluate_rag_samples(samples, RAGASConfig())
    df = result.to_pandas()
    df.insert(0, "id", [rec["id"] for rec in records])

    for metric in RAGAS_METRICS:
        missing = df.loc[df[metric].isna(), "id"].tolist()
        if len(missing) == len(df):
            raise RuntimeError(
                f"Every {metric} score is NaN, so all judge calls probably failed. "
                "Check the RAGAS log for exceptions."
            )
        if missing:
            log.warning(
                "%s is NaN for %d/%d questions (timeout or unparseable judge output?): %s",
                metric,
                len(missing),
                len(df),
                ", ".join(missing),
            )
    return df


def run_all_evaluation(
    label: str, from_run: str | None, skip_ragas: bool, collect_only: bool = False
) -> None:
    run = load_run(from_run) if from_run else collect_run(label)
    label = run["label"]

    if collect_only:
        print(
            f"\nLabel each record's `refusal` as true/false in {_run_path(label)}, "
            f"then run with --from-run {label}"
        )
        return

    retrieval = score_retrieval(run)
    retrieval_results_path = RESULTS_DIR / f"run_{label}_retrieval.csv"
    retrieval.to_csv(retrieval_results_path, index=False)
    print(
        f"\n=== Retrieval metrics ({len(retrieval)} questions, MRR@{GENERATION_K}) ==="
    )
    print(_summarise_retrieval(retrieval).round(2).to_string())
    print(f"Per-question results saved to {retrieval_results_path}")

    validate_refusal_labels(run)
    refusals, summary = score_refusals(run)
    refusals_results_path = RESULTS_DIR / f"run_{label}_refusals.csv"
    refusals.to_csv(refusals_results_path, index=False)
    n_unans, n_ans = summary["n_unanswerable"], summary["n_answerable"]
    n_refused, n_false = (
        summary["n_refused_unanswerable"],
        summary["n_refused_answerable"],
    )
    print("\n=== Refusal metrics ===")
    print(f"Refusal rate:         {_format_rate(n_refused, n_unans)}")
    print(f"Missed refusal rate:  {_format_rate(n_unans - n_refused, n_unans)}")
    print(f"False refusal rate:   {_format_rate(n_false, n_ans)}")
    missed = refusals.loc[refusals["outcome"] == "missed_refusal", "id"].tolist()
    if missed:
        print(f"Missed refusals: {', '.join(missed)}")
    print(f"Per-question results saved to {refusals_results_path}")

    if skip_ragas:
        return
    ragas = score_ragas(run)
    ragas_results_path = RESULTS_DIR / f"run_{label}_ragas.csv"
    ragas.to_csv(ragas_results_path, index=False)
    print(f"\n=== RAGAS metrics ({len(ragas)} answerable, answered questions) ===")
    for metric in RAGAS_METRICS:
        print(f"{metric:<18} {ragas[metric].mean():.2f}  (n={ragas[metric].count()})")
    print(ragas[["faithfulness", "answer_relevancy"]].mean().round(2).to_string())
    print(f"Per-question results saved to {ragas_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--label",
        default=datetime.now().strftime("%Y%m%d_%H%M"),
        help="Name for a new run, used in the results file names.",
    )
    parser.add_argument(
        "--from-run",
        metavar="LABEL",
        help="Re-score a saved run instead of collecting a new one.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Only compute retrieval and refusal metrics.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect and save a run for refusal labelling, without scoring it.",
    )
    args = parser.parse_args()
    run_all_evaluation(args.label, args.from_run, args.skip_ragas, args.collect_only)
