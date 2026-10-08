"""Run retrieval and generation once over the ground truth set, then score both.

uv run python -m rag_pipeline.all_evaluation --label baseline
uv run python -m rag_pipeline.all_evaluation --from-run baseline   # re-score a saved run
uv run python -m rag_pipeline.all_evaluation --label baseline --skip-ragas

Collecting a run (retrieval + generation for every question) writes
results/run_<label>.json. Retrieval metrics and RAGAS metrics are then computed
from that file, so both describe exactly the same retrieved chunks and answers,
and RAGAS (the slow part) can be re-run without regenerating answers.
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


def score_ragas(run: dict) -> pd.DataFrame:
    """RAGAS metrics for answerable questions only.

    AnswerRelevancy scores a refusal as 0, so unanswerable questions (where a
    refusal is the right answer) would drag the average down for the wrong reason.
    """
    records = [rec for rec in run["records"] if rec["answerable"]]
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
    return df


def run_all_evaluation(label: str, from_run: str | None, skip_ragas: bool) -> None:
    run = load_run(from_run) if from_run else collect_run(label)
    label = run["label"]

    retrieval = score_retrieval(run)
    retrieval_results_path = RESULTS_DIR / f"run_{label}_retrieval.csv"
    retrieval.to_csv(retrieval_results_path, index=False)
    print(
        f"\n=== Retrieval metrics ({len(retrieval)} questions, MRR@{GENERATION_K}) ==="
    )
    print(_summarise_retrieval(retrieval).round(2).to_string())
    print(f"Per-question results saved to {retrieval_results_path}")

    if skip_ragas:
        return
    ragas = score_ragas(run)
    ragas_results_path = RESULTS_DIR / f"run_{label}_ragas.csv"
    ragas.to_csv(ragas_results_path, index=False)
    print(f"\n=== RAGAS metrics ({len(ragas)} answerable questions) ===")
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
        help="Only compute retrieval metrics.",
    )
    args = parser.parse_args()
    run_all_evaluation(args.label, args.from_run, args.skip_ragas)
