import logging

from rag_pipeline.evaluation import evaluate_rag_samples
from rag_pipeline.generation import create_rag_chain
from rag_pipeline.retrieval import create_retriever
from rag_pipeline.vector_store import (
    create_embedding_model,
    load_vector_store,
    vector_store_exists,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# A small test set of questions about the PIL documents.
TEST_SET = [
    {"question": "", "ground_truth": ""},
]


def _collect_sample(question: str, chain) -> dict:
    """Run retrieval and generation for one question, capturing both contexts and answer."""
    result = chain.invoke(question)
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": [doc.page_content for doc in result["docs"]],
    }


def run_evaluation() -> None:
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first."
        )

    embedding_model = create_embedding_model()
    store = load_vector_store(embedding_model)
    retriever = create_retriever(store, embedding_model)
    chain = create_rag_chain(retriever)

    log.info("Collecting answers for %d test questions...", len(TEST_SET))
    samples = []
    for entry in TEST_SET:
        q = entry["question"]
        log.info("  Q: %s", q)
        sample = _collect_sample(q, chain)
        sample["ground_truth"] = entry.get("ground_truth", "")
        samples.append(sample)

    log.info("Running RAGAS evaluation...")
    result = evaluate_rag_samples(samples)

    print("\n=== RAGAS Results ===")
    print(result)
    print()
    df = result.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))


if __name__ == "__main__":
    run_evaluation()
