import ollama
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from rag_pipeline.config import EmbeddingModelConfig, RAGASConfig

log = logging.getLogger(__name__)


def _make_ragas_llm(config: RAGASConfig) -> LangchainLLMWrapper:
    """Wrap a Langchain chat Ollama model in a RAGAS LLM interface."""
    return LangchainLLMWrapper(
        ChatOllama(
            model=config.model_name,
            num_predict=config.max_tokens,
            temperature=config.temperature,
            reasoning=config.reasoning_level,
            num_ctx=config.context_window,
        )
    )


def _make_ragas_embeddings(config: EmbeddingModelConfig) -> LangchainEmbeddingsWrapper:
    """Wrap a Langchain embedding model in a RAGAS embeddings interface."""
    emb = HuggingFaceEmbeddings(
        model_name=config.model_name,
        encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
    )
    return LangchainEmbeddingsWrapper(emb)


def _build_metrics(
    ragas_config: RAGASConfig,
    emb_config: EmbeddingModelConfig,
) -> list:
    """Return RAGAS metrics configured to use the local Ollama model & HuggingFace embeddings."""
    ragas_llm = _make_ragas_llm(ragas_config)
    ragas_embeddings = _make_ragas_embeddings(emb_config)
    return [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]


def require_ollama_model(model_name: str) -> None:
    """Fail fast if Ollama isn't running or the model hasn't been pulled."""
    try:
        ollama.Client().show(model_name)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(
                f"Ollama model {model_name!r} not found. Run `ollama pull {model_name}`."
            ) from e
        raise
    except ConnectionError as e:
        raise RuntimeError("Can't reach Ollama. Is `ollama serve` running?") from e


def evaluate_rag_samples(
    samples: list[dict],
    ragas_config: RAGASConfig | None = None,
    emb_config: EmbeddingModelConfig | None = None,
):
    """Run RAGAS evaluation on a list of collected RAG outputs.

    Each sample dict must have:
        question  (str)
        answer    (str)
        contexts  (list[str])   — the retrieved chunk texts

    Optionally:
        ground_truth (str) — needed for context_precision / context_recall metrics
    """
    if ragas_config is None:
        ragas_config = RAGASConfig()
    if emb_config is None:
        emb_config = EmbeddingModelConfig()

    require_ollama_model(ragas_config.model_name)

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("ground_truth", ""),
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    run_config = RunConfig(
        timeout=ragas_config.timeout_seconds,
        max_workers=ragas_config.max_workers,
    )
    metrics = _build_metrics(ragas_config, emb_config)
    log.info("Running RAGAS on %d samples with %d metrics.", len(samples), len(metrics))
    return evaluate(dataset, metrics=metrics, run_config=run_config)
