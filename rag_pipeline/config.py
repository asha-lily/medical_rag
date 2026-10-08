from pathlib import Path
from typing import Literal
from dataclasses import dataclass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ReasoningSetting = bool | Literal["low", "medium", "high"] | None


@dataclass
class PILDocsConfig:
    docs_dir: Path = _PROJECT_ROOT / "data" / "PILs"


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150


@dataclass
class EmbeddingModelConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    normalize_embeddings: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class VectorStoreConfig:
    vector_store_name: str = "PIL_docs_vector_store"


@dataclass
class RetrievalConfig:
    k_chunks_to_retrieve: int = 5


@dataclass
class GenerationConfig:
    model_name: str = "llama3.2"
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class RAGASConfig:
    """Define which LLM to use for RAGAS evaluation."""

    model_name: str = "gpt-oss:20b"
    temperature: float = 0.0

    # Needs to cover judge reasoning & output JSONs
    max_tokens: int = 4096
    reasoning_level: ReasoningSetting = "low"
    context_window: int = 20000

    # Number of requests that can be processed in parallel is limited by my laptop's processing power
    max_workers: int = 2
    timeout_seconds: int = 600
