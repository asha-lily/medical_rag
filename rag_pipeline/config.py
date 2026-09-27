from pathlib import Path
from dataclasses import dataclass


@dataclass
class PILDocsConfig:
    docs_dir: Path = Path(
        "/Users/ashapatel/Documents/projects/rag_cc/leaflets_and_guidelines/pils"
    )


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


@dataclass
class RAGASConfig:
    """Define which LLM to use for RAGAS evaluation."""

    model_name: str = "llama3.2"
    max_tokens: int = 1024
