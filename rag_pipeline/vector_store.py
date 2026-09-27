from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline.config import VectorStoreConfig, EmbeddingModelConfig

vector_store_config = VectorStoreConfig()
embedding_model_config = EmbeddingModelConfig()

_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PERSIST_DIR = str(_PROJECT_ROOT / vector_store_config.vector_store_name)


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Configure and return a HuggingFace embedding model.

    normalize_embeddings=True makes cosine similarity equivalent to dot product
    on the resulting vectors; this is slightly faster at search time and the recommended
    setting for BGE models.

    The model is downloaded from HuggingFace on first use and cached locally so that
    subsequent calls load from cache.
    """
    return HuggingFaceEmbeddings(
        model_name=embedding_model_config.model_name,
        encode_kwargs={
            "normalize_embeddings": embedding_model_config.normalize_embeddings
        },
    )


def embed_chunks(
    chunks: list[Document],
    model: HuggingFaceEmbeddings,
) -> list[list[float]]:
    """Return one embedding vector per chunk, in input order."""
    return model.embed_documents([chunk.page_content for chunk in chunks])


def build_vector_store(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Embed document chunks and write each vector to a local Chroma store along with its chunk's text and metadata.

    If the store already exists at persist_directory, this function duplicates it; use load_vector_store() instead.
    """
    return Chroma.from_documents(
        documents=chunks,  # placeholder for this example
        embedding=embedding_model,
        persist_directory=persist_directory,
    )


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """Load an existing Chroma store from disk without re-embedding."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def vector_store_exists(persist_directory: str = DEFAULT_PERSIST_DIR) -> bool:
    """Return True if a persisted Chroma store exists at persist_directory."""
    path = Path(persist_directory)
    return path.exists() and any(path.iterdir())
