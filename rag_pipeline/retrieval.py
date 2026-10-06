from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline.config import EmbeddingModelConfig, RetrievalConfig

embedding_config = EmbeddingModelConfig()
retrieval_config = RetrievalConfig()

DEFAULT_K = retrieval_config.k_chunks_to_retrieve


def retrieve(
    query: str,
    store: Chroma,
    embedding_model: HuggingFaceEmbeddings,
    k: int = DEFAULT_K,
) -> list[Document]:
    """Embed the query and return the k most relevant chunks."""
    query_vector = embedding_model.embed_query(
        embedding_config.query_instruction + query
    )
    return store.similarity_search_by_vector(query_vector, k=k)


def retrieve_with_scores(
    query: str,
    store: Chroma,
    embedding_model: HuggingFaceEmbeddings,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    """
    Return the k most relevant chunks along with their relevance scores.
    Scores are normalised cosine similarities in [0, 1].
    """
    query_vector = embedding_model.embed_query(
        embedding_config.query_instruction + query
    )
    return store.similarity_search_by_vector_with_relevance_scores(query_vector, k=k)


class DocumentRetriever(BaseRetriever):
    """LangChain-compatible retriever.

    This is the expected format for use in LCEL chains. It has a `.invoke()` method.
    """

    store: Chroma
    embedding_model: HuggingFaceEmbeddings
    k: int = DEFAULT_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return retrieve(query, self.store, self.embedding_model, self.k)


def create_retriever(
    store: Chroma,
    embedding_model: HuggingFaceEmbeddings,
    k: int = DEFAULT_K,
) -> DocumentRetriever:
    """Return a DocumentRetriever ready for use in a chain or standalone."""
    return DocumentRetriever(store=store, embedding_model=embedding_model, k=k)
