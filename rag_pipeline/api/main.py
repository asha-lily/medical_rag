"""
uv run uvicorn rag_pipeline.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from langchain_core.documents import Document

from rag_pipeline.api.schemas import QueryRequest, QueryResponse, Source
from rag_pipeline.generation import create_rag_chain
from rag_pipeline.retrieval import create_retriever
from rag_pipeline.vector_store import (
    create_embedding_model,
    load_vector_store,
    vector_store_exists,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models and build the RAG chain once, before serving requests."""
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first."
        )
    log.info("Loading embedding model and vector store...")
    embedding_model = create_embedding_model()
    store = load_vector_store(embedding_model)
    retriever = create_retriever(store, embedding_model)
    app.state.rag_chain = create_rag_chain(retriever)
    log.info("RAG chain ready.")
    yield


app = FastAPI(lifespan=lifespan)


def _to_source(doc: Document) -> Source:
    return Source(
        source=doc.metadata["source"],
        page=doc.metadata.get("page"),
        chunk_id=doc.metadata["chunk_id"],
    )


@app.get("/health")
def get_health():
    return {"status": "ok"}


@app.post("/query")
def post_query(query_request: QueryRequest, request: Request) -> QueryResponse:
    try:
        result = request.app.state.rag_chain.invoke(query_request.question)
    except ConnectionError as e:
        log.error("ConnectionError: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to the LLM. Check your model name and server.",
        ) from e

    return QueryResponse(
        answer=result["answer"],
        sources=[_to_source(doc) for doc in result["docs"]],
    )
