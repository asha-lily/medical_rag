"""
uv run uvicorn rag_pipeline.api.main:app --reload
"""

import ollama
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
from rag_pipeline.config import GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
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
        log.error("Could not reach Ollama. Is `ollama serve` running? %s", e)
        raise HTTPException(
            status_code=503,
            detail="LLM backend unavailable. Try again later.",
        ) from e
    except ollama.ResponseError as e:
        log.error("Ollama returned an error (status %s): %s", e.status_code, e.error)
        raise HTTPException(
            status_code=502,
            detail="LLM backend returned an error.",
        ) from e

    return QueryResponse(
        answer=result["answer"],
        sources=[_to_source(doc) for doc in result["docs"]],
    )


@app.get("/ready")
def check_if_ready():
    try:
        ollama_response = ollama.Client().list()
    except ConnectionError as e:
        log.error("Could not reach Ollama. Is `ollama serve` running? %s", e)
        raise HTTPException(
            status_code=503,
            detail="LLM backend unavailable. Try again later.",
        ) from e

    ollama_available_models = {m.model for m in ollama_response.models}
    model_name = GenerationConfig().model_name
    if ":" not in model_name:
        model_name = f"{model_name}:latest"

    if model_name not in ollama_available_models:
        log.error(
            "Model %r not pulled in Ollama. Run `ollama pull %s`.",
            model_name,
            model_name,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Model not found.",
        )

    return {"status": "ready"}
