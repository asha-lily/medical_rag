"""
uv run --with pytest pytest tests/test_api.py -v
"""

from types import SimpleNamespace
from unittest.mock import Mock

import ollama
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_pipeline.api.main import app
from rag_pipeline.config import GenerationConfig

QUESTION = {"question": "Can I take ibuprofen with paracetamol?"}


@pytest.fixture
def rag_chain() -> Mock:
    """A stand-in for the RAG chain, so tests don't need Ollama or Chroma."""
    return Mock()


@pytest.fixture
def client(rag_chain):
    """Produce a TestClient with the fake chain in app.state.rag_chain"""
    app.state.rag_chain = rag_chain
    yield TestClient(app)
    del app.state.rag_chain


def fake_ollama_client(models=(), error=None) -> Mock:
    """A replacement for the ollama.Client class whose list() reports
    `models`, or raises `error`."""
    instance = Mock()
    if error:
        instance.list.side_effect = error
    else:
        instance.list.return_value = SimpleNamespace(
            models=[SimpleNamespace(model=name) for name in models]
        )
    return Mock(return_value=instance)


def test_health_returns_ok(client: TestClient):
    # Given / when
    response = client.get("/health")
    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client: TestClient, rag_chain: Mock):
    # Given
    rag_chain.invoke.return_value = {
        "question": "Can I take ibuprofen?",
        "answer": "Yes, but...",
        "docs": [
            Document(
                page_content="...",
                metadata={
                    "source": "ibuprofen_PIL.pdf",
                    "page": 2,
                    "chunk_id": "ibuprofen_PIL.pdf#3.0",
                },
            )
        ],
    }

    # When
    response = client.post("/query", json={"question": "Can I take ibuprofen?"})

    # Then
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Yes, but...",
        "sources": [
            {
                "source": "ibuprofen_PIL.pdf",
                "page": 2,
                "chunk_id": "ibuprofen_PIL.pdf#3.0",
            }
        ],
    }
    rag_chain.invoke.assert_called_once_with("Can I take ibuprofen?")


def test_query_strips_whitespace_from_question(client: TestClient, rag_chain: Mock):
    # Given
    rag_chain.invoke.return_value = {"question": "", "answer": "", "docs": []}
    # When
    client.post("/query", json={"question": "  Can I take ibuprofen?  "})
    # Then
    rag_chain.invoke.assert_called_once_with("Can I take ibuprofen?")


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "   "}, {"text": "hello"}, {"question": 123}],
)
def test_query_rejects_invalid_request(client: TestClient, rag_chain: Mock, body):
    # Given / when
    response = client.post("/query", json=body)
    # Then
    assert response.status_code == 422
    rag_chain.invoke.assert_not_called()


def test_query_returns_503_when_ollama_unreachable(client: TestClient, rag_chain: Mock):
    # Given
    rag_chain.invoke.side_effect = ConnectionError()
    # When
    response = client.post("/query", json=QUESTION)
    # Then
    assert response.status_code == 503
    assert response.json() == {"detail": "LLM backend unavailable. Try again later."}


def test_query_returns_502_when_ollama_returns_error(
    client: TestClient, rag_chain: Mock
):
    # Given
    rag_chain.invoke.side_effect = ollama.ResponseError("model not found", 404)
    # When
    response = client.post("/query", json=QUESTION)
    # Then
    assert response.status_code == 502


def test_ready_when_model_is_pulled(client: TestClient, monkeypatch):
    # Given
    model = f"{GenerationConfig().model_name}:latest"
    monkeypatch.setattr(
        "rag_pipeline.api.main.ollama.Client",
        fake_ollama_client(models=[model, "qwen3:8b"]),
    )
    # When
    response = client.get("/ready")
    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_ollama_unreachable(client: TestClient, monkeypatch):
    # Given / When
    monkeypatch.setattr(
        "rag_pipeline.api.main.ollama.Client",
        fake_ollama_client(error=ConnectionError()),
    )
    # Then
    assert client.get("/ready").status_code == 503


def test_ready_returns_503_when_model_not_pulled(client: TestClient, monkeypatch):
    # Given / When
    monkeypatch.setattr(
        "rag_pipeline.api.main.ollama.Client",
        fake_ollama_client(models=["qwen3:8b"]),
    )
    # Then
    assert client.get("/ready").status_code == 503
