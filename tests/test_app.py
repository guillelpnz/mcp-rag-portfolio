"""Test suite — runs offline by mocking the LLM provider.

Covers chunking, the provider factory/error paths, and the FastAPI endpoints
end to end. No network, no Ollama needed in CI.
"""
from __future__ import annotations

import pytest

from app.llm import LLMError
from app.rag import _PROMPT_TEMPLATE, _chunk, _factual_guardrail


# --- chunking -------------------------------------------------------------

def test_chunk_splits_with_overlap():
    text = " ".join(str(i) for i in range(2000))
    chunks = _chunk(text, size=800, overlap=100)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunk_empty_returns_empty():
    assert _chunk("") == []


def test_chunk_short_text_single_chunk():
    assert _chunk("just a few words") == ["just a few words"]


def test_prompt_requires_correction_of_misleading_premises():
    assert "Treat assumptions in the question as unverified" in _PROMPT_TEMPLATE
    assert "Correct any false or misleading premise" in _PROMPT_TEMPLATE


@pytest.mark.parametrize(
    "question",
    [
        "Did Guillermo migrate from RabbitMQ to Kafka?",
        "Was RabbitMQ replaced by Kafka?",
        "Tell me about the RabbitMQ to Kafka migration.",
    ],
)
def test_factual_guardrail_corrects_rabbitmq_kafka_premise(question):
    answer = _factual_guardrail(question)
    assert answer is not None
    assert answer.startswith("No. Kafka already existed")
    assert "Confluent Cloud to Amazon MSK" in answer


def test_factual_guardrail_does_not_intercept_unrelated_kafka_question():
    assert _factual_guardrail("How was Kafka used?") is None


# --- provider factory -----------------------------------------------------

def test_unknown_provider_raises(monkeypatch):
    from app.llm import get_provider

    monkeypatch.setenv("LLM_PROVIDER", "does-not-exist")
    with pytest.raises(LLMError):
        get_provider()


def test_default_provider_is_ollama(monkeypatch):
    from app.llm import get_provider

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_provider().name == "ollama"


# --- API (engine mocked) --------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    def fake_query(self, question, *, k=4):
        if not question.strip():
            raise LLMError("empty")
        return {"answer": f"Mocked answer to: {question}", "sources": ["profile.md"]}

    monkeypatch.setattr(main.RAGEngine, "__init__", lambda self: None)
    monkeypatch.setattr(main.RAGEngine, "query", fake_query)
    monkeypatch.setattr(main.RAGEngine, "ingest", lambda self: 5)
    main._engine = None  # reset lazy singleton between tests
    return TestClient(main.app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_returns_answer(client):
    resp = client.post("/ask", json={"question": "What did he do at Plénitas?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "Plénitas" in body["answer"]
    assert body["sources"] == ["profile.md"]


def test_ask_rejects_empty_question(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # pydantic validation
