from fastapi.testclient import TestClient

from config import API_KEY
from main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "HalluciGuard Backend is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_verify_with_valid_api_key():
    response = client.post(
        "/verify",
        headers={"x-api-key": API_KEY},
        json={"query": "What is artificial intelligence?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "verification_status" in data
    assert "confidence" in data
    assert "hallucination_score" in data
    assert "citations" in data


def test_verify_without_api_key():
    response = client.post(
        "/verify",
        json={"query": "What is artificial intelligence?"},
    )

    assert response.status_code == 401


def test_verify_with_wrong_api_key():
    response = client.post(
        "/verify",
        headers={"x-api-key": "wrong-key"},
        json={"query": "What is artificial intelligence?"},
    )

    assert response.status_code == 401


def test_verify_with_empty_query():
    response = client.post(
        "/verify",
        headers={"x-api-key": API_KEY},
        json={"query": ""},
    )

    assert response.status_code == 422


def test_verify_with_whitespace_query():
    response = client.post(
        "/verify",
        headers={"x-api-key": API_KEY},
        json={"query": "     "},
    )

    assert response.status_code == 422


def test_verify_with_surrounding_whitespace():
    response = client.post(
        "/verify",
        headers={"x-api-key": API_KEY},
        json={
            "query": "   What is artificial intelligence?   "
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Test response for: What is artificial intelligence?"
    )


def test_verify_when_workflow_fails(monkeypatch):
    def mock_workflow_failure(query):
        raise RuntimeError("Internal workflow failure")

    monkeypatch.setattr(
        "main.run_halluciguard",
        mock_workflow_failure,
    )

    response = client.post(
        "/verify",
        headers={"x-api-key": API_KEY},
        json={"query": "What is artificial intelligence?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Unable to process the query"
    }


def test_verify_when_workflow_returns_invalid_response(monkeypatch):
    def mock_invalid_workflow(query):
        return {
            "answer": "Invalid test response",
            "verification_status": "mock",
            "confidence": 1.5,
            "hallucination_score": 0.0,
            "citations": [],
        }

    monkeypatch.setattr(
        "main.run_halluciguard",
        mock_invalid_workflow,
    )

    response = client.post(
        "/verify",
        headers={"x-api-key": API_KEY},
        json={"query": "What is artificial intelligence?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Unable to process the query"
    }