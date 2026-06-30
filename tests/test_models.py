from unittest.mock import MagicMock, patch

import pytest

from src.models import query_ollama


def _mock_response(content: str, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {"message": {"content": content}}
    mock.raise_for_status.return_value = None
    return mock


def _mock_error_response(status_code: int = 500) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return mock


@patch("src.models.requests.post")
def test_query_ollama_success(mock_post):
    mock_post.return_value = _mock_response("The capital of France is Paris.")

    result = query_ollama("What is the capital of France?")

    assert result["text"] == "The capital of France is Paris."
    assert result["error"] is None
    assert isinstance(result["latency"], float)
    assert result["latency"] >= 0


@patch("src.models.requests.post")
def test_query_ollama_sends_correct_payload(mock_post):
    mock_post.return_value = _mock_response("ok")

    query_ollama("Hello", model="mistral:7b")

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "mistral:7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )


@patch("src.models.requests.post")
def test_query_ollama_http_error_returns_error_dict(mock_post):
    mock_post.return_value = _mock_error_response(500)

    result = query_ollama("test prompt")

    assert result["text"] is None
    assert result["latency"] is None
    assert "HTTP 500" in result["error"]


@patch("src.models.requests.post")
def test_query_ollama_connection_error_returns_error_dict(mock_post):
    mock_post.side_effect = ConnectionError("Connection refused")

    result = query_ollama("test prompt")

    assert result["text"] is None
    assert result["latency"] is None
    assert result["error"] is not None


@patch("src.models.requests.post")
def test_query_ollama_default_model(mock_post):
    mock_post.return_value = _mock_response("ok")

    query_ollama("hi")

    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["model"] == "llama3.2:3b"


def test_ollama_in_models_dict():
    from src.models import MODELS
    assert "ollama" in MODELS
    assert MODELS["ollama"] is query_ollama
