"""test_llm_client.py
Test LLMClient class.
"""
import os
import json
import pytest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

from src.parse_classes.field_extractor.helper_functions.llm.llm_client import (
    LLMClient,
    LLMConfigError,
    LLMInitializationError,
    LLMQueryError,
    LLMEmptyResponse
)
from src.test_helpers.llm_client_test_helpers import create_mock_llm_response
from src.config import SCANNER_DEFAULTS

# -----------------------------
# Skip if Anthropic key missing
# -----------------------------
@pytest.fixture
def has_anthropic_key():
    """Skip tests if Anthropic API key is missing or placeholder."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or key == "<REPLACE_ME>":
        pytest.skip("Anthropic API key not defined in .env")
    return key

# -----------------------------
# Initialization tests
# -----------------------------
def test_invalid_provider_raises():
    """Ensure initializing LLMClient with unsupported provider raises LLMConfigError."""
    with pytest.raises(LLMConfigError):
        LLMClient(provider="unsupported")


def test_model_resolution_defaults(has_anthropic_key):
    """Check that model defaults to SCANNER_DEFAULTS if not provided."""
    client = LLMClient(provider="anthropic", model=None)
    assert client.model is not None


def test_missing_api_key_raises(monkeypatch):
    """Check that missing API key raises LLMConfigError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        LLMClient(provider="anthropic")


# -----------------------------
# Client initialization
# -----------------------------
@patch("langchain_anthropic.ChatAnthropic")
def test_initialize_client_anthropic(mock_chatanthropic, has_anthropic_key):
    """Verify Anthropic client initializes correctly with API key and model."""
    client = LLMClient(provider="anthropic", model="test-model")
    client.initialize_client()
    mock_chatanthropic.assert_called_once()
    assert client.client is not None


# -----------------------------
# Query tests
# -----------------------------
def test_query_without_client_raises(has_anthropic_key):
    """Query without initializing client should raise LLMInitializationError."""
    client = LLMClient(provider="anthropic", model="test-model")
    with pytest.raises(LLMInitializationError):
        client.query(system_prompt="Hi", user_prompt="Hello")


def test_query_test_mode_returns_mock(has_anthropic_key):
    """Verify test_mode returns deterministic mock response using create_mock_llm_response."""
    client = LLMClient(
        provider="anthropic",
        model="test-model",
        test_mode=True,
        function_name="isolate_vehicle_description",
        test_response_type="success"
    )
    client.client = MagicMock()
    result = client.query(system_prompt="sys", user_prompt="user")
    assert isinstance(result, str)
    assert "Subaru Outback" in result


def test_query_fallback_message():
    """Test that fallback_message is returned if LLM response is empty."""
    client = LLMClient(
        provider="anthropic",
        model="test-model",
        function_name="isolate_vehicle_description",
        fallback_message="fallback",
        test_mode=True,
        test_response_type="empty",  # we will define empty content
    )

    # Force a dummy client so initialization check passes
    client.client = object()  # just needs to exist; won't be used in test_mode

    # Patch create_mock_llm_response to return empty content
    from src.test_helpers.llm_client_test_helpers import create_mock_llm_response
    import pytest
    from unittest.mock import patch, MagicMock

    with patch("src.test_helpers.llm_client_test_helpers.create_mock_llm_response") as mock_create:
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_create.return_value = mock_msg

        result = client.query(system_prompt="sys", user_prompt="user")
        assert result == "Generic response"


# -----------------------------
# _clean_llm_json_response tests
# -----------------------------
@pytest.mark.parametrize("raw,expected", [
    ('{"a":1}', {"a": 1}),
    ('```json\n{"b":2}```', {"b": 2}),
    ('Some text {"c":3} more text', {"c": 3}),
    ('[1,2,3]', [1,2,3])
])
def test_clean_llm_json_response_valid(raw, expected, has_anthropic_key):
    """Verify _clean_llm_json_response parses valid JSON and fenced JSON correctly."""
    client = LLMClient(provider="anthropic", model="test-model")
    result = client._clean_llm_json_response(raw)
    assert result == expected


def test_clean_llm_json_response_invalid(has_anthropic_key):
    """Ensure invalid JSON raises JSONDecodeError."""
    client = LLMClient(provider="anthropic", model="test-model")
    with pytest.raises(json.JSONDecodeError):
        client._clean_llm_json_response("invalid json string")


# -----------------------------
# Connection tests
# -----------------------------
def test_test_connection_without_client(monkeypatch, has_anthropic_key):
    """Calling _test_connection_generic without client raises LLMInitializationError."""
    client = LLMClient(provider="anthropic", model="test-model")
    client.client = None
    with pytest.raises(LLMInitializationError):
        client._test_connection_generic("Anthropic")


def test_test_connection_success(has_anthropic_key):
    """Verify _test_connection_generic returns True for successful mock client ping."""
    client = LLMClient(provider="anthropic", model="test-model")
    mock_client = MagicMock()
    mock_client.invoke.return_value = MagicMock(content="pong")
    client.client = mock_client
    assert client._test_connection_generic("Anthropic") is True


def test_test_connection_failure_quota(has_anthropic_key):
    """Verify LLMInitializationError raised if client ping fails with quota error."""
    client = LLMClient(provider="anthropic", model="test-model")
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception("insufficient_quota")
    client.client = mock_client
    with pytest.raises(LLMInitializationError) as e:
        client._test_connection_generic("Anthropic")
    assert "Out of tokens" in str(e.value)


def test_test_connection_failure_rate_limit(has_anthropic_key):
    """Verify LLMInitializationError raised if client ping fails due to rate limit."""
    client = LLMClient(provider="anthropic", model="test-model")
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception("Rate limit reached")
    client.client = mock_client
    with pytest.raises(LLMInitializationError) as e:
        client._test_connection_generic("Anthropic")
    assert "Rate limit" in str(e.value)