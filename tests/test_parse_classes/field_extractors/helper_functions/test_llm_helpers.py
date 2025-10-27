"""test_llm_helpers.py
Test files in llm_helpers.py
"""

import pytest
from unittest.mock import MagicMock, patch

from src.parse_classes.field_extractor.helper_functions.llm.llm_helpers import initialize_llm_if_needed
from src.parse_classes.field_extractor.helper_functions.llm.llm_client import LLMClient

class DummyExtractor:
    def __init__(self, extraction_method=None):
        self.extraction_method = extraction_method


def test_existing_llm_client_returned():
    client = MagicMock(spec=LLMClient)
    # Should return the same client if provided
    result = initialize_llm_if_needed(llm_client=client)
    assert result is client

def test_existing_llm_client_wrong_type_raises():
    with pytest.raises(TypeError):
        initialize_llm_if_needed(llm_client="not-a-client")

def test_extraction_method_not_llm_returns_none():
    result = initialize_llm_if_needed(extraction_method="other")
    assert result is None

def test_no_llm_needed_in_extraction_class_list_returns_none():
    extractors = [DummyExtractor("other"), DummyExtractor("other2")]
    result = initialize_llm_if_needed(extraction_class_list=extractors)
    assert result is None

@patch("src.parse_classes.field_extractor.helper_functions.llm.llm_helpers.LLMClient")
def test_llm_initialized_when_needed(mock_llm_class):
    mock_instance = MagicMock(spec=LLMClient)
    mock_llm_class.return_value = mock_instance

    # Either extraction_method is "llm"
    result = initialize_llm_if_needed(extraction_method="llm")
    mock_llm_class.assert_called_once()
    mock_instance.initialize_client.assert_called_once()
    assert result is mock_instance

@patch("src.parse_classes.field_extractor.helper_functions.llm.llm_helpers.LLMClient")
def test_llm_initialized_if_any_extractor_requires_llm(mock_llm_class):
    mock_instance = MagicMock(spec=LLMClient)
    mock_llm_class.return_value = mock_instance

    extractors = [DummyExtractor("other"), DummyExtractor("llm")]
    result = initialize_llm_if_needed(extraction_class_list=extractors)
    mock_llm_class.assert_called_once()
    mock_instance.initialize_client.assert_called_once()
    assert result is mock_instance
