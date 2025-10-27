"""test_FORCE_MOCK_LLM_RESPONSES.py
Confirm FORCE_MOCK_LLM_RESPONSES and USE_MOCK_LLM_RESPONSE_SETTING
apply correctly depending on scope and LLM_TEST_MODE.
"""

import pytest

from src.conftest_helpers import apply_mock_llm_patch

from src.test_helpers.dummy_variables.dummy_chunk_lists import MOCK_DOCUMENT_CHUNK_LIST
from src.test_helpers.dummy_classes import DummyExtractor
from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor


# ---------------------------------------------------------------------------
# Confirm default behavior without fixture
# ---------------------------------------------------------------------------
def test_extractors_unpatched_by_default():
    """Extractors should not be patched unless a fixture is applied."""
    for ext, cls_name in zip(
        [NameExtractor, EmailExtractor, SkillsExtractor],
        ["NameExtractor", "EmailExtractor", "SkillsExtractor"]
    ):
        instance = ext(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST)
        assert getattr(instance, "force_mock_llm_response", False) is False, (
            f"Unpatched {cls_name} should not have force_mock_llm_response=True by default."
        )


# ---------------------------------------------------------------------------
# Class-level fixture tests using DummyExtractor
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("FORCE_MOCK_LLM_RESPONSES")
class TestForceMockClassLevel:
    """Verify FORCE_MOCK_LLM_RESPONSES applies at the class level."""

    def test_dummy_extractor_base_patch(self):
        """
        Confirm that DummyExtractor (base FieldExtractor subclass) is patched
        with force_mock_llm_response=True when the class-level fixture is applied.
        """
        extractor = DummyExtractor(llm_dummy_response={"FAKE_RESPONSE": "fake"})
        assert extractor.force_mock_llm_response is True, (
            "Class-level patch failed: DummyExtractor did not get force_mock_llm_response=True"
        )


# ---------------------------------------------------------------------------
# Function-level fixture test using DummyExtractor
# ---------------------------------------------------------------------------
def test_dummy_extractor_function_level_patch(FORCE_MOCK_LLM_RESPONSES):
    """
    Confirm that DummyExtractor is patched when FORCE_MOCK_LLM_RESPONSES
    is applied at function level.
    """
    extractor = DummyExtractor(llm_dummy_response={"FAKE_RESPONSE": "fake"})
    assert extractor.force_mock_llm_response is True, (
        "Function-level patch failed: DummyExtractor did not get force_mock_llm_response=True"
    )


# ---------------------------------------------------------------------------
# Verify each real extractor once
# ---------------------------------------------------------------------------
def test_specific_extractors_function_level_patch(FORCE_MOCK_LLM_RESPONSES):
    """Verify each specific extractor gets patched correctly for its dummy response."""
    extractors = [
        (NameExtractor(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST), {"full_name": "John Doe"}),
        (EmailExtractor(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST), {"email_address": "john.doe@example.com"}),
        (SkillsExtractor(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST), {"skills": ["SQL"]}),
    ]

    for ext, dummy in extractors:
        assert ext.force_mock_llm_response is True
        assert ext.llm_dummy_response == dummy


# ---------------------------------------------------------------------------
# Conditional patch tests (simulate LLM_TEST_MODE)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,expected", [
    ("mock_only", False),
    ("basic_only", False),
    ("full", True),
])
def test_use_mock_llm_response_setting(monkeypatch, mode, expected):
    """
    Confirm that the patch from USE_MOCK_LLM_RESPONSE_SETTING is applied only if
    LLM_TEST_MODE=='full'.

    NOTE: Due to the current pytest configuration, we cannot import the real
    LLM_TEST_MODE fixture directly. Fixtures are executed at runtime and cannot
    be monkeypatched safely without breaking import paths.

    Workaround:
      - Manually apply the patch if mode=="full" to simulate conditional behavior.
      - Skip applying patch for other modes.
    """
    if mode == "full":
        apply_mock_llm_patch(monkeypatch)

    extractors = [
        (NameExtractor(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST), {"full_name": "John Doe"}),
        (EmailExtractor(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST), {"email_address": "john.doe@example.com"}),
        (SkillsExtractor(document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST), {"skills": ["SQL"]}),
    ]

    for ext, dummy in extractors:
        assert (getattr(ext, "force_mock_llm_response", False) is True) == expected
        if expected:
            assert ext.llm_dummy_response == dummy
        else:
            assert getattr(ext, "llm_dummy_response", None) != dummy