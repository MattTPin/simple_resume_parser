"""test_extractor_map.py
Tests for extractor_map helper functions
"""

import pytest
from typing import List, Dict, Optional

from src.exceptions import ExtractorMapConfigError, FieldExtractionConfigError
from src.models import DocumentChunk

from src.test_helpers.dummy_variables.dummy_chunk_lists import MOCK_RESUME_GENERATOR_0

from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.test_helpers.dummy_classes import DummyExtractor
from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor

from src.parse_classes.field_extractor.helper_functions.llm.llm_client import LLMClient

from src.parse_classes.resume_extractor.helpers.extractor_map import (
    verify_extractor_map,
    build_default_extractor_map,
    unify_extractor_map_model_references,
)

# ---------- Helper for building a minimal valid extractor_map ----------
# Generate simple mock resumes
EXAMPLE_RESUME_SIMPLE_0 = MOCK_RESUME_GENERATOR_0.generate()

def make_valid_extractor_map() -> Dict[str, List[FieldExtractor]]:
    """Return a minimal valid extractor_map using DummyExtractor instances."""
    return {
        "name": [DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)],
        "email": [DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)],
        "skills": [DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)],
    }


# -------------------------
# Tests for verify_extractor_map
# -------------------------
class TestVerifyExtractorMap:
    """Tests for verify_extractor_map function."""

    def test_valid_map(self):
        """Should not raise on a valid extractor_map."""
        extractor_map = make_valid_extractor_map()
        verify_extractor_map(extractor_map)  # No exception should be raised

    def test_not_a_dict(self):
        """Passing a non-dict should raise TypeError."""
        with pytest.raises(TypeError):
            verify_extractor_map(["not", "a", "dict"])

    def test_field_key_not_string(self):
        """Non-string field keys should raise TypeError."""
        extractor_map = make_valid_extractor_map()
        extractor_map[123] = extractor_map.pop("name")  # replace key with int
        with pytest.raises(TypeError) as e:
            verify_extractor_map(extractor_map)
        assert "Field names in extractor_map must be strings" in str(e.value)

    def test_value_not_list(self):
        """Field values must be lists."""
        extractor_map = make_valid_extractor_map()
        extractor_map["name"] = "not a list"
        with pytest.raises(TypeError) as e:
            verify_extractor_map(extractor_map)
        assert "Value for field 'name' must be a list" in str(e.value)

    def test_item_not_field_extractor(self):
        """Items in extractor lists must be FieldExtractor instances."""
        extractor_map = make_valid_extractor_map()
        extractor_map["skills"] = ["not an extractor"]
        with pytest.raises(TypeError) as e:
            verify_extractor_map(extractor_map)
        assert "All items in extractor list for field 'skills' must be FieldExtractor instances" in str(e.value)


# -------------------------
# Tests for build_default_extractor_map
# -------------------------
class TestBuildDefaultExtractorMap:
    """Tests for build_default_extractor_map function with new extractor dict structure."""

    def test_returns_dict_with_expected_keys(self):
        """Should return a dictionary containing 'name', 'email', and 'skills' keys."""
        extractor_map = build_default_extractor_map(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        assert isinstance(extractor_map, dict)
        for key in ["name", "email", "skills"]:
            assert key in extractor_map
            # Each key should be a non-empty list
            assert isinstance(extractor_map[key], list)
            assert len(extractor_map[key]) > 0

    def test_values_are_lists_of_extractors(self):
        """Each value in the extractor map should be a list of FieldExtractor instances."""
        extractor_map = build_default_extractor_map(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        for key, extractors in extractor_map.items():
            assert isinstance(extractors, list)
            for ex in extractors:
                assert isinstance(ex, FieldExtractor)

    def test_extractors_have_correct_class_and_method(self):
        """Check that each key maps to the expected extractor class and extraction_method is set correctly."""
        extractor_map = build_default_extractor_map(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        
        # Name has multiple extractors
        name_extractors = extractor_map["name"]
        assert any(isinstance(ex, NameExtractor) for ex in name_extractors)
        # Check extraction_method is either specified or default
        for ex in name_extractors:
            assert isinstance(ex, NameExtractor)
            assert ex.extraction_method in NameExtractor.SUPPORTED_EXTRACTION_METHODS

        # Email
        email_extractors = extractor_map["email"]
        assert all(isinstance(ex, EmailExtractor) for ex in email_extractors)
        # Skills
        skills_extractors = extractor_map["skills"]
        assert all(isinstance(ex, SkillsExtractor) for ex in skills_extractors)

    def test_extractors_have_defaults_set(self):
        """Verify that extractors correctly store the passed document chunks, models, and LLM client."""
        loaded_spacy = {"en_core_web_sm": "fake_spacy_model"}
        loaded_hf = {"yashpwr/resume-ner-bert-v2": "fake_hf_pipeline"}
        fake_llm_client = LLMClient()

        extractor_map = build_default_extractor_map(
            document_chunk_list=EXAMPLE_RESUME_SIMPLE_0,
            loaded_spacy_models=loaded_spacy,
            loaded_hf_models=loaded_hf,
            llm_client=fake_llm_client
        )

        for extractors in extractor_map.values():
            for ex in extractors:
                assert ex.document_chunk_list == EXAMPLE_RESUME_SIMPLE_0
                assert ex.loaded_spacy_models == loaded_spacy
                assert ex.loaded_hf_models == loaded_hf
                assert ex.llm_client == fake_llm_client

    def test_handles_none_inputs(self):
        """Function should handle None inputs and still return valid FieldExtractor instances."""
        extractor_map = build_default_extractor_map(
            document_chunk_list=None,
            loaded_spacy_models=None,
            loaded_hf_models=None,
            llm_client=None
        )
        for extractors in extractor_map.values():
            for ex in extractors:
                assert isinstance(ex, FieldExtractor)
                assert ex.document_chunk_list == []  # <-- check empty list instead of None
                assert ex.loaded_spacy_models is None
                assert ex.loaded_hf_models is None
                assert hasattr(ex, "llm_client")


# -------------------------
# Tests for unify_extractor_map_model_references
# -------------------------
class TestOverrideExtractorMapModelReferences:
    """Tests for unify_extractor_map_model_references function."""

    def test_document_chunk_list_is_overridden(self):
        """document_chunk_list should always be overridden in all extractors."""
        extractor_map = make_valid_extractor_map()
        new_chunks = EXAMPLE_RESUME_SIMPLE_0
        unify_extractor_map_model_references(
            extractor_map, document_chunk_list=new_chunks
        )
        for extractors in extractor_map.values():
            for ex in extractors:
                assert ex.document_chunk_list == new_chunks

    def test_llm_client_only_set_if_none(self):
        """llm_client should only be set for extractors that do not already have one."""
        extractor_map = make_valid_extractor_map()
        fake_llm = LLMClient()

        extractor_map["name"][0].llm_client = "existing_client"

        unify_extractor_map_model_references(
            extractor_map, document_chunk_list=None, llm_client=fake_llm
        )

        assert extractor_map["name"][0].llm_client == "existing_client"
        assert extractor_map["email"][0].llm_client == fake_llm
        assert extractor_map["skills"][0].llm_client == fake_llm

    def test_spacy_models_set_only_if_empty(self):
        """loaded_spacy_models should only override extractors that do not already have models set."""
        extractor_map = make_valid_extractor_map()
        spacy_models = {"en_core_web_sm": "fake_model"}

        extractor_map["name"][0].loaded_spacy_models = {"already": "set"}

        unify_extractor_map_model_references(
            extractor_map,
            document_chunk_list=None,
            loaded_spacy_models=spacy_models
        )

        assert extractor_map["name"][0].loaded_spacy_models == {"already": "set"}
        assert extractor_map["email"][0].loaded_spacy_models == spacy_models
        assert extractor_map["skills"][0].loaded_spacy_models == spacy_models

    def test_hf_models_set_only_if_empty(self):
        """loaded_hf_models should only override extractors that do not already have models set."""
        extractor_map = make_valid_extractor_map()
        hf_models = {"yashpwr/resume-ner-bert-v2": "fake_pipeline"}

        extractor_map["email"][0].loaded_hf_models = {"already": "set"}

        unify_extractor_map_model_references(
            extractor_map,
            document_chunk_list=None,
            loaded_hf_models=hf_models
        )

        assert extractor_map["email"][0].loaded_hf_models == {"already": "set"}
        assert extractor_map["name"][0].loaded_hf_models == hf_models
        assert extractor_map["skills"][0].loaded_hf_models == hf_models

    def test_returns_same_map(self):
        """Since the function mutates in place, the extractor_map should reflect updates correctly."""
        extractor_map = make_valid_extractor_map()
        new_chunks = EXAMPLE_RESUME_SIMPLE_0
        unify_extractor_map_model_references(
            extractor_map, document_chunk_list=new_chunks
        )
        for extractors in extractor_map.values():
            for ex in extractors:
                assert ex.document_chunk_list == new_chunks