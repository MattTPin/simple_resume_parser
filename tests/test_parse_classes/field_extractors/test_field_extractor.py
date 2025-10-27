"""test_field_extractor.py
Test the abstractor FieldExtractor class
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError, FieldExtractionConfigError

from src.parse_classes.field_extractor.field_extractor import FieldExtractor, EXTRACTION_METHODS

from src.test_helpers.dummy_classes import DummyExtractor
from src.test_helpers.dummy_variables.dummy_chunk_lists import(
    MOCK_RESUME_GENERATOR_0,
    MOCK_RESUME_GENERATOR_1,
    MOCK_RESUME_GENERATOR_2,
)

# ---------------------------------------------------------------------------
# Tests for FieldExtractor
# ---------------------------------------------------------------------------
# Generate simple mock resumes
EXAMPLE_RESUME_SIMPLE_0 = MOCK_RESUME_GENERATOR_0.generate()
EXAMPLE_RESUME_SIMPLE_1 = MOCK_RESUME_GENERATOR_1.generate()
EXAMPLE_RESUME_SIMPLE_2 = MOCK_RESUME_GENERATOR_2.generate()

SPACY_TEST_MODEL_NAME = "en_core_web_sm"
HF_TEST_MODEL_NAME = "dslim/bert-base-NER"


# Sample controlled document chunks
TEST_HF_CHUNKS = [
    DocumentChunk(chunk_index=1, text="John Doe works at Google."),
    DocumentChunk(chunk_index=2, text="Jane Smith lives in New York."),
]

class TestFieldExtractorInit:
    """Comprehensive tests for FieldExtractor base class and its error handling."""
    # ----------------------
    # Initialization & config tests
    # ----------------------
    def test_cannot_instantiate_directly(self):
        """Ensure abstract FieldExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FieldExtractor([])

    def test_default_extraction_method_is_used(self):
        """Default extraction_method should be 'regex' when not specified."""
        chunks = [DocumentChunk(0, "text")]
        extractor = DummyExtractor(chunks)
        assert extractor.extraction_method == "regex"

    def test_invalid_extraction_method_raises(self):
        """Passing an unsupported extraction_method should raise NotImplementedError."""
        chunks = [DocumentChunk(0, "text")]
        with pytest.raises(NotImplementedError):
            DummyExtractor(chunks, extraction_method="invalid_method")

    def test_valid_extraction_method(self):
        """Ensure that a valid extraction method is accepted and set correctly."""
        extractor = DummyExtractor(extraction_method="ner")
        assert extractor.extraction_method == "ner"


    def test_default_extraction_method_used_when_none_provided(self):
        """Ensure that if no extraction method is provided, the default is used."""
        extractor = DummyExtractor(extraction_method=None)
        assert extractor.extraction_method == "regex"


    def test_unsupported_extraction_method_raises(self):
        """Ensure that providing an unsupported extraction method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            DummyExtractor(extraction_method="ml")


    def test_missing_supported_methods_raises(self, monkeypatch):
        """Ensure that a subclass with no SUPPORTED_EXTRACTION_METHODS defined raises ValueError."""
        # Patch DummyExtractor to have empty SUPPORTED_EXTRACTION_METHODS
        class BrokenExtractor(DummyExtractor):
            SUPPORTED_EXTRACTION_METHODS = []
            DEFAULT_EXTRACTION_METHOD = None

        with pytest.raises(ValueError):
            BrokenExtractor(extraction_method=None)

class TestFieldExtractorSpacy:
    # ----------------------
    # Spacy search
    # ----------------------
    def test_ner_search_returns_entity(self):
        """Test that _spacy_search returns a matching entity for ner_label."""
        extractor = DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        result = extractor._spacy_search(model_name=SPACY_TEST_MODEL_NAME, ner_label="PERSON")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_token_attr_search_returns_token(self):
        """Test that _spacy_search returns a token when token_attr is used."""
        extractor = DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        # 'like_email' is a standard spaCy token attribute
        result = extractor._spacy_search(model_name=SPACY_TEST_MODEL_NAME, token_attr="like_email")
        assert isinstance(result, str)
        assert "@" in result  # token text should contain an email

    def test_missing_ner_and_token_attr_raises(self):
        """Test that providing neither ner_label nor token_attr raises FieldExtractionConfigError."""
        extractor = DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        with pytest.raises(FieldExtractionConfigError):
            extractor._spacy_search(model_name=SPACY_TEST_MODEL_NAME)

    def test_no_match_raises_field_extraction_error(self):
        """Test that if no NER entity or token matches, FieldExtractionError is raised."""
        extractor = DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        with pytest.raises(FieldExtractionError):
            extractor._spacy_search(model_name=SPACY_TEST_MODEL_NAME, ner_label="NON_EXISTENT_LABEL")

    def test_search_batch_size_none_searches_all(self):
        """Test that setting search_batch_size=None searches all chunks at once."""
        extractor = DummyExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        result = extractor._spacy_search(model_name=SPACY_TEST_MODEL_NAME, ner_label="PERSON", search_batch_size=None)
        assert isinstance(result, str)
        assert len(result) > 0

class TestFieldExtractorHuggingFace:
    # ----------------------
    # HuggingFace (hf) search
    # ----------------------
    def test_return_ner_results(self):
        """Test returning raw NER results."""
        extractor = DummyExtractor(document_chunk_list=TEST_HF_CHUNKS)

        fake_results = [{"entity_group": "PER", "word": "John Doe"}]

        with patch("src.parse_classes.field_extractor.helper_functions.ml.hf_loader.load_hf_model") as mock_loader:
            mock_nlp = MagicMock(return_value=fake_results)
            mock_loader.return_value = mock_nlp

            results = extractor._hf_search(model_name=HF_TEST_MODEL_NAME, return_target="ner_results")
            assert results == fake_results

    def test_return_target_label(self):
        """Test returning first entity matching target_label."""
        extractor = DummyExtractor(document_chunk_list=TEST_HF_CHUNKS)

        fake_results = [
            {"entity_group": "PER", "word": "John"},
            {"entity_group": "PER", "word": "Doe"},
            {"entity_group": "ORG", "word": "Google"}
        ]

        with patch("src.parse_classes.field_extractor.helper_functions.ml.hf_loader.load_hf_model") as mock_loader:
            mock_nlp = MagicMock(return_value=fake_results)
            mock_loader.return_value = mock_nlp

            result = extractor._hf_search(
                model_name=HF_TEST_MODEL_NAME, target_label="PER", return_target="target_label"
            )
            assert result == "John Doe"

    def test_missing_target_label_raises_value_error(self):
        """Check ValueError if target_label is None with return_target='target_label'."""
        extractor = DummyExtractor(document_chunk_list=TEST_HF_CHUNKS)
        with pytest.raises(ValueError):
            extractor._hf_search(model_name=HF_TEST_MODEL_NAME, return_target="target_label", target_label=None)

    def test_no_match_raises_field_extraction_error(self):
        """Check FieldExtractionError if no entities match target_label."""
        extractor = DummyExtractor(document_chunk_list=TEST_HF_CHUNKS)

        fake_results = [{"entity_group": "ORG", "word": "Google"}]  # no PER

        with patch("src.parse_classes.field_extractor.helper_functions.ml.hf_loader.load_hf_model") as mock_loader:
            mock_nlp = MagicMock(return_value=fake_results)
            mock_loader.return_value = mock_nlp

            with pytest.raises(FieldExtractionError):
                extractor._hf_search(
                    model_name=HF_TEST_MODEL_NAME, target_label="PER", return_target="target_label"
                )

    def test_search_batch_size_none_processes_all(self):
        """Check search_batch_size=None processes all chunks as one batch."""
        extractor = DummyExtractor(document_chunk_list=TEST_HF_CHUNKS)

        fake_results = [{"entity_group": "PER", "word": "John"}]

        with patch("src.parse_classes.field_extractor.helper_functions.ml.hf_loader.load_hf_model") as mock_loader:
            mock_nlp = MagicMock(return_value=fake_results)
            mock_loader.return_value = mock_nlp

            result = extractor._hf_search(
                model_name=HF_TEST_MODEL_NAME, target_label="PER", return_target="target_label", search_batch_size=None
            )
            assert result == "John"

class TestFieldExtractorRegex:
    # ----------------------
    # Regex extraction
    # ----------------------
    def test_regex_extract_term_single_and_batch(self):
        """Verify regex extraction finds matches correctly in both single and batch mode."""
        chunks = [
            DocumentChunk(0, "no match"),
            DocumentChunk(1, "match: test@example.com"),
            DocumentChunk(2, "another")
        ]
        extractor = DummyExtractor(chunks)
        result = extractor._regex_extract_term(r"test@example\.com")
        assert result == "test@example.com"

        # Test batch processing
        result = extractor._regex_extract_term(r"another", search_batch_size=2)
        assert result == "another"

    def test_regex_extract_term_not_found_raises(self):
        """Ensure FieldExtractionError is raised when no regex match is found."""
        chunks = [DocumentChunk(0, "abc"), DocumentChunk(1, "def")]
        extractor = DummyExtractor(chunks)
        with pytest.raises(FieldExtractionError):
            extractor._regex_extract_term(r"xyz")

    # ----------------------
    # _get_chunks_with_matching_regex
    # ----------------------
    def test_get_chunks_with_matching_regex_multiple_patterns_and_buffers(self):
        """Ensure regex matching returns correct chunk ranges with multiple patterns and buffers."""
        chunks = [
            DocumentChunk(0, "ignore"),
            DocumentChunk(1, "skills: Python"),
            DocumentChunk(2, "Java"),
            DocumentChunk(3, "C++"),
        ]
        extractor = DummyExtractor(chunks)
        result = extractor._get_chunks_with_matching_regex(
            chunks,
            patterns=[r"skills", r"C\+\+"],
            backward_buffer=1,
            forward_buffer=1
        )
        assert result[0].text == "ignore"
        assert result[-1].text == "C++"

    def test_get_chunks_with_matching_regex_no_match_raises(self):
        """Ensure FieldExtractionError is raised when no regex patterns match."""
        chunks = [DocumentChunk(0, "abc"), DocumentChunk(1, "def")]
        extractor = DummyExtractor(chunks)
        with pytest.raises(FieldExtractionError):
            extractor._get_chunks_with_matching_regex(chunks, "xyz")

    # ----------------------
    # Decorator _requires_document_chunks
    # ----------------------
    def test_requires_document_chunks_decorator_raises_if_missing(self):
        """Ensure _requires_document_chunks raises FieldExtractionConfigError when list is missing."""
        chunks = [DocumentChunk(0, "text")]
        extractor = DummyExtractor(chunks)
        extractor.document_chunk_list = None
        with pytest.raises(FieldExtractionConfigError):
            extractor._regex_extract_term(r".*")

    def test_extract_with_empty_chunks(self):
        """Ensure FieldExtractionError is raised when regex finds no non-empty text."""
        chunks = [DocumentChunk(0, ""), DocumentChunk(1, " ")]
        extractor = DummyExtractor(chunks)
        with pytest.raises(FieldExtractionError):
            extractor._regex_extract_term(r"\S+")

class TestFieldLLMHandling:
    # ----------------------
    # LLM handling
    # ----------------------
    def test_query_llm_returns_dummy_in_test_mode(self):
        """Ensure _query_llm returns dummy response when force_mock_llm_response=True."""
        chunks = [DocumentChunk(0, "text")]
        dummy_response = "dummy"
        extractor = DummyExtractor(
            chunks,
            extraction_method="llm",
            force_mock_llm_response=True,
            llm_dummy_response=dummy_response
        )
        result = extractor._query_llm("system", "user")
        assert result == dummy_response
            
    def test_query_llm_raises_if_no_dummy_with_mock_mode(self):
        """If force_mock_llm_response=True but no dummy provided, raises FieldExtractionConfigError."""
        chunks = [DocumentChunk(0, "text")]
        extractor = DummyExtractor(
            chunks,
            extraction_method="llm",
            force_mock_llm_response=True,
            llm_dummy_response=None
        )
        with pytest.raises(FieldExtractionConfigError) as exc:
            extractor._query_llm("system", "user")
        assert "llm_dummy_response" in str(exc.value)


    def test_llm_not_initialized_when_force_mock(monkeypatch):
        extractor = DummyExtractor(extraction_method="llm", force_mock_llm_response=True)
        extractor._initiate_llm()
        # Should not initialize client when mock is forced
        assert extractor.llm_client is None

    # -----------------------------------------------------------------
    # LLM dict verification tests
    # -----------------------------------------------------------------
    def test_verify_llm_dict_output_success(self):
        """Should pass when dict has expected field with correct type."""
        llm_response = {"skills": ["python", "sql"]}
        # Should not raise
        extractor = DummyExtractor()
        extractor._verify_llm_dict_output(
            llm_response=llm_response,
            field_name="skills",
            expected_type=list,
            user_prompt="Extract skills"
        )

    def test_verify_llm_dict_output_not_a_dict(self):
        """Raise FieldExtractionError if LLM response is not a dict."""
        llm_response = ["python", "sql"]  # invalid type
        with pytest.raises(FieldExtractionError) as exc:
            extractor = DummyExtractor()
            extractor._verify_llm_dict_output(
                llm_response=llm_response,
                field_name="skills",
                expected_type=list,
                user_prompt="Extract skills"
            )
        assert "LLM did not return a valid JSON" in str(exc.value)

    def test_verify_llm_dict_output_missing_field(self):
        """Raise FieldExtractionError if expected field is missing in dict."""
        llm_response = {"wrong_field": ["python"]}
        with pytest.raises(FieldExtractionError) as exc:
            extractor = DummyExtractor()
            extractor._verify_llm_dict_output(
                llm_response=llm_response,
                field_name="skills",
                expected_type=list,
                user_prompt="Extract skills"
            )
        assert "LLM JSON missing expected field `skills`" in str(exc.value)

    def test_verify_llm_dict_output_wrong_type(self):
        """Raise FieldExtractionError if expected field exists but has wrong type."""
        llm_response = {"skills": "python"}  # should be list
        with pytest.raises(FieldExtractionError) as exc:
            extractor = DummyExtractor()
            extractor._verify_llm_dict_output(
                llm_response=llm_response,
                field_name="skills",
                expected_type=list,
                user_prompt="Extract skills"
            )
        assert "LLM field `skills` has wrong type" in str(exc.value)

    # ----------------------
    # Edge case: empty chunk text
    # ----------------------


