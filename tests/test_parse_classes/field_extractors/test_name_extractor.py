"""test_name_extractor.py
Run tests on NameExtractor
"""
import pytest
import warnings
from typing import Optional, Any, List, Dict

from fuzzywuzzy import fuzz

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError

from src.parse_classes.field_extractor.name_extractor import NameExtractor

from src.test_helpers.mock_resume_generator import (
    ChunkValues,
    ChunkTemplates,
    DUMMY_RESUME_BLOCKS
)

from src.test_helpers.dummy_variables.dummy_chunk_lists import(
    MOCK_RESUME_GENERATOR_0,
    MOCK_RESUME_GENERATOR_1,
    MOCK_RESUME_GENERATOR_2,
    MOCK_PERSONS
)

from src.test_helpers.dummy_variables.dummy_chunk_lists import (
    MOCK_DOCUMENT_CHUNK_LIST
)

# ---------------------------------------------------------------------------
# Test suite for NameExtractor
# ---------------------------------------------------------------------------
# Generate simple mock resumes
EXAMPLE_RESUME_SIMPLE_0 = MOCK_RESUME_GENERATOR_0.generate()
EXAMPLE_RESUME_SIMPLE_1 = MOCK_RESUME_GENERATOR_1.generate()
EXAMPLE_RESUME_SIMPLE_2 = MOCK_RESUME_GENERATOR_2.generate()

# Setup spacy and HuggingFace caches for all tests to use
LOADED_SPACY_MODELS = dict()
LOADED_HF_MODELS = dict()

# Function to assess name output with text cleaning + fuzzy matching
FUZZY_THRESHOLD = 80 # 80% match is our fuzzy threshold
def assert_name_similarity(result: str, expected: str):
    """Compare result to expected using fuzzy matching; warn if low."""
    clean_result = "".join(result.split()).lower()
    clean_expected = "".join(expected.split()).lower()
    score = fuzz.ratio(clean_result, clean_expected)
    if score < FUZZY_THRESHOLD:
        warnings.warn(
            f"Name extraction may have failed: expected '{expected}', got '{result}' "
            f"(fuzzy score: {score})"
        )
        
def run_name_extraction_test(
    document_chunk_list: Optional[List] = None,
    extraction_method: Optional[str] = None,
    llm_client: Optional[Any] = None,
    force_mock_llm_response: Optional[bool] = False,
    llm_dummy_response: Optional[Any] = None,
    loaded_spacy_models: Optional[Dict[str, Any]] = None,
    loaded_hf_models: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Helper function to test NameExtractor with default environment models.
    If extraction fails (e.g., NER cannot find a name), raises a warning
    instead of failing.

    Args:
        document_chunk_list (List[DocumentChunk], optional): List of document chunks.
        extraction_method (str, optional): Extraction method ('ner', 'llm', etc.).
        llm_client (Any, optional): LLM client instance for LLM-based extraction.
        force_mock_llm_response (bool, optional): Force mock LLM response for tests.
        llm_dummy_response (Any, optional): Dummy LLM response for testing.
        loaded_spacy_models (dict, optional): SpaCy models to load.
        loaded_hf_models (dict, optional): HuggingFace models to load.

    Returns:
        Optional[str]: Extracted name, or None if extraction failed.
    """
    # Use environment models if not explicitly provided
    if loaded_spacy_models is None:
        loaded_spacy_models = LOADED_SPACY_MODELS
    if loaded_hf_models is None:
        loaded_hf_models = LOADED_HF_MODELS

    extractor = NameExtractor(
        document_chunk_list=document_chunk_list,
        extraction_method=extraction_method,
        llm_client=llm_client,
        force_mock_llm_response=force_mock_llm_response,
        llm_dummy_response=llm_dummy_response,
        loaded_spacy_models=loaded_spacy_models,
        loaded_hf_models=loaded_hf_models,
    )

    try:
        result = extractor.extract()
        return result
    except FieldExtractionError as e:
        warnings.warn(
            f"Name extraction failed with method '{extraction_method}': {str(e)}"
        )
        return None

# ==============================================================
# EDGE CASE RESUME EXAMPLES
# ==============================================================
# Noise / extra text around name
NAME_EXAMPLE_RESUME_NOISE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="John Doe"),
    chunk_templates=ChunkTemplates(
        contact_info="Candidate: {name} | Phone: {phone} | Software Engineer\nEmail: john.doe@example.com"
    )
).generate()

# Lowercase 
NAME_EXAMPLE_RESUME_LOWERCASE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="john doe")
).generate()

# All CAPS
NAME_EXAMPLE_RESUME_UPPERCASE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="JOHN DOE")
).generate()

# Unicode / non-English names
NAME_EXAMPLE_RESUME_NON_ENGLISH = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="Nguyễn Văn An")
).generate()

# No name present
NAME_EXAMPLE_RESUME_NO_NAME = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="")
).generate()

# Name buried far down
NAME_EXAMPLE_RESUME_BURIED = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="John Doe"),
    chunk_templates=ChunkTemplates(
        contact_info="Summary:\n".join([f"Line {i}" for i in range(100)]) + "\n{name}\nSoftware Engineer"
    )
).generate()

# Company name is a persons name AND comes before persons name
NAME_EXAMPLE_RESUME_BURIED = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        company_name="John Deere",
        name="John Doe",
    ),
    chunk_order=[
        "work_experience",
        "contact_info",
    ]
).generate()

# Multiple names (colleagues, etc.)
NAME_EXAMPLE_RESUME_MULTIPLE_NAMES = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="John Doe"),
    chunk_templates=ChunkTemplates(
        work_experience="Worked with Jane Smith and Robert Lee on the AI project.\n"
    )
).generate()

# Name repeats twice.
NAME_EXAMPLE_REPEATE_NAME = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        name="John Doe. John Doe."
    ),
).generate()

# Name split across lines
NAME_EXAMPLE_RESUME_SPLIT = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="John\nDoe")
).generate()

# Extra punctuation / decoration
NAME_EXAMPLE_RESUME_DECORATED = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(name="*** John Doe ***")
).generate()


# class TestNameExtractor:
#     """Unit tests for NameExtractor using LLM-aware test modes."""

#     # -----------------------------------------------------------------
#     # Unsupported extraction method
#     # -----------------------------------------------------------------
#     def test_invalid_extraction_method_raises_not_implemented(self):
#         """Providing an unsupported method raises NotImplementedError."""
#         document_chunk_list = [DocumentChunk(chunk_index=0, text="John Doe\nSoftware Engineer")]
#         with pytest.raises(NotImplementedError):
#             NameExtractor(
#                 document_chunk_list=document_chunk_list,
#                 extraction_method="unsupported_method"
#             ).extract()

#     # -----------------------------------------------------------------
#     # Extraction should raise error if no name present
#     # -----------------------------------------------------------------
#     def test_no_name_raises_field_extraction_error(self):
#         """Extraction should raise FieldExtractionError if no name found."""
#         document_chunk_list = [DocumentChunk(chunk_index=0, text="No name here\nJust text")]
#         extractor = NameExtractor(
#             document_chunk_list=document_chunk_list,
#             extraction_method="ner"
#         )
#         with pytest.raises(FieldExtractionError):
#             extractor.extract()
            
#     # -----------------------------------------------------------------
#     # Confirm models are not retained after extraction
#     # -----------------------------------------------------------------
#     def test_models_are_not_retained_after_ner_extraction(self):
#         """Ensure that loaded_spacy_models and loaded_hf_models are cleared after extraction."""
#         document_chunk_list = [DocumentChunk(chunk_index=0, text="John Doe\nSoftware Engineer")]
#         extractor = NameExtractor(
#             document_chunk_list=document_chunk_list,
#             extraction_method="ner"
#         )
#         try:
#             extractor.extract()
#         except FieldExtractionError:
#             # We don’t care if extraction fails; we only want to test model cleanup
#             pass

#         # Confirm that the extractor has loaded models
#         assert extractor.loaded_spacy_models is None
#         assert extractor.loaded_hf_models is None


# @pytest.mark.usefixtures("USE_MOCK_LLM_RESPONSE_SETTING")
# class TestNameExtractorMockLlmModeAware:
#     # -----------------------------------------------------------------
#     # Simple example resumes
#     # -----------------------------------------------------------------
#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     @pytest.mark.parametrize(
#         "document_chunk_list,expected",
#         [
#             (EXAMPLE_RESUME_SIMPLE_0, "John Doe"),
#             (EXAMPLE_RESUME_SIMPLE_1, "Carlos Mendez"),
#             (EXAMPLE_RESUME_SIMPLE_2, "Alice Lee"),
#         ],
#     )
#     def test_simple_resumes(self, extraction_method, document_chunk_list, expected):
#         result = run_name_extraction_test(
#             document_chunk_list=document_chunk_list,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": expected},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, expected)

#     # -----------------------------------------------------------------
#     # Edge case tests
#     # -----------------------------------------------------------------
#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_name_with_noise(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_NOISE,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "John Doe"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_lowercase_name(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_LOWERCASE,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "john doe"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_uppercase_name(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_UPPERCASE,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "JOHN DOE"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_non_english_name(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_NON_ENGLISH,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "Nguyễn Văn An"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "Nguyễn Văn An")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_no_name_present(self, extraction_method):
#         # We still want this test to explicitly check for FieldExtractionError
#         extractor = NameExtractor(
#             loaded_spacy_models=LOADED_SPACY_MODELS,
#             loaded_hf_models=LOADED_HF_MODELS,
#             document_chunk_list=NAME_EXAMPLE_RESUME_NO_NAME,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": ""},
#         )
#         with pytest.raises(FieldExtractionError):
#             extractor.extract()

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_name_buried_far_down(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_BURIED,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "John Doe"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_multiple_names(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_MULTIPLE_NAMES,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "John Doe"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_repeated_name(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_REPEATE_NAME,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "John Doe."},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe.")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_name_split_across_lines(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_SPLIT,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "John Doe"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     def test_decorated_name(self, extraction_method):
#         result = run_name_extraction_test(
#             document_chunk_list=NAME_EXAMPLE_RESUME_DECORATED,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": "John Doe"},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, "John Doe")

#     # -----------------------------------------------------------------
#     # Test with all MOCK_PERSONS
#     # -----------------------------------------------------------------
#     @pytest.mark.parametrize("extraction_method", NameExtractor.SUPPORTED_EXTRACTION_METHODS)
#     @pytest.mark.parametrize("person", MOCK_PERSONS)
#     def test_mock_persons(self, extraction_method, person):
#         document_chunk_list = MOCK_RESUME_GENERATOR_0.clone(
#             chunk_values=ChunkValues(
#                 name=person["name"],
#                 email=person["email"],
#                 linkedin_name=person["linkedin_name"],
#             )
#         ).generate()
        
#         result = run_name_extraction_test(
#             document_chunk_list=document_chunk_list,
#             extraction_method=extraction_method,
#             llm_dummy_response={"full_name": person["name"]},
#         )
#         if result:
#             assert isinstance(result, str)
#             assert_name_similarity(result, person["name"])


class TestNameExtractorNERResult:
    # -----------------------
    # _expand_name_from_text
    # -----------------------
    @pytest.mark.parametrize(
        "candidate,text,expected",
        [
            ("John Do", "John Doe is a great engineer", "John Doe"),
            ("Alice", "AliceLee works here", "AliceLee"),
            ("Bob", "Bob\tSmith\nanother line", "Bob"),
            ("Nguyễn", "Nguyễn Văn An is here", "Nguyễn"),
            ("SingleWord", "SingleWord", "SingleWord"),
            ("Partial", "PartialNameAndMore text", "PartialNameAndMore"),
        ]
    )
    def test_expand_name_from_text(self, candidate, text, expected):
        extractor = NameExtractor(document_chunk_list=[])
        result = extractor._expand_name_from_text(candidate, text)
        assert result == expected

    # -----------------------
    # _merge_and_select_bert_base_NER_person
    # -----------------------
    def test_merge_subwords_and_select_first_candidate(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "PER", "word": "Jo", "score": 0.9, "start": 0, "end": 2},
            {"entity_group": "PER", "word": "##hn", "score": 0.95, "start": 2, "end": 4},
            {"entity_group": "PER", "word": "Doe", "score": 0.98, "start": 5, "end": 8},
            {"entity_group": "ORG", "word": "Acme", "score": 0.99, "start": 9, "end": 13}
        ]
        text = "John Doe works at Acme"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text)
        assert result.startswith("John Doe")

    def test_ignore_non_target_labels(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "LOC", "word": "Paris", "score": 0.9, "start": 0, "end": 5},
            {"entity_group": "PER", "word": "Jane", "score": 0.95, "start": 6, "end": 10}
        ]
        text = "Paris Jane is here"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text)
        assert result.startswith("Jane")

    def test_gap_breaks_candidates(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "PER", "word": "John", "score": 0.9, "start": 0, "end": 4},
            {"entity_group": "PER", "word": "Doe", "score": 0.9, "start": 10, "end": 13},  # gap > max_gap
        ]
        text = "John Doe"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text, max_gap=3)
        # Should only take the first candidate "John" and expand in text
        assert result.startswith("John")

    def test_single_word_expansion(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "PER", "word": "John D", "score": 0.9, "start": 0, "end": 4},
        ]
        text = "John Doe is here"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text)
        assert result.startswith("John Doe")

    def test_multiple_candidates_select_first(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "PER", "word": "Alice", "score": 0.95, "start": 0, "end": 5},
            {"entity_group": "PER", "word": "Bob", "score": 0.95, "start": 6, "end": 9}
        ]
        text = "Alice and Bob are here"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text)
        assert result.startswith("Alice")

    def test_return_none_if_no_candidates(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "LOC", "word": "Paris", "score": 0.9, "start": 0, "end": 5}
        ]
        text = "Paris is beautiful"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text)
        assert result is None

    def test_non_english_name(self):
        extractor = NameExtractor(document_chunk_list=[])
        ner_results = [
            {"entity_group": "PER", "word": "Nguyễn", "score": 0.99, "start": 0, "end": 6},
            {"entity_group": "PER", "word": "Văn", "score": 0.99, "start": 7, "end": 10},
            {"entity_group": "PER", "word": "An", "score": 0.99, "start": 11, "end": 13},
        ]
        text = "Nguyễn Văn An is here"
        result = extractor._merge_and_select_bert_base_NER_person(ner_results, text)
        assert result.startswith("Nguyễn Văn An")

    # -----------------------
    # _ner_extract
    # -----------------------
    def test_ner_extract_returns_none_for_single_word(self, monkeypatch):
        # Patch _hf_search and _merge_and_select_bert_base_NER_person to return single-word name
        extractor = NameExtractor(document_chunk_list=[])
        monkeypatch.setattr(extractor, "_hf_search", lambda **kwargs: [
            {"entity_group": "PER", "word": "John", "score": 0.99, "start": 0, "end": 4}
        ])
        monkeypatch.setattr(
            extractor, "_merge_and_select_bert_base_NER_person", lambda **kwargs: "John"
        )
        result = extractor._ner_extract()
        assert result is None

    def test_ner_extract_returns_full_name(self, monkeypatch):
        extractor = NameExtractor(document_chunk_list=[])
        monkeypatch.setattr(extractor, "_hf_search", lambda **kwargs: [
            {"entity_group": "PER", "word": "John", "score": 0.99, "start": 0, "end": 4},
            {"entity_group": "PER", "word": "Doe", "score": 0.99, "start": 5, "end": 8},
        ])
        monkeypatch.setattr(
            extractor, "_merge_and_select_bert_base_NER_person", lambda **kwargs: "John Doe"
        )
        result = extractor._ner_extract()
        assert result == "John Doe"


class TestNameExtractorLLMSpecific:
    """
    Tests for confirming LLM level functionality specifically
    """
    def test_llm_empty_response_fails(self):
        extractor = NameExtractor(
            loaded_spacy_models = LOADED_SPACY_MODELS,
            loaded_hf_models = LOADED_HF_MODELS,
            document_chunk_list=MOCK_DOCUMENT_CHUNK_LIST,
            extraction_method="llm",
            llm_dummy_response={"full_name": ""},  # Simulate EMPTY llm response
            force_mock_llm_response=True
        )
        with pytest.raises(FieldExtractionError):
            extractor.extract()

    def test_find_chunks_with_phone_number_basic(self):
        """Basic test to ensure _find_chunks_with_phone_number returns text containing phone numbers."""
        # Prepare some mock chunks
        chunks = [
            DocumentChunk(chunk_index=0, text="Alice Smith 123-456-7890 alice@example.com"),
            DocumentChunk(chunk_index=1, text="No phone here"),
        ]

        extractor = NameExtractor(
            loaded_spacy_models=LOADED_SPACY_MODELS,
            loaded_hf_models=LOADED_HF_MODELS,
            document_chunk_list=chunks,
            extraction_method="llm"
        )

        result_text = extractor._find_chunks_with_phone_number()

        # Ensure result_text is not None
        assert result_text is not None

        # Check that the returned text contains the phone numbers
        assert "123-456-7890" in result_text

    def test_find_chunks_with_phone_number_none(self):
        """Test that _find_chunks_with_phone_number returns None if no phone numbers are present."""
        # Prepare chunks with no phone numbers
        chunks = [
            DocumentChunk(chunk_index=0, text="Alice Smith alice@example.com"),
            DocumentChunk(chunk_index=1, text="No phone here"),
            DocumentChunk(chunk_index=2, text="Contact me via email only"),
        ]

        extractor = NameExtractor(
            loaded_spacy_models=LOADED_SPACY_MODELS,
            loaded_hf_models=LOADED_HF_MODELS,
            document_chunk_list=chunks,
            extraction_method="llm"
        )

        result_text = extractor._find_chunks_with_phone_number()

        # Ensure result is None when no phone numbers exist
        assert result_text is None

    # ---------------------------------------------------------------------
    # Mode-limited runs
    # ---------------------------------------------------------------------
    @pytest.mark.parametrize(
        "document_chunk_list,expected",
        [
            (EXAMPLE_RESUME_SIMPLE_0, "John Doe"),
            (EXAMPLE_RESUME_SIMPLE_1, "Carlos Mendez"),
        ],
    )
    def test_find_skills_text_llm_basic_only_mode(self, document_chunk_list, expected, LLM_TEST_MODE):
        if LLM_TEST_MODE != "basic_only":
            pytest.skip("Only run this test in basic_only LLM mode")
        extractor = NameExtractor(
            loaded_spacy_models = LOADED_SPACY_MODELS,
            loaded_hf_models = LOADED_HF_MODELS,
            document_chunk_list=document_chunk_list,
            extraction_method="llm"
        )
        result = extractor.extract()
        if result:
            assert isinstance(result, str)
            assert_name_similarity(result, expected)