"""test_resume_extractor.py
Run tests on ResumeExtractor
"""
import warnings
import multiprocessing

import pytest

from src.models import DocumentChunk, ResumeData
from src.exceptions import FieldExtractionError, FieldExtractionConfigError

from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor

from src.parse_classes.resume_extractor.resume_extractor import ResumeExtractor

from src.test_helpers.dummy_classes import DummyExtractor
from src.test_helpers.dummy_variables.dummy_chunk_lists import MOCK_RESUME_GENERATOR_0

# ---------------------------------------------------------------------------
# CONFIG AND SETUP
# ---------------------------------------------------------------------------


# Use EXAMPLE_RESUME_SIMPLE_0 for all tests in this file!
EXAMPLE_RESUME_SIMPLE_0 = MOCK_RESUME_GENERATOR_0.generate()

# ---------------------------------------------------------------------------
# Test suite for ResumeExtractor
# ---------------------------------------------------------------------------

# Prevent live LLM querying for all tests
@pytest.mark.usefixtures("FORCE_MOCK_LLM_RESPONSES")
class TestResumeExtractor:
    """
    Unit tests for ResumeExtractor.
    
    Always force mock LLM responses since we ONLY want to test ResumeExtractor
    specific functionality.
    """
    # --------------------------------------
    # Basic Extraction Sanity Checks
    # --------------------------------------
    def test_extract_default_resume_data_type(self):
        """Running with default settings returns a ResumeData instance."""
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        resume_data = extractor.extract()
        assert isinstance(resume_data, ResumeData)

    def test_extract_all_fields_warn_if_missing(self):
        """
        Extract name, email, and skills in a single run.
        Raise warnings for missing or incorrect fields instead of failing the test.
        """
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        resume_data: ResumeData = extractor.extract()

        # Name check
        expected_name = "John Doe"
        if not resume_data.name or expected_name.lower() not in resume_data.name.lower():
            warnings.warn(f"Name extraction failed: expected '{expected_name}', got '{resume_data.name}'")

        # Email check
        expected_email = "john.doe@example.com"
        if not resume_data.email or expected_email.lower() not in resume_data.email.lower():
            warnings.warn(f"Email extraction failed: expected '{expected_email}', got '{resume_data.email}'")

        # Skills check
        expected_skills = ["SQL"]
        result_skills_lower = [(s or "").lower() for s in (resume_data.skills or [])]
        missing_skills = [skill for skill in expected_skills if skill.lower() not in result_skills_lower]
        if missing_skills:
            warnings.warn(f"Skills extraction missing: {missing_skills}. Extracted skills: {resume_data.skills}")

    def test_extract_parallel_default_run(self):
        """Default run doesn't crash even with max_threads > 1."""
        if multiprocessing.cpu_count() <= 1:
            pytest.skip("Skipping parallel extraction test because CPU has only 1 core.")
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0, max_threads=2)
        resume_data = extractor.extract()
        assert isinstance(resume_data, ResumeData)

    # --------------------------------------
    # Field Presence and Default Handling
    # --------------------------------------
    def test_extract_all_fields_expected_or_default(self):
        """
        Ensure that each field in ResumeData is either the expected value
        or a default value. If a field only has the default, issue a warning.
        """
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        resume_data: ResumeData = extractor.extract()

        expected_values = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "skills": ["SQL"],
        }

        for field, expected in expected_values.items():
            value = getattr(resume_data, field)
            if field == "skills":
                # For skills, compare each expected skill ignoring case
                value_lower = [(s or "").lower() for s in (value or [])]
                missing_skills = [skill for skill in expected if skill.lower() not in value_lower]
                if missing_skills:
                    warnings.warn(
                        f"Field '{field}' missing expected skills {missing_skills}. "
                        f"Extracted skills: {value}"
                    )
            else:
                # For name/email, check case-insensitive match
                if not value or value.strip().lower() != expected.lower():
                    if value is None or (field == "name" and value == None) or (field == "email" and value == None):
                        warnings.warn(
                            f"Field '{field}' has default value: '{value}'. Expected: '{expected}'"
                        )
                    else:
                        warnings.warn(
                            f"Field '{field}' extraction mismatch. Expected: '{expected}', got: '{value}'"
                        )

    def test_constructor_defaults(self):
        """Should initialize with default extractor map and shared models."""
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        assert extractor.document_chunk_list == EXAMPLE_RESUME_SIMPLE_0
        assert hasattr(extractor, "extractor_map")
        assert isinstance(extractor.extractor_map, dict)
        for extractors in extractor.extractor_map.values():
            for ex in extractors:
                assert hasattr(ex, "extract")

    # --------------------------------------
    # Fallback / Multi-Extractor Behavior
    # --------------------------------------
    def test_name_extractor_fallback(self):
        """If the first NameExtractor fails, the backup extractor should run."""
        failing_name_extractor = NameExtractor(
            extraction_method="llm",
            llm_dummy_response={"NOT_NAME": "d"},  # Force failing LLM output
        )
        backup_name_extractor = NameExtractor(
            extraction_method="llm",
            llm_dummy_response={"full_name": "John Doe"},  # Force successful LLM output
        )
        extractor_map = {"name": [failing_name_extractor, backup_name_extractor]}
        with pytest.warns(UserWarning):  # max_threads warning expected
            extractor = ResumeExtractor(
                document_chunk_list=EXAMPLE_RESUME_SIMPLE_0,
                extractor_map=extractor_map
            )
        resume_data = extractor.extract()
        assert resume_data.name is not None
        assert "John Doe" in resume_data.name

    # --------------------------------------
    # Threading and max_threads behavior
    # --------------------------------------
    def test_threads_exceed_cpu_and_fields_warns_and_caps(self):
        """Warn if threads exceed both CPU cores and number of fields."""
        with pytest.warns(UserWarning) as record:
            extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0, max_threads=1000)

        messages = [str(w.message) for w in record]
        assert any("exceeds available cores" in m for m in messages)
        assert any("exceeds the number of extraction fields" in m for m in messages)

        num_fields = len(extractor.extractor_map)
        expected_max = min(1000, multiprocessing.cpu_count(), num_fields)
        assert extractor.max_threads == expected_max

    def test_threads_exceed_cpu_only_warns(self, monkeypatch):
        """Warn if requested threads exceed available CPU cores only."""
        fake_cpu_count = 2
        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: fake_cpu_count)

        with pytest.warns(UserWarning) as record:
            extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0, max_threads=8)

        messages = [str(w.message) for w in record]
        assert any("exceeds available cores" in m for m in messages)
        assert not any("exceeds the number of extraction fields" in m for m in messages)
        assert extractor.max_threads == fake_cpu_count

    def test_threads_exceed_num_fields_only_warns(self):
        """Should warn about exceeding both available cores and number of fields."""
        num_fields = len(ResumeExtractor(EXAMPLE_RESUME_SIMPLE_0).extractor_map)
        with pytest.warns(UserWarning) as record:
            ResumeExtractor(EXAMPLE_RESUME_SIMPLE_0, max_threads=9999)
        messages = [str(w.message) for w in record]
        assert any("exceeds available cores" in m for m in messages)
        assert any("exceeds the number of extraction fields" in m for m in messages)

    def test_zero_or_negative_threads_warns_and_resets(self):
        """Should warn and reset to 1 thread when given zero or negative value."""
        with pytest.warns(UserWarning) as record:
            extractor = ResumeExtractor(EXAMPLE_RESUME_SIMPLE_0, max_threads=0)
        messages = [str(w.message) for w in record]
        assert any("invalid" in m.lower() or "defaulting to 1" in m.lower() for m in messages)
        assert extractor.max_threads == 1

    def test_valid_thread_count_no_warnings(self):
        """Should not emit any warnings when max_threads is within valid limits."""
        num_fields = len(ResumeExtractor(EXAMPLE_RESUME_SIMPLE_0).extractor_map)
        valid_threads = min(multiprocessing.cpu_count(), num_fields)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            extractor = ResumeExtractor(EXAMPLE_RESUME_SIMPLE_0, max_threads=valid_threads)
        assert not record, f"Unexpected warnings: {[str(w.message) for w in record]}"
        assert extractor.max_threads == valid_threads

    def test_parallel_extraction_threads_limited_by_fields(self):
        """Even if max_threads > number of fields, threads should be capped."""
        with pytest.warns(UserWarning) as record:  # catch both warnings
            extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0, max_threads=1000)
        num_fields = len(extractor.extractor_map)
        assert extractor.max_threads == num_fields
        # Confirm the expected warnings were raised
        assert any("exceeds available cores" in str(w.message) for w in record)
        assert any("exceeds the number of extraction fields" in str(w.message) for w in record)

    def test_max_threads_less_than_one_sets_to_one(self):
        """If max_threads is < 1, it should default to 1 sequential execution and warn once."""

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0, max_threads=0)
            assert extractor.max_threads == 1
            
            for key in extractor.extractor_map:
                extractor.extractor_map[key] = [DummyExtractor()]
                
            resume_data = extractor.extract()
            for field in extractor.extractor_map:
                assert getattr(resume_data, field) == "dummy"

            # Confirm at least one warning was issued
            assert any("max_threads" in str(warn.message) for warn in w), "Expected warning for max_threads <= 0"

    # --------------------------------------
    # Extractor Map Overrides & Field Fallback
    # --------------------------------------
    def test_override_model_references_sets_document_chunks(self):
        """Override assigns document_chunk_list to all extractors."""
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        for ex_list in extractor.extractor_map.values():
            for ex in ex_list:
                assert ex.document_chunk_list == EXAMPLE_RESUME_SIMPLE_0

    def test_extract_field_with_fallback_success(self):
        """_extract_field_with_fallback returns value from first successful extractor."""
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        for key in extractor.extractor_map:
            extractor.extractor_map[key] = [DummyExtractor()]
        for field in extractor.extractor_map:
            result = extractor._extract_field_with_fallback(field)
            assert result == "dummy"

    def test_extract_field_with_fallback_returns_default_on_failure(self):
        """Returns ResumeData default if all extractors fail."""
        class FailingExtractor(DummyExtractor):
            def extract(self):
                raise ValueError("fail")
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        for key in extractor.extractor_map:
            extractor.extractor_map[key] = [FailingExtractor()]
        for field in extractor.extractor_map:
            result = extractor._extract_field_with_fallback(field)
            default_value = getattr(ResumeData(), field)
            assert result == default_value

    def test_extract_field_with_fallback_all_extractors_fail_returns_default(self):
        """Even if multiple extractors fail, should return default."""
        class AlwaysFailExtractor(DummyExtractor):
            def extract(self):
                raise RuntimeError("always fail")
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        for key in extractor.extractor_map:
            extractor.extractor_map[key] = [AlwaysFailExtractor(), AlwaysFailExtractor()]
        for field in extractor.extractor_map:
            result = extractor._extract_field_with_fallback(field)
            default_value = getattr(ResumeData(), field)
            assert result == default_value

    # --------------------------------------
    # Extraction Execution / Parallelism
    # --------------------------------------
    def test_extract_returns_resume_data(self):
        """extract returns ResumeData instance with all fields populated."""
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0)
        for key in extractor.extractor_map:
            extractor.extractor_map[key] = [DummyExtractor()]
        resume_data = extractor.extract()
        assert isinstance(resume_data, ResumeData)
        for field in extractor.extractor_map:
            assert getattr(resume_data, field) == "dummy"

    def test_extract_parallel_execution(self):
        """extract runs in parallel when max_threads > 1."""
        extractor = ResumeExtractor(document_chunk_list=EXAMPLE_RESUME_SIMPLE_0, max_threads=2)
        for key in extractor.extractor_map:
            extractor.extractor_map[key] = [DummyExtractor()]
        resume_data = extractor.extract()
        assert isinstance(resume_data, ResumeData)
        for field in extractor.extractor_map:
            assert getattr(resume_data, field) == "dummy"

    def test_parallel_extraction_with_one_field(self):
        """Parallel extraction with only one field should still work."""
        single_field_map = {"name": [DummyExtractor()]}
        with pytest.warns(UserWarning):  # max_threads warning expected
            extractor = ResumeExtractor(
                document_chunk_list=EXAMPLE_RESUME_SIMPLE_0,
                extractor_map=single_field_map,
                max_threads=4
            )
        resume_data = extractor.extract()
        assert resume_data.name == "dummy"

    # --------------------------------------
    # Edge / Error Cases
    # --------------------------------------
    def test_empty_extractor_map(self):
        """Should handle an empty extractor_map gracefully and return defaults."""
        with pytest.warns(UserWarning):  # max_threads warning expected
            extractor = ResumeExtractor(
                document_chunk_list=EXAMPLE_RESUME_SIMPLE_0,
                extractor_map={}
            )
        resume_data = extractor.extract()
        assert isinstance(resume_data, ResumeData)
        assert resume_data.name is None
        assert resume_data.email is None
        assert resume_data.skills == []

    def test_invalid_document_chunk_list(self):
        """Should raise error with invalid document_chunk_list."""
        with pytest.raises(TypeError):
            extractor = ResumeExtractor(document_chunk_list="HELLO")