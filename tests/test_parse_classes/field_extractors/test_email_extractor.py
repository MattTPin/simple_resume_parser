"""test_email_extractor.py
Run tests on EmailExtractor
"""
import pytest

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError

from src.parse_classes.field_extractor.email_extractor import EmailExtractor

from src.test_helpers.mock_resume_generator import (
    ChunkValues,
    ChunkTemplates,
)
from src.test_helpers.dummy_variables.dummy_chunk_lists import(
    MOCK_RESUME_GENERATOR_0,
    MOCK_RESUME_GENERATOR_1,
    MOCK_RESUME_GENERATOR_2,
    MOCK_PERSONS
)

# TODO: PASS IN LLM CLIENT AND MODELS

# ---------------------------------------------------------------------------
# Tests for EmailExtractor
# ---------------------------------------------------------------------------
# Generate simple mock resumes
EXAMPLE_RESUME_SIMPLE_0 = MOCK_RESUME_GENERATOR_0.generate()
EXAMPLE_RESUME_SIMPLE_1 = MOCK_RESUME_GENERATOR_1.generate()
EXAMPLE_RESUME_SIMPLE_2 = MOCK_RESUME_GENERATOR_2.generate()

# ==============================================================
# EDGE CASE RESUME EXAMPLES
# ==============================================================
# Noise / extra text around email
EMAIL_EXAMPLE_RESUME_NOISE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        name="John Doe",
        email="john.doe@example.com"
    ),
    chunk_templates=ChunkTemplates(
        contact_info="Please reach out (email: {email}).\nMy LinkedIn: linkedin.com/in/{linkedin_name}, Website: www.johndoe.com"
    )
).generate()

# Uncommon email formats
EMAIL_EXAMPLE_RESUME_UNCOMMON = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        name="John Doe",
        email="user+tag@subdomain.example.co.uk"
    )
).generate()

# No email present
EMAIL_EXAMPLE_RESUME_NO_EMAIL = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        name="John Doe",
        email=""
    )
).generate()

# Invalid email formats
EMAIL_EXAMPLE_RESUME_INVALID = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        name="John Doe",
        email="john.doe[at]example.com, john.doe_example.com"
    )
).generate()

# Large resume with multiple emails in different chunks
EMAIL_EXAMPLE_RESUME_LARGE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        name="John Doe",
        email="first.email@example.com, second.email@example.org"
    ),
    chunk_templates=ChunkTemplates(
        contact_info=" ".join([f"Line {i}" for i in range(100)]) + " {email}",
        work_experience=" ".join([f"Line {i}" for i in range(50)])
    )
).generate()


class TestEmailExtractor:
    """Unit tests for EmailExtractor using LLM-aware test modes."""

    # -----------------------------------------------------------------
    # Abstract FieldExtractor cannot be instantiated
    # -----------------------------------------------------------------
    def test_cannot_instantiate_directly_field_extractor(self):
        """FieldExtractor is abstract and cannot be instantiated."""
        from src.parse_classes.field_extractor.field_extractor import FieldExtractor
        with pytest.raises(TypeError):
            FieldExtractor(document_chunk_list=[])

    # -----------------------------------------------------------------
    # Unsupported extraction method
    # -----------------------------------------------------------------
    def test_invalid_extraction_method_raises_notimplemented(self):
        """Providing an unsupported extraction method raises NotImplementedError."""
        chunks = [DocumentChunk(chunk_index=1, text="Contact: john.doe@example.com")]
        with pytest.raises(NotImplementedError):
            EmailExtractor(chunks, extraction_method="llm_invalid")

    # -----------------------------------------------------------------
    # Default extraction method
    # -----------------------------------------------------------------
    def test_default_method_is_regex(self):
        """If no method is specified, default should be 'regex'."""
        chunks = [DocumentChunk(chunk_index=1, text="Email: john.doe@example.com")]
        extractor = EmailExtractor(chunks)
        assert extractor.extraction_method == "regex"
        email = extractor.extract()
        assert email == "john.doe@example.com"


@pytest.mark.usefixtures("USE_MOCK_LLM_RESPONSE_SETTING")
class TestEmailExtractorMockLlmModeAware:
    # -----------------------------------------------------------------
    # Simple example resumes
    # -----------------------------------------------------------------
    @pytest.mark.parametrize("extraction_method", EmailExtractor.SUPPORTED_EXTRACTION_METHODS)
    @pytest.mark.parametrize(
        "document_chunk_list,expected",
        [
            (EXAMPLE_RESUME_SIMPLE_0, "john.doe@example.com"),
            (EXAMPLE_RESUME_SIMPLE_1, "c.mendez@company.net"),
            (EXAMPLE_RESUME_SIMPLE_2, "alice.lee@example.co.uk"),
        ],
    )
    def test_simple_resumes(self, extraction_method, document_chunk_list, expected):
        """Each extraction method should correctly extract emails from simple resumes."""
        extractor = EmailExtractor(
            document_chunk_list=document_chunk_list,
            extraction_method=extraction_method,
            llm_dummy_response={"email_address": expected},
        )
        # Optionally call extract() to validate the output
        result = extractor.extract()
        assert result == expected

    # -----------------------------------------------------------------
    # Dynamically generated mock persons
    # -----------------------------------------------------------------
    @pytest.mark.parametrize("method", EmailExtractor.SUPPORTED_EXTRACTION_METHODS)
    @pytest.mark.parametrize("person", MOCK_PERSONS)
    def test_mock_persons(self, method, person):
        """Validate all methods for mock dynamically generated resumes."""
        resume_chunks = MOCK_RESUME_GENERATOR_0.clone(
            chunk_values=ChunkValues(
                name=person["name"],
                email=person["email"],
                linkedin_name=person["linkedin_name"],
            )
        ).generate()
        extractor = EmailExtractor(
            document_chunk_list=resume_chunks,
            extraction_method=method,
            llm_dummy_response={"email_address": person["email"]},
        )
        result = extractor.extract()
        assert result == person["email"]

    # -----------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------
    @pytest.mark.parametrize("method", EmailExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_uncommon_email(self, method):
        """Ensure uncommon but valid email formats are detected."""
        extractor = EmailExtractor(
            document_chunk_list=EMAIL_EXAMPLE_RESUME_UNCOMMON,
            extraction_method=method,
            llm_dummy_response={"email_address": "user+tag@subdomain.example.co.uk"},
        )
        result = extractor.extract()
        assert result == "user+tag@subdomain.example.co.uk"

    @pytest.mark.parametrize("method", EmailExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_no_email_raises_error(self, method):
        """Verify missing emails raise FieldExtractionError."""
        extractor = EmailExtractor(
            document_chunk_list=EMAIL_EXAMPLE_RESUME_NO_EMAIL,
            extraction_method=method,
            llm_dummy_response={"email_address": None},
        )
        with pytest.raises(FieldExtractionError):
            extractor.extract()

    @pytest.mark.parametrize("method", EmailExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_invalid_email_formats(self, method):
        """Ensure invalid email-like strings raise FieldExtractionError."""
        extractor = EmailExtractor(
            document_chunk_list=EMAIL_EXAMPLE_RESUME_INVALID,
            extraction_method=method,
            llm_dummy_response={"email_address": None},
        )
        with pytest.raises(FieldExtractionError):
            extractor.extract()

    @pytest.mark.parametrize("method", EmailExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_large_resume_multiple_emails(self, method):
        """Large resume should return the first valid email detected."""
        extractor = EmailExtractor(
            document_chunk_list=EMAIL_EXAMPLE_RESUME_LARGE,
            extraction_method=method,
            llm_dummy_response={"email_address": "first.email@example.com"},
        )
        result = extractor.extract()
        assert result == "first.email@example.com"