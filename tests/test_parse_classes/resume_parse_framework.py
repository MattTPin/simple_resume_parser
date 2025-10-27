"""test_parse_resume_framework.py
Run tests on ResumeParserFramework
"""
import pytest
import os
from pathlib import Path
import warnings

from docx import Document
from reportlab.pdfgen import canvas

from src.models import DocumentChunk, ResumeData
from src.exceptions import (
    FieldExtractionError,
    FieldExtractionConfigError,
    FileNotSupportedError
)

from src.parse_classes.file_parser.pdf_parser import PDFParser
from src.parse_classes.file_parser.word_document_parser import WordDocumentParser

from src.parse_classes.field_extractor.helper_functions.llm.llm_client import LLMClient
from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor

from src.parse_classes.resume_extractor.resume_extractor import ResumeExtractor

from src.parse_classes.resume_parse_framework import ResumeParserFramework

from src.test_helpers.file_parsing import (
    list_files,
    TEST_FILE_PATHS,
    assert_chunks_are_readable,
)

# Import entire files (for patching in tests)
from src.parse_classes import resume_parse_framework
from src.parse_classes.field_extractor.helper_functions.llm import llm_helpers
from src.parse_classes.field_extractor.helper_functions.llm import llm_client
from src.parse_classes.field_extractor.helper_functions.ml import spacy_loader
from src.parse_classes.field_extractor.helper_functions.ml import hf_loader


# ---------------------------------------------------------------------
# SETUP TEST VARIABLES
# ---------------------------------------------------------------------
TEST_TEXT = (
    "John Doe  123-456-7890   john.doe@example.com   linkedIn/john_doe23"
    "Skills: SQL, Python, Matlab"
)
TEST_DOCUMENT_CHUNK_LIST = [
    DocumentChunk(
        chunk_index=1,
        text=TEST_TEXT,
    )
]
TEST_RESUME_DATA = ResumeData(
    name = "John Doe",
    email = "john.doe@example.com",
    skills = ["Google Analytics", "Data Cleaning", "SQL"]
)


# ---------------------------------------------------------------------
# FIXTURES FOR FAKE FILES
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_pdf_file(tmp_path_factory) -> Path:
    """Create a reusable fake PDF file containing TEST_TEXT."""
    tmp_dir = tmp_path_factory.mktemp("fake_files")
    pdf_path = tmp_dir / "fake_resume.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, TEST_TEXT)
    c.save()
    return pdf_path


@pytest.fixture(scope="module")
def fake_docx_file(tmp_path_factory) -> Path:
    """Create a reusable fake DOCX file containing TEST_TEXT."""
    tmp_dir = tmp_path_factory.mktemp("fake_files")
    docx_path = tmp_dir / "fake_resume.docx"
    doc = Document()
    doc.add_paragraph(TEST_TEXT)
    doc.save(docx_path)
    return docx_path


@pytest.fixture(scope="module")
def fake_txt_file(tmp_path_factory) -> Path:
    """Create a reusable fake TXT file containing TEST_TEXT."""
    tmp_dir = tmp_path_factory.mktemp("fake_files")
    txt_path = tmp_dir / "fake_resume.txt"
    txt_path.write_text(TEST_TEXT)
    return txt_path


# ---------------------------------------------------------------------
# Integration-level tests using actual resume files
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "resume_file, file_type",
    [
        (TEST_FILE_PATHS['pdf_dir'] / "professional-ms-word-resume-template.pdf", "PDF"),
        (TEST_FILE_PATHS['docx_dir'] / "professional-ms-word-resume-template.docx", "DOCX"),
    ]
)
def test_parse_real_resume_real_llm(resume_file, file_type, LLM_TEST_MODE):
    """Parse a real PDF or DOCX resume using LLM if LLM_TEST_MODE is 'basic_only' or 'full'."""

    if LLM_TEST_MODE not in ("basic_only", "full"):
        pytest.skip(f"Skipping real LLM test because LLM_TEST_MODE={LLM_TEST_MODE}")

    expected_skills = [
        "Pet footwear", "Secchi disks", "Partial Budget", "Sensory Evaluations",
        "Water Flow Meters", "Canva", "CoolClimate"
    ]

    # Use normal extractors without forced LLM mock responses
    framework = ResumeParserFramework(
        extractor_map={
            "name": [NameExtractor()],
            "email": [EmailExtractor()],
            "skills": [SkillsExtractor()],
        }
    )

    result: ResumeData = framework.parse_resume(resume_file)
    assert isinstance(result, ResumeData)

    expected_name = "Arlo Bishop"
    if (result.name or "").strip().lower() != expected_name.lower():
        warnings.warn(
            f"{file_type} extraction failed for name: expected '{expected_name}', got '{result.name}'"
        )

    expected_email = "ronna.jackson99@gmail.com"
    if (result.email or "").strip().lower() != expected_email.lower():
        warnings.warn(
            f"{file_type} extraction failed for email: expected '{expected_email}', got '{result.email}'"
        )

    result_skills_lower = [(s or "").strip().lower() for s in (result.skills or [])]
    missing_skills = [
        skill for skill in expected_skills if skill.lower() not in result_skills_lower
    ]
    if missing_skills:
        warnings.warn(
            f"{file_type} extraction missing skills: {missing_skills}. Extracted skills: {result.skills}"
        )


# NOTE: From this point on ALWAYS Prevent live LLM querying for all tests in this file
@pytest.mark.usefixtures("FORCE_MOCK_LLM_RESPONSES")
@pytest.mark.parametrize(
    "resume_file, file_type",
    [
        (TEST_FILE_PATHS['pdf_dir'] / "professional-ms-word-resume-template.pdf", "PDF"),
        (TEST_FILE_PATHS['docx_dir'] / "professional-ms-word-resume-template.docx", "DOCX"),
    ]
)
def test_parse_real_resume_fake_llm(resume_file, file_type):
    """
    Parse a real PDF or DOCX resume and verify expected fields. ALWAYS uses a fake
    LLM response for simplicity / saving on costs.
    """

    expected_skills = [
        "Pet footwear", "Secchi disks", "Partial Budget", "Sensory Evaluations",
        "Water Flow Meters", "Canva", "CoolClimate"
    ]

    framework = ResumeParserFramework(
        extractor_map={
            "name": [NameExtractor()],
            "email": [EmailExtractor()],
            "skills": [SkillsExtractor(
                llm_dummy_response={"skills": expected_skills}
            )],
        }
    )

    result: ResumeData = framework.parse_resume(resume_file)
    assert isinstance(result, ResumeData)

    expected_name = "Arlo Bishop"
    if (result.name or "").strip().lower() != expected_name.lower():
        warnings.warn(
            f"{file_type} extraction failed for name: expected '{expected_name}', got '{result.name}'"
        )

    expected_email = "ronna.jackson99@gmail.com"
    if (result.email or "").strip().lower() != expected_email.lower():
        warnings.warn(
            f"{file_type} extraction failed for email: expected '{expected_email}', got '{result.email}'"
        )

    result_skills_lower = [(s or "").strip().lower() for s in (result.skills or [])]
    missing_skills = [
        skill for skill in expected_skills if skill.lower() not in result_skills_lower
    ]
    if missing_skills:
        warnings.warn(
            f"{file_type} extraction missing skills: {missing_skills}. Extracted skills: {result.skills}"
        )


# ---------------------------------------------------------------------
# Standard functionality
# ---------------------------------------------------------------------
@pytest.mark.usefixtures("FORCE_MOCK_LLM_RESPONSES")
class TestResumeParserFrameworkBasic:
    """Confirm ResumeParserFramework runs and behaves correctly in normal use."""

    def test_init_runs_without_crashing(self):
        """Ensure initialization with default params does not raise errors."""
        framework = ResumeParserFramework()
        assert isinstance(framework, ResumeParserFramework)

    def test_pdf_parser_selected(self, mocker, fake_pdf_file):
        """
        Verify that PDFParser is selected when given a .pdf file.
        """
        mock_parse = mocker.patch.object(PDFParser, "parse", return_value=[])
        rpf = ResumeParserFramework()

        result = rpf._parse_file(str(fake_pdf_file))

        mock_parse.assert_called_once()
        assert isinstance(result, list)

    def test_docx_parser_selected(self, mocker, fake_docx_file):
        """
        Verify that WordDocumentParser is selected when given a .docx file.
        """
        mock_parse = mocker.patch.object(WordDocumentParser, "parse", return_value=[])
        rpf = ResumeParserFramework()

        result = rpf._parse_file(str(fake_docx_file))

        mock_parse.assert_called_once()
        assert isinstance(result, list)

    def test_parse_resume_returns_resumedata(self, fake_pdf_file):
        """Ensure full parse_resume pipeline returns ResumeData."""
        framework = ResumeParserFramework(
            forced_document_chunk_list_output=TEST_DOCUMENT_CHUNK_LIST,
        )
        result = framework.parse_resume(str(fake_pdf_file))
        assert isinstance(result, ResumeData)
        
    def test_parse_file_default_unsupported_extension_raises(self, tmp_path):
        """Ensure unsupported file types raise FileNotSupportedError."""
        fake_txt = tmp_path / "fake_resume.txt"
        fake_txt.write_text(TEST_TEXT)
        
        framework = ResumeParserFramework()
        with pytest.raises(FileNotSupportedError):
            framework._parse_file(str(fake_txt))

    def test_parse_resume_calls_resume_extractor(self, mocker, fake_pdf_file):
        """Verify that ResumeExtractor.extract() is invoked."""
        mock_extract = mocker.patch.object(ResumeExtractor, "extract", return_value=TEST_RESUME_DATA)
        framework = ResumeParserFramework(
            forced_document_chunk_list_output=TEST_DOCUMENT_CHUNK_LIST
        )

        result = framework.parse_resume(str(fake_pdf_file))

        mock_extract.assert_called_once()
        assert isinstance(result, ResumeData)
        
    def test_setup_llm_client_and_extractor_map_with_default(self):
        """Verify default extractor_map is built when None is passed."""
        framework = ResumeParserFramework()
        # Internal call should already have run in __init__, just check values
        assert isinstance(framework.extractor_map, dict)
        assert all(isinstance(lst[0], FieldExtractor) for lst in framework.extractor_map.values())
        # LLM client is either None or LLMClient instance
        assert framework.llm_client is None or isinstance(framework.llm_client, LLMClient)
    
    def test_setup_llm_client_and_extractor_map_with_passed_map(self):
        """Verify passing a custom extractor_map is preserved."""
        custom_map = {
            "name": [NameExtractor(extraction_method="ner")],
            "email": [EmailExtractor()]
        }
        framework = ResumeParserFramework(extractor_map=custom_map)
        assert framework.extractor_map == custom_map
        
    def test_preload_spacy_models_is_called(self, mocker):
        """Ensure _preload_spacy_models invokes the helper correctly."""
        framework = ResumeParserFramework()
        mocker.patch.object(
            spacy_loader,
            "preload_spacy_models",
            return_value={}
        )
        framework._preload_spacy_models()  # Should not raise
    
    def test_preload_hf_models_is_called(self, mocker):
        """Ensure _preload_hf_models invokes the helper correctly."""
        framework = ResumeParserFramework()
        mocker.patch.object(
            hf_loader,
            "preload_hf_models",
            return_value={}
        )
        framework._preload_hf_models()  # Should not raise
        

    def test_preload_llm_client_only_initializes_once(self, mocker):
        """Confirm that LLM client is initialized only when needed."""
        framework = ResumeParserFramework(
            extractor_map={
                "skills": [SkillsExtractor(extraction_method="llm")]
            }
        )
        mock_init = mocker.patch.object(
            resume_parse_framework,
            "initialize_llm_if_needed",
            return_value=LLMClient()
        )
        framework.llm_client = None
        framework._preload_llm_client()
        mock_init.assert_called_once()
        
    def test_parse_file_with_forced_chunks(self, fake_pdf_file):
        """Explicitly test that forced_document_chunk_list_output bypasses actual parsing in _parse_file"""
        framework = ResumeParserFramework(forced_document_chunk_list_output=TEST_DOCUMENT_CHUNK_LIST)
        chunks = framework._parse_file(str(fake_pdf_file))
        assert chunks == TEST_DOCUMENT_CHUNK_LIST
    

# ---------------------------------------------------------------------
# TYPE B TESTS — Edge and failure cases
# ---------------------------------------------------------------------
@pytest.mark.usefixtures("FORCE_MOCK_LLM_RESPONSES")
class TestResumeParserFrameworkEdgeCases:
    """Test robustness against unusual inputs or states."""

    def test_parse_file_invalid_extension_raises_file_not_supported_error(self, tmp_path):
        """
        Ensure FileNotSupportedError is raised for unsupported file types.

        This test creates a fake .txt file in a temporary directory and confirms
        that the framework raises FileNotSupportedError when attempting to parse it.
        """
        fake_txt = tmp_path / "unsupported_resume.txt"
        fake_txt.write_text("This is not a supported resume format.")

        framework = ResumeParserFramework()

        with pytest.raises(FileNotSupportedError):
            framework._parse_file(str(fake_txt))

    def test_empty_document_chunk_list_does_not_crash(self, fake_pdf_file):
        """Ensure forcing an empty DocumentChunk list does not raise errors."""
        framework = ResumeParserFramework(forced_document_chunk_list_output=[])
        result = framework.parse_resume(str(fake_pdf_file))
        assert isinstance(result, ResumeData)

    def test_large_file_and_chunk_config_stores_correctly(self):
        """Check that custom config parameters (chunk_size, max_file_size_mb) are preserved."""
        framework = ResumeParserFramework(chunk_size=9999, max_file_size_mb=123.45)
        assert framework.chunk_size == 9999
        assert framework.max_file_size_mb == 123.45

    def test_setup_llm_client_and_extractor_map_with_empty_map(self):
        """Verify internal setup function handles empty extractor_map correctly."""
        framework = ResumeParserFramework()
        # Manually call with empty map
        framework._setup_llm_client_and_extractor_map(extractor_map={})
        assert isinstance(framework.extractor_map, dict)
        assert framework.extractor_map == {}  # Should not raise

    def test_preload_spacy_and_hf_models_with_empty_map(self):
        """Ensure _preload_spacy_models and _preload_hf_models handle empty extractor_map."""
        framework = ResumeParserFramework()
        framework.extractor_map = {}
        # Should not raise errors
        framework._preload_spacy_models()
        framework._preload_hf_models()

    def test_preload_llm_client_no_llm_needed(self, mocker):
        """Ensure _preload_llm_client does nothing if no extractor requires LLM."""
        framework = ResumeParserFramework()
        framework.extractor_map = {
            "name": [NameExtractor(extraction_method="ner")],
            "email": [EmailExtractor()]
        }
        mock_init = mocker.patch.object(
            llm_helpers,
            "initialize_llm_if_needed",
        )
        framework._preload_llm_client()
        mock_init.assert_not_called()


