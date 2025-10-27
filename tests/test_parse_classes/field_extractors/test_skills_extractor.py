"""test_skills_extractor.py
Run tests on SkillsExtractor
"""
import pytest

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError

from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor

from src.test_helpers.mock_resume_generator import (
    ChunkValues,
    ChunkTemplates,
    DUMMY_RESUME_BLOCKS
)

from src.test_helpers.dummy_variables.dummy_chunk_lists import(
    MOCK_RESUME_GENERATOR_0,
    MOCK_RESUME_GENERATOR_1,
    MOCK_RESUME_GENERATOR_2
)

# TODO: PASS IN LLM CLIENT AND MODELS

# ---------------------------------------------------------------------------
# Test suite for SkillsExtractor
# ---------------------------------------------------------------------------
# Generate simple mock resumes
EXAMPLE_RESUME_SIMPLE_0 = MOCK_RESUME_GENERATOR_0.generate()
EXAMPLE_RESUME_SIMPLE_1 = MOCK_RESUME_GENERATOR_1.generate()
EXAMPLE_RESUME_SIMPLE_2 = MOCK_RESUME_GENERATOR_2.generate()

# ==============================================================
# EDGE CASE RESUME EXAMPLES
# ==============================================================
# Lowercase header
SKILLS_EXAMPLE_LOWERCASE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="python, java, sql"
    ),
    chunk_templates=ChunkTemplates(
        skills="skills: {skills}"
    )
).generate()

# Header followed by colon and comma-separated list
SKILLS_EXAMPLE_INLINE_LIST = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Python, C++, TensorFlow, AWS"
    ),
    chunk_templates=ChunkTemplates(
        skills="Skills: {skills}"
    )
).generate()

# Header repeated twice
SKILLS_EXAMPLE_INLINE_LIST = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Skills: Python, C++, TensorFlow, AWS"
    ),
    chunk_templates=ChunkTemplates(
        skills="Skills: {skills}"
    )
).generate()

# “Technical Skills” instead of “Skills”
SKILLS_EXAMPLE_TECHNICAL = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Technical Skills\nPython\nR\nSQL"
    ),
    chunk_templates=ChunkTemplates(
        skills="Technical Skills\n{skills}"
    )
).generate()

# “Expertise” label used instead
SKILLS_EXAMPLE_EXPERTISE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Python, Docker, Kubernetes"
    ),
    chunk_templates=ChunkTemplates(
        skills="Areas of Expertise - {skills}"
    )
).generate()

# “Strengths” section instead of “Skills”
SKILLS_EXAMPLE_STRENGTHS = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Problem-solving, Adaptability, Collaboration"
    ),
    chunk_templates=ChunkTemplates(
        skills="Strengths:\n{skills}"
    )
).generate()

# Skills embedded inline with other text (not a clear section)
SKILLS_EXAMPLE_INLINE_EMBEDDED = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Proficient in Python, SQL, and Excel with years of experience."
    ),
    chunk_templates=ChunkTemplates(
        skills=DUMMY_RESUME_BLOCKS['skills_fillable'][0]
    )
).generate()

# Skills section appears at the top before contact info
SKILLS_EXAMPLE_AT_TOP = MOCK_RESUME_GENERATOR_0.clone(
    chunk_order=["skills", "contact_info", "experience", "education"],
    chunk_values=ChunkValues(
        skills="Python, JavaScript, HTML, CSS"
    ),
    chunk_templates=ChunkTemplates(
        skills=DUMMY_RESUME_BLOCKS['skills_fillable'][0]
    )
).generate()

# No explicit “skills” label (list only)
SKILLS_EXAMPLE_NO_LABEL = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Python | Java | SQL | Tableau"
    ),
    chunk_templates=ChunkTemplates(
        skills="{skills}"
    )
).generate()

# Skills with subcategories
SKILLS_EXAMPLE_SUBCATEGORIES = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Programming: Python, Java\nTools: Docker, AWS\nSoft Skills: Leadership, Mentorship"
    ),
    chunk_templates=ChunkTemplates(
        skills="Technical Skills\n{skills}"
    )
).generate()

# Skills hidden in long paragraph form
SKILLS_EXAMPLE_PARAGRAPH = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills=(
            "During my career, I have developed strong skills in Python, SQL, and cloud "
            "infrastructure management with AWS and Azure, while mentoring junior developers."
        )
    ),
    chunk_templates=ChunkTemplates(
        skills=DUMMY_RESUME_BLOCKS['skills_fillable'][0]
    )
).generate()

# Misleading header (contains word 'skills' but not actually skills)
SKILLS_EXAMPLE_FALSE_POSITIVE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="Soft Skills Training\nLed workshops on management and communication."
    ),
    chunk_templates=ChunkTemplates(
        skills="{skills}"
    )
).generate()

# Empty skills section
SKILLS_EXAMPLE_EMPTY = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="\n"
    ),
    chunk_templates=ChunkTemplates(
        skills=DUMMY_RESUME_BLOCKS['skills_fillable'][0]
    )
).generate()

# Skills section surrounded by noisy formatting or whitespace
SKILLS_EXAMPLE_NOISY_FORMATTING = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="\n\n=== SKILLS ===\n\n- Python\n- SQL\n- Machine Learning\n\n"
    ),
    chunk_templates=ChunkTemplates(
        skills="{skills}"
    )
).generate()

# Resume missing skills entirely
SKILLS_EXAMPLE_NO_SECTION = MOCK_RESUME_GENERATOR_0.clone(
    chunk_order=["contact_info", "education", "work_experience"],
).generate()

# Skills section appears twice (duplicated)
SKILLS_EXAMPLE_DUPLICATE = MOCK_RESUME_GENERATOR_0.clone(
    chunk_values=ChunkValues(
        skills="SKILLS\nPython, SQL\nSKILLS\nLeadership, Communication"
    ),
    chunk_templates=ChunkTemplates(
        skills=DUMMY_RESUME_BLOCKS['skills_fillable'][0]
    )
).generate()


class TestSkillsExtractor:
    """Unit tests for the SkillsExtractor using multiple LLM testing modes."""
    # ---------------------------------------------------------------------
    # Config tests
    # ---------------------------------------------------------------------
    def test_invalid_extraction_method_raises_notimplemented(self):
        """Providing an unsupported extraction method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SkillsExtractor(EXAMPLE_RESUME_SIMPLE_0, extraction_method="ner")

    def test_default_method_is_llm(self):
        """If no method is specified, default should be llm."""
        extractor = SkillsExtractor(EXAMPLE_RESUME_SIMPLE_0)
        assert extractor.extraction_method == "llm"


    # ---------------------------------------------------------------------
    # Internal helper: _find_skills_text
    # ---------------------------------------------------------------------
    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_find_skills_text_success(self, extraction_method):
        """_find_skills_text should return relevant text chunks when skills pattern exists."""
        chunks = [
            DocumentChunk(chunk_index=1, text="Experience with Python and Java."),
            DocumentChunk(chunk_index=2, text="Skills: Python, SQL, AWS"),
            DocumentChunk(chunk_index=3, text="Education: BS Computer Science"),
        ]
        extractor = SkillsExtractor(chunks, extraction_method=extraction_method)
        result = extractor._find_skills_text()
        assert isinstance(result, str)
        assert "Python" in result
        assert "AWS" in result

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_find_skills_text_failure(self, extraction_method):
        """_find_skills_text should return raise FieldExtractionError when no result found."""
        chunks = [
            DocumentChunk(chunk_index=1, text="Experience with customer service."),
            DocumentChunk(chunk_index=2, text="Education: BA in English Literature"),
        ]
        extractor = SkillsExtractor(chunks, extraction_method=extraction_method)
        with pytest.raises(FieldExtractionError):
            extractor._llm_extract()
            
    # ---------------------------------------------------------------------
    # Explicit error cases in _llm_extract
    # ---------------------------------------------------------------------
    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_llm_extract_no_skills_section(self, extraction_method):
        """Raises FieldExtractionError when _find_skills_text returns empty."""
        chunks = [DocumentChunk(chunk_index=1, text="Completely unrelated content.")]
        extractor = SkillsExtractor(
            chunks,
            extraction_method=extraction_method,
            force_mock_llm_response=True,
            llm_dummy_response=["python", "SQL"],
        )
        with pytest.raises(FieldExtractionError):
            extractor._llm_extract()

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_llm_extract_missing_skills_key(self, extraction_method):
        """Raises FieldExtractionError when LLM dict response lacks 'skills' key."""
        chunks = [DocumentChunk(chunk_index=1, text="Skills: Python, SQL")]
        extractor = SkillsExtractor(
            chunks,
            extraction_method=extraction_method,
            force_mock_llm_response=True,
            llm_dummy_response={"not_skills": ["python"]},  # wrong key
        )
        with pytest.raises(FieldExtractionError, match="`skills`"):
            extractor._llm_extract()
    
    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_llm_extract_wrong_type_skills_value(self, extraction_method):
        """Raises FieldExtractionError when LLM dict response 'skills' isn't a list."""
        chunks = [DocumentChunk(chunk_index=1, text="Skills: Python, SQL")]
        extractor = SkillsExtractor(
            chunks,
            extraction_method=extraction_method,
            force_mock_llm_response=True,
            llm_dummy_response={"skills": "python"},  # value is str not list
        )
        with pytest.raises(FieldExtractionError, match="`skills`"):
            extractor._llm_extract()

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_llm_extract_empty_skills_list(self, extraction_method):
        """Returns empty list when LLM returns a valid but empty 'skills' array."""
        chunks = [DocumentChunk(chunk_index=1, text="Skills:")]
        extractor = SkillsExtractor(
            chunks,
            extraction_method=extraction_method,
            force_mock_llm_response=True,
            llm_dummy_response={"skills": []},
        )

        result = extractor._llm_extract()
        assert result == []

@pytest.mark.usefixtures("USE_MOCK_LLM_RESPONSE_SETTING")
class TestSkillsExtractorMockLlmModeAware:
    # ----------------------
    # Basic Positive Extractions on Simple Resumes
    # ----------------------
    @pytest.mark.parametrize(
        "document_chunk_list,expected",
        [
            (EXAMPLE_RESUME_SIMPLE_0, ["Google Analytics", "Data Cleaning", "SQL", "Custom Funnels", "Power BI"]),
            (EXAMPLE_RESUME_SIMPLE_1, ["Pet footwear", "Partial Budget", "Water Flow Meters"]),
        ],
    )
    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_simple_resumes(self, document_chunk_list, expected, extraction_method):
        """
        Each extraction method should correctly extract from simple resumes.

        Instead of exact match, verify that the expected entries are contained
        in the extracted output (case-insensitive).
        """
        extractor = SkillsExtractor(
            document_chunk_list=document_chunk_list,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": expected},
        )
        result = extractor.extract()

        assert isinstance(result, list)
        
        # Make everything lowercase for case-insensitive comparison
        lower_result = [str(r).lower() for r in result]

        for item in expected:
            assert item.lower() in lower_result, f"Expected '{item}' not found in result: {result}"

    # ---------------------------------------------------------------------
    # Edge Case Positive extraction scenarios
    # ---------------------------------------------------------------------

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_lowercase_header(self, extraction_method):
        """Extracts skills listed under a lowercase 'skills' header."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_LOWERCASE,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "java", "sql"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == ["java", "python", "sql"]

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_inline_list(self, extraction_method):
        """Extracts inline comma-separated list of skills."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_INLINE_LIST,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "c++", "tensorflow", "aws"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["python", "c++", "tensorflow", "aws"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_technical_skills(self, extraction_method):
        """Extracts skills under a 'Technical Skills' section."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_TECHNICAL,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "r", "sql"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["python", "r", "sql"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_expertise_label(self, extraction_method):
        """Extracts skills under a section labeled 'Areas of Expertise'."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_EXPERTISE,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "docker", "kubernetes"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["python", "docker", "kubernetes"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_strengths_section(self, extraction_method):
        """Extracts soft skills listed under 'Strengths'."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_STRENGTHS,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["problem-solving", "adaptability", "collaboration"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["problem-solving", "adaptability", "collaboration"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_skills_at_top(self, extraction_method):
        """Extracts skills listed at the top of the resume."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_AT_TOP,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "javascript", "html", "css"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["python", "javascript", "html", "css"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_subcategories(self, extraction_method):
        """Handles skills organized by subcategories like 'Technical' and 'Leadership'."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_SUBCATEGORIES,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "java", "docker", "aws", "leadership", "mentorship"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(
            ["python", "java", "docker", "aws", "leadership", "mentorship"]
        )

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_noisy_formatting(self, extraction_method):
        """Extracts correctly even with irregular whitespace and symbols."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_NOISY_FORMATTING,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "sql", "machine learning"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["python", "sql", "machine learning"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_duplicate_skills(self, extraction_method):
        """Removes duplicates from extracted skill list."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_DUPLICATE,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": ["python", "sql", "leadership", "communication"]},
        )
        result = extractor.extract()
        assert sorted([s.lower() for s in result]) == sorted(["python", "sql", "leadership", "communication"])

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_no_label_should_fail(self, extraction_method):
        """Returns empty list when no recognizable 'skills' label exists."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_NO_LABEL,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": []},
        )
        with pytest.raises(FieldExtractionError):
            extractor.extract()

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_paragraph_form_should_fail(self, extraction_method):
        """Returns empty list when skills are written in paragraph form."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_PARAGRAPH,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": []},
        )
        print("extractor.force_mock_llm_response is", extractor.force_mock_llm_response)
        result = extractor.extract()
        assert result == []

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_false_positive_should_fail(self, extraction_method):
        """Returns empty list when section is misidentified as 'skills' but isn't."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_FALSE_POSITIVE,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": []},
        )
        result = extractor.extract()
        assert result == []

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_empty_section_should_fail(self, extraction_method):
        """Returns empty list when the skills section exists but is empty."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_EMPTY,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": []},
        )
        result = extractor.extract()
        assert result == []

    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_no_section_should_fail(self, extraction_method):
        """Returns empty list when the resume has no skills section at all."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_NO_SECTION,
            extraction_method=extraction_method,
            llm_dummy_response={"skills": []},
        )
        with pytest.raises(FieldExtractionError):
            extractor.extract()
            

class TestSkillsExtractorLlmSpecific:
    # ---------------------------------------------------------------------
    # Mode-limited runs
    # ---------------------------------------------------------------------
    
    # ---------------------------------------------------------------------
    # Edge Case Negative / failure scenarios
    # ---------------------------------------------------------------------
    @pytest.mark.parametrize("extraction_method", SkillsExtractor.SUPPORTED_EXTRACTION_METHODS)
    def test_empty_llm_response_should_fail(self, extraction_method):
        """Confirm failure if llm response is an empty list."""
        extractor = SkillsExtractor(
            document_chunk_list=SKILLS_EXAMPLE_INLINE_EMBEDDED,
            extraction_method=extraction_method,
            force_mock_llm_response=True,
            llm_dummy_response={"skills": []},
        )
        result = extractor.extract()
        assert result == []
    
    @pytest.mark.parametrize(
        "document_chunk_list,expected",
        [
            (EXAMPLE_RESUME_SIMPLE_0, ["Google Analytics", "Data Cleaning", "SQL", "Custom Funnels", "Power BI"]),
            (EXAMPLE_RESUME_SIMPLE_1, ["Pet footwear", "Partial Budget", "Water Flow Meters"]),
        ],
    )
    def test_find_skills_text_llm_basic_only_mode(self, document_chunk_list, expected, LLM_TEST_MODE):
        if LLM_TEST_MODE != "basic_only":
            pytest.skip("Only run this test in basic_only LLM mode")
        extractor = SkillsExtractor(
            document_chunk_list=document_chunk_list,
            extraction_method="llm"
        )
        result = extractor.extract()
        assert isinstance(result, list)
        result_lower = [r.lower() for r in result]
        for item in expected:
            assert item.lower() in result_lower