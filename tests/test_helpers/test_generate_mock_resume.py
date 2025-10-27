"""test_generate_mock_resume.py
Test MockResumeGenerator
"""

import pytest

from src.models import DocumentChunk

from src.test_helpers.mock_resume_generator import (
    MockResumeGenerator,
    ChunkValues,
    ChunkTemplates,
    DUMMY_RESUME_BLOCKS,
)

class TestGenerateMockResume:
    """Unit tests for the MockResumeGenerator helper function."""

    # ----------------------
    # Basic generation tests
    # ----------------------
    def test_generate_default_resume(self):
        """Should generate a resume using default chunk values and templates."""
        chunks = MockResumeGenerator().generate()
        assert isinstance(chunks, list)
        assert all(isinstance(b, DocumentChunk) for b in chunks)
        assert len(chunks) > 0

    def test_generate_resume_with_custom_values(self):
        """Custom ChunkValues should appear in the output text."""
        values = ChunkValues(
            name="Alice Smith",
            email="alice@example.com",
            phone="+1 212-555-9876",
            linkedin_name="alice_smith",
            skills="Python, NLP, Deep Learning",
        )
        chunks = MockResumeGenerator(chunk_values=values).generate()
        joined_text = "\n".join(b.text for b in chunks)
        assert "Alice Smith" in joined_text
        assert "alice@example.com" in joined_text
        assert "+1 212-555-9876" in joined_text
        assert "Python, NLP, Deep Learning" in joined_text
        assert "alice_smith" in joined_text

    def test_generate_resume_custom_templates(self):
        """Custom ChunkTemplates should replace default text."""
        templates = ChunkTemplates(
            contact_info="CONTACT: {name} | {email} | {phone}",
            work_experience="CUSTOM WORK EXPERIENCE CHUNK",
            education="CUSTOM EDUCATION CHUNK",
            projects="CUSTOM PROJECTS CHUNK",
            skills="SKILLS: {skills}",
        )
        values = ChunkValues(name="Bob")
        chunks = MockResumeGenerator(chunk_values=values, chunk_templates=templates).generate()
        joined_text = "\n".join(b.text for b in chunks)
        assert "CUSTOM WORK EXPERIENCE CHUNK" in joined_text
        assert "CUSTOM EDUCATION CHUNK" in joined_text
        assert "CUSTOM PROJECTS CHUNK" in joined_text
        assert "CONTACT: Bob" in joined_text
        assert "SKILLS:" in joined_text

    def test_generate_resume_chunk_order(self):
        """Only chunks in chunk_order should appear and in the correct sequence."""
        order = ["skills", "contact_info", "projects"]
        values = ChunkValues(name="Charlie")
        chunks = MockResumeGenerator(chunk_values=values, chunk_order=order).generate()
        text_order = "\n".join(b.text for b in chunks)
        skills_idx = text_order.find(values.skills.split()[0])
        contact_idx = text_order.find("Charlie")
        projects_idx = text_order.find("PROJECTS")
        assert skills_idx < contact_idx < projects_idx

    # ----------------------
    # Chunking tests
    # ----------------------
    def test_chunk_size_applied(self):
        """Generated chunks should respect chunk_size if set."""
        chunks = MockResumeGenerator(chunk_size=50).generate()
        # We expect chunk_size to go a little over since it doesn't slice words 
        # so give 10 char buffer
        assert all(len(b.text) <= 70 for b in chunks)

    # ----------------------
    # Edge cases
    # ----------------------
    def test_empty_chunk_order(self):
        """If chunk_order is empty, no chunks should be generated."""
        chunks = MockResumeGenerator(chunk_order=[]).generate()
        assert chunks == []