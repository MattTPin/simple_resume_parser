"""test_pdf_parser.py
Comprehensive test suite for:
  - PDFParser
"""

import pytest
from pathlib import Path

from src.test_helpers.file_parsing import (
    list_files,
    TEST_FILE_PATHS,
    assert_chunks_are_readable
)
from src.models import DocumentChunk
from src.exceptions import (
    FileOpenError,
    FileTooLargeError,
    FileNotSupportedError,
    FileEmptyError
)

from src.parse_classes.file_parser.pdf_parser import PDFParser

# ---------------------------------------------------------------------------
# PDFParser tests
# ---------------------------------------------------------------------------
class TestPDFParser:
    """Tests for the PDFParser class."""

    @pytest.mark.parametrize("pdf_file", list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf"))
    def test_valid_pdf_parsing(self, pdf_file):
        """Parse valid PDF files and confirm DocumentChunk structure."""
        parser = PDFParser(pdf_file, chunk_size=100)
        chunks = parser.parse()

        assert isinstance(chunks, list), "Output must be a list."
        assert all(isinstance(b, DocumentChunk) for b in chunks), "Each chunk must be a DocumentChunk."
        assert all(len(b.text) > 0 for b in chunks), "Chunks should contain non-empty text."

    def test_invalid_filetypes_raise_filenotsupportederror(self):
        """Ensure non-PDF files (e.g., DOCX, TXT) raise FileNotSupportedError on initialization."""
        # Test DOCX files
        for docx_file in list_files(TEST_FILE_PATHS['docx_dir'], ".docx"):
            with pytest.raises(FileNotSupportedError):
                PDFParser(docx_file)

        # Test TXT files
        for txt_file in list_files(TEST_FILE_PATHS['txt_dir'], ".txt"):
            with pytest.raises(FileNotSupportedError):
                PDFParser(txt_file)

    def test_file_too_large_raises_error(self):
        """Simulate an oversized file by setting a very small max_file_size_mb."""
        pdf_file = list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf")[0]
        with pytest.raises(FileTooLargeError):
            PDFParser(pdf_file, max_file_size_mb=0).parse()

    def test_empty_pdf_raises_error(self):
        """Parsing an empty PDF should raise FileEmptyError."""
        parser = PDFParser(str(TEST_FILE_PATHS['empty_pdf']))
        with pytest.raises(FileEmptyError) as exc_info:
            parser.parse()

        # Confirm error message mentions no readable text
        assert "no parsable text" in str(exc_info.value).lower()
        # Optional: ensure file_path is stored in exception
        assert exc_info.value.file_path == str(TEST_FILE_PATHS['empty_pdf'])

    def test_corrupted_pdf_raises_file_open_error(self, tmp_path):
        """Corrupted or non-PDF content should raise FileOpenError."""
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_text("this is not a pdf")
        parser = PDFParser(str(pdf_path))
        with pytest.raises(FileOpenError):
            parser.parse()

    def test_pathlib_path_works(self):
        """PDFParser should accept pathlib.Path objects as file_path."""
        pdf_file = Path(list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf")[0])
        parser = PDFParser(pdf_file)
        chunks = parser.parse()
        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_parse_returns_nonempty_for_nonempty_pdf(self):
        """Ensure parse returns at least one chunk for non-empty PDF."""
        pdf_file = list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf")[0]
        parser = PDFParser(pdf_file)
        chunks = parser.parse()
        assert len(chunks) > 0
        assert all(isinstance(c.text, str) and c.text.strip() != "" for c in chunks)

    def test_multiple_files(self):
        """Parse multiple PDFs consecutively without issues."""
        pdf_files = list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf")[:3]
        for pdf_file in pdf_files:
            parser = PDFParser(pdf_file)
            chunks = parser.parse()
            assert len(chunks) > 0
    
    @pytest.mark.parametrize("pdf_file", list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf"))
    def test_pdf_chunks_are_readable(self, pdf_file):
        """Ensure chunks returned by PDFParser are readable."""
        parser = PDFParser(pdf_file, chunk_size=100)
        chunks = parser.parse()
        assert_chunks_are_readable(chunks)

    def test_empty_pdf_raises_error(self):
        """Empty PDF should raise FileEmptyError (cannot be readable)."""
        parser = PDFParser(str(TEST_FILE_PATHS['empty_pdf']))
        with pytest.raises(FileEmptyError):
            chunks = parser.parse()

    def test_corrupted_pdf_raises_fileopenerror(self, tmp_path):
        """Corrupted PDF should raise FileOpenError (cannot be read)."""
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_text("not a pdf")
        parser = PDFParser(str(pdf_path))
        with pytest.raises(FileOpenError):
            parser.parse()

    def test_minimum_letter_ratio_too_high(self, tmp_path):
        """
        If min_letter_ratio is unrealistically high, assert_chunks_are_readable should fail.
        Simulate a PDF that returns mostly symbols.
        """
        chunks = [
            DocumentChunk(1, "@#$%^&*()1234567890"),
            DocumentChunk(2, "!~`|<>")
        ]

        with pytest.raises(AssertionError):
            assert_chunks_are_readable(chunks, min_letter_ratio=0.9)
            
    # ----- Future PDF Specific edge cases to check
    # def test_encrypted_pdf_raises_file_open_error(self):
    #     """PDFs that are password-protected should raise FileOpenError."""
    #     parser = PDFParser(str(TEST_FILE_PATHS['encrypted_pdf']))
    #     with pytest.raises(FileOpenError):
    #         parser.parse()

    # def test_scanned_pdf_raises_file_empty_error(self):
    #     """PDFs containing only images with no text should raise FileEmptyError."""
    #     parser = PDFParser(str(TEST_FILE_PATHS['scanned_pdf']))
    #     with pytest.raises(FileEmptyError):
    #         parser.parse()

    # def test_multi_column_pdf_includes_all_text(self):
    #     """PDFs with multi-column layout should include all text, even if extraction order is linear."""
    #     parser = PDFParser(str(TEST_FILE_PATHS['multi_column_pdf']))
    #     chunks = parser.parse()
    #     full_text = "".join(c.text for c in chunks)
    #     assert "Column 1 text" in full_text
    #     assert "Column 2 text" in full_text

    # def test_pdf_with_empty_pages(self):
    #     """PDFs containing empty pages should still parse non-empty pages correctly."""
    #     parser = PDFParser(str(TEST_FILE_PATHS['pdf_with_blank_pages']))
    #     chunks = parser.parse()
    #     assert len(chunks) > 0
    #     full_text = "".join(c.text for c in chunks)
    #     assert "Content from non-empty page" in full_text

    # def test_pdf_with_headers_and_footers_included(self):
    #     """Text from headers and footers should be extracted and included in chunks."""
    #     parser = PDFParser(str(TEST_FILE_PATHS['headers_footers_pdf']))
    #     chunks = parser.parse()
    #     full_text = "".join(c.text for c in chunks)
    #     assert "Header content" in full_text
    #     assert "Footer content" in full_text

    # def test_corrupted_pdf_raises_file_open_error(self, tmp_path):
    #     """Corrupted or truncated PDFs should raise FileOpenError."""
    #     pdf_path = tmp_path / "corrupt.pdf"
    #     pdf_path.write_text("not a real pdf content")
    #     parser = PDFParser(str(pdf_path))
    #     with pytest.raises(FileOpenError):
    #         parser.parse()