"""test_word_document_parser.py
Comprehensive test suite for:
  - WordDocumentParser
"""

import os
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

from src.parse_classes.file_parser.word_document_parser import WordDocumentParser

# Helper function for retrieving file paths
def list_files(directory: str, extension: str):
    """Return all files in a directory matching a specific extension."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extension)
    ]


# ---------------------------------------------------------------------------
# WordDocumentParser tests
# ---------------------------------------------------------------------------
class TestWordDocumentParser:
    """Tests for the WordDocumentParser class."""

    @pytest.mark.parametrize("docx_file", list_files(TEST_FILE_PATHS['docx_dir'], ".docx"))
    def test_valid_docx_parsing(self, docx_file):
        """Parse valid DOCX files and confirm DocumentChunk structure."""
        parser = WordDocumentParser(str(docx_file), chunk_size=100)
        chunks = parser.parse()

        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(len(c.text) > 0 for c in chunks)

    def test_invalid_filetypes_raise_filenotsupportederror(self):
        """Ensure non-DOCX files (PDF, TXT) raise FileNotSupportedError on init."""
        # PDF files
        for pdf_file in list_files(TEST_FILE_PATHS['pdf_dir'], ".pdf"):
            with pytest.raises(FileNotSupportedError):
                WordDocumentParser(str(pdf_file))
        # TXT files
        for txt_file in list_files(TEST_FILE_PATHS['txt_dir'], ".txt"):
            with pytest.raises(FileNotSupportedError):
                WordDocumentParser(str(txt_file))

    def test_file_too_large_raises_error(self):
        """Simulate oversized file via very small max_file_size_mb."""
        docx_file = list_files(TEST_FILE_PATHS['docx_dir'], ".docx")[0]
        with pytest.raises(FileTooLargeError):
            WordDocumentParser(str(docx_file), max_file_size_mb=0).parse()

    def test_empty_docx_raises_fileemptyerror(self):
        """Empty DOCX should raise FileEmptyError."""
        parser = WordDocumentParser(str(TEST_FILE_PATHS['empty_docx']))
        with pytest.raises(FileEmptyError) as exc_info:
            parser.parse()
        assert "no parsable text" in str(exc_info.value).lower()
        assert exc_info.value.file_path == str(TEST_FILE_PATHS['empty_docx'])

    def test_corrupted_docx_raises_fileopenerror(self, tmp_path):
        """Corrupted DOCX should raise FileOpenError."""
        docx_path = tmp_path / "corrupt.docx"
        docx_path.write_text("not a docx file")
        parser = WordDocumentParser(str(docx_path))
        with pytest.raises(FileOpenError):
            parser.parse()

    def test_pathlib_path_works(self):
        """WordDocumentParser should accept pathlib.Path objects as file_path."""
        docx_file = Path(list_files(TEST_FILE_PATHS['docx_dir'], ".docx")[0])
        parser = WordDocumentParser(docx_file)
        chunks = parser.parse()
        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_parse_returns_nonempty_for_nonempty_docx(self):
        """Ensure parse returns at least one chunk for non-empty DOCX."""
        docx_file = list_files(TEST_FILE_PATHS['docx_dir'], ".docx")[0]
        parser = WordDocumentParser(docx_file)
        chunks = parser.parse()
        assert len(chunks) > 0
        assert all(isinstance(c.text, str) and c.text.strip() != "" for c in chunks)

    def test_multiple_files(self):
        """Parse multiple DOCX files consecutively without issues."""
        docx_files = list_files(TEST_FILE_PATHS['docx_dir'], ".docx")[:3]
        for f in docx_files:
            parser = WordDocumentParser(f)
            chunks = parser.parse()
            assert len(chunks) > 0

    @pytest.mark.parametrize("docx_file", list_files(TEST_FILE_PATHS['docx_dir'], ".docx"))
    def test_docx_chunks_are_readable(self, docx_file):
        """Ensure chunks returned by WordDocumentParser are readable."""
        parser = WordDocumentParser(docx_file, chunk_size=100)
        chunks = parser.parse()
        assert_chunks_are_readable(chunks)

    def test_minimum_letter_ratio_too_high(self, tmp_path):
        """If min_letter_ratio is unrealistically high, assert_chunks_are_readable should fail."""
        chunks = [
            DocumentChunk(1, "@#$%^&*()1234567890"),
            DocumentChunk(2, "!~`|<>")
        ]
        with pytest.raises(AssertionError):
            assert_chunks_are_readable(chunks, min_letter_ratio=0.9)

    # ----- Future WordDoc Specific edge cases to check
    # def test_docx_with_textbox_included(self):
    #     """Ensure text inside textboxes is extracted and included in the parsed chunks."""
    #     parser = WordDocumentParser(str(TEST_FILE_PATHS['textbox_docx']))
    #     chunks = parser.parse()
    #     assert "Text from textbox" in "".join(c.text for c in chunks)

    # def test_docx_with_table_text(self):
    #     """Ensure text inside tables is extracted and included in the parsed chunks."""
    #     parser = WordDocumentParser(str(TEST_FILE_PATHS['table_docx']))
    #     chunks = parser.parse()
    #     assert "Table cell content" in "".join(c.text for c in chunks)

    # def test_password_protected_docx_raises_error(self):
    #     """Password-protected DOCX files should raise FileOpenError when parsing."""
    #     parser = WordDocumentParser(str(TEST_FILE_PATHS['protected_docx']))
    #     with pytest.raises(FileOpenError):
    #         parser.parse()