"""test_check_file_extension.py
Test check_file_extension function.
"""
import pytest

from src.exceptions import FileNotSupportedError

from src.parse_classes.file_parser.helpers.check_file_extension import check_file_extension

class TestCheckFileExtension:
    """Tests for the check_file_extension utility."""

    def test_valid_extension_returns_lowercase(self):
        """Return the lowercase file extension if it's supported."""
        supported = [".pdf", ".docx"]
        file_path = "resume.PDF"

        result = check_file_extension(file_path, supported)
        assert result == ".pdf"

    def test_valid_extension_in_lowercase(self):
        """Return correct extension when already lowercase."""
        supported = [".pdf", ".docx"]
        file_path = "resume.docx"

        result = check_file_extension(file_path, supported)
        assert result == ".docx"

    def test_unsupported_extension_raises_error(self):
        """Raise FileNotSupportedError if extension is not supported."""
        supported = [".pdf", ".docx"]
        file_path = "resume.txt"

        with pytest.raises(FileNotSupportedError) as exc_info:
            check_file_extension(file_path, supported)

        err = exc_info.value
        assert isinstance(err, FileNotSupportedError)
        assert err.extension == ".txt"
        assert err.supported_extensions == supported

    def test_no_extension_raises_error(self):
        """Raise FileNotSupportedError when file has no extension."""
        supported = [".pdf", ".docx"]
        file_path = "resume"

        with pytest.raises(FileNotSupportedError) as exc_info:
            check_file_extension(file_path, supported)

        err = exc_info.value
        assert err.extension == ""
        assert err.supported_extensions == supported

    def test_dot_in_filename_not_extension(self):
        """Ensure it extracts only the final extension."""
        supported = [".pdf", ".docx"]
        file_path = "resume.v1.docx"

        result = check_file_extension(file_path, supported)
        assert result == ".docx"

    def test_supported_extension(self):
        """Supports normal single-dot extensions."""
        supported = [".txt", ".md"]
        file_path = "document.txt"

        ext = check_file_extension(file_path, supported)
        assert ext == ".txt"