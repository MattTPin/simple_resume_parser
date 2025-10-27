"""test_file_parser.py
Comprehensive test suite for:
  - FileParser (abstract base)
"""

import os
import pytest

from src.exceptions import FileTooLargeError, FileEmptyError

from src.test_helpers.file_parsing import DummyTxtParser
from src.parse_classes.file_parser.file_parser import FileParser


class TestFileParser:
    """Unit tests for FileParser validation and abstract behavior."""

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate abstract FileParser directly."""
        with pytest.raises(TypeError):
            FileParser("some_file.txt")

    def test_file_not_found_error(self):
        """Raises FileNotFoundError if file does not exist."""
        with pytest.raises(FileNotFoundError):
            DummyTxtParser("non_existent_file.txt")

    def test_file_too_large_error(self, tmp_path):
        """Raises FileTooLargeError if file exceeds max_file_size_mb."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("small content")
        with pytest.raises(FileTooLargeError):
            DummyTxtParser(str(test_file), max_file_size_mb=0.000001)

    def test_unsupported_extension_error(self, tmp_path):
        """Raises FileNotSupportedError if file extension not in SUPPORTED_EXTENSIONS."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        from src.exceptions import FileNotSupportedError

        with pytest.raises(FileNotSupportedError):
            DummyTxtParser(str(test_file))  # DummyTxtParser supports only .txt

    def test_valid_file_passes_validation(self, tmp_path):
        """Valid file with supported extension and size passes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("some content")
        parser = DummyTxtParser(str(test_file))
        assert parser.file_path == str(test_file)
        assert parser.chunk_size is None
        assert parser.max_file_size_mb is None

    def test_pathlib_path_works(self, tmp_path):
        """Passing a Path object is allowed and stored correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("some content")
        parser = DummyTxtParser(test_file)  # Path object
        assert parser.file_path == test_file

    def test_empty_file_raises_file_empty_error(self, tmp_path):
        """_check_and_chunk_final_text should raise FileEmptyError for empty file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("")
        parser = DummyTxtParser(str(test_file))

        with pytest.raises(FileEmptyError) as exc_info:
            parser._check_and_chunk_final_text("")

        # Optional: assert the file_path is stored in the exception
        assert exc_info.value.file_path == str(test_file)

    def test_max_file_size_edge_cases(self, tmp_path):
        """Edge cases for max_file_size_mb boundary conditions."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("abc") # Bigger than 1 byte
        # Exactly equal to file size should pass
        size_mb = os.path.getsize(test_file) / (1024 * 1024)
        parser = DummyTxtParser(str(test_file), max_file_size_mb=size_mb)
        assert parser.file_path == str(test_file)
        # Slightly smaller than actual size triggers FileTooLargeError
        with pytest.raises(FileTooLargeError):
            # Set max file size to 1 byte
            DummyTxtParser(str(test_file), max_file_size_mb=size_mb - 1e-7)
    
    # ------------------------------------------
    # TEST _validate_file
    # ------------------------------------------
    
    def test_file_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError when file does not exist."""
        fake_path = tmp_path / "missing.txt"
        parser = DummyTxtParser.__new__(DummyTxtParser)
        parser.file_path = str(fake_path)
        parser.max_file_size_mb = None

        with pytest.raises(FileNotFoundError):
            parser._validate_file()

    def test_file_too_large_raises(self, tmp_path):
        """Should raise FileTooLargeError when file exceeds max_file_size_mb."""
        test_file = tmp_path / "large.txt"
        test_file.write_text("a" * 100)
        parser = DummyTxtParser.__new__(DummyTxtParser)
        parser.file_path = str(test_file)
        parser.max_file_size_mb = 0.000001  # Very small size limit

        with pytest.raises(FileTooLargeError):
            parser._validate_file()

    def test_valid_file_no_size_limit(self, tmp_path):
        """Valid file with no size limit should pass without errors."""
        test_file = tmp_path / "small.txt"
        test_file.write_text("some content")

        parser = DummyTxtParser.__new__(DummyTxtParser)
        parser.file_path = str(test_file)
        parser.max_file_size_mb = None  # No size limit

        # Should not raise anything
        parser._validate_file()

    def test_valid_file_within_size_limit(self, tmp_path):
        """File under the maximum file size should validate successfully."""
        test_file = tmp_path / "ok.txt"
        test_file.write_text("abc")

        parser = DummyTxtParser.__new__(DummyTxtParser)
        parser.file_path = str(test_file)
        parser.max_file_size_mb = 1  # 1 MB limit — very lenient

        parser._validate_file()  # Should not raise