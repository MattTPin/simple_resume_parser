"""file_parsing.py
Helper functions to test loading files in.
"""

import os
from pathlib import Path
import string
from typing import List

from src.models import DocumentChunk
from src.parse_classes.file_parser.file_parser import FileParser

# TEST FILE PATHS
TEST_FILE_PATHS = {
    'pdf_dir': Path("test_documents/test_pdfs"),
    'docx_dir': Path("test_documents/test_docx"),
    'txt_dir': Path("test_documents/test_txt"),
    'empty_pdf': Path("test_documents/empty/empty_pdf.pdf"),
    'empty_docx': Path("test_documents/empty/empty_docx.docx")
}

# DummyTxtParser to test with
class DummyTxtParser(FileParser):
    """Simple subclass of FileParser to test _validate_file logic."""
    SUPPORTED_EXTENSIONS = [".txt"]
    def parse(self):
        return []

# Helper function for retrieving file paths
def list_files(directory: str, extension: str):
    """Return all files in a directory matching a specific extension."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extension)
    ]

# Function to check readability of final text
def assert_chunks_are_readable(
    chunks: List[DocumentChunk],
    min_letter_ratio: float = 0.5,
    extra_allowed: str = "–—•◦·"  # add extra symbols commonly found in resumes
):
    """
    IMPROVEMENT: Consider upgrading to python package that handles this for you.
    
    Assert that the text contained in a list of DocumentChunk objects is readable.

    Checks performed:
        1. All characters are printable or whitespace (includes common resume symbols).
        2. At least a certain proportion of characters are alphabetic.
        3. Each non-empty line is of reasonable length (3–200 chars).

    Args:
        chunks (List[DocumentChunk]): The list of chunks to validate.
        min_letter_ratio (float, optional): Minimum ratio of letters to total characters. Defaults to 0.5.
        extra_allowed (str, optional): Additional characters to allow besides letters, digits, punctuation, whitespace.

    Raises:
        AssertionError: If any of the checks fail, including a snippet of the offending text.
    """
    full_text = "".join(c.text for c in chunks)

    # 1. Printable characters (Unicode-aware)
    non_printable = [
        c for c in full_text
        if not (c.isprintable() or c.isspace() or c in extra_allowed)
    ]
    if non_printable:
        snippet = "".join(non_printable[:50])
        raise AssertionError(f"Parsed text contains unreadable characters: {snippet!r}")

    # 2. Sufficient letters
    letters = sum(c.isalpha() for c in full_text)
    total_chars = len(full_text) if len(full_text) > 0 else 1
    ratio = letters / total_chars
    if ratio < min_letter_ratio:
        snippet = full_text[:100]
        raise AssertionError(
            f"Parsed text seems gibberish (letter ratio {ratio:.2f} < {min_letter_ratio}): {snippet!r}"
        )