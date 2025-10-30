"""file_parsers.py

Holds abstract FileParser class inherited by filetype-specific parsers.
"""

import os
from typing import List
from abc import ABC, abstractmethod

from src.config import SCANNER_DEFAULTS
from src.models import DocumentChunk
from src.exceptions import FileTooLargeError, FileEmptyError

from src.parse_classes.file_parser.helpers.chunk_text import(
    chunk_text,
)
from src.parse_classes.file_parser.helpers.check_file_extension import check_file_extension

class FileParser(ABC):
    """
    Abstract base class representing a generic file parser.

    All concrete parsers must implement the `parse` method.

    Args:
        file_path (str): Path to the file to parse.
        chunk_size (int | None, optional): Maximum number of characters per chunk. 
            If None, returns the whole text as a single chunk. Defaults to None.
        max_file_size_mb (float | None, optional): Maximum allowed file size in megabytes. 
            If None, no size limit is enforced. Defaults to None.

    Attributes:
        file_path (str): Path to the file.
        chunk_size (int | None): Maximum characters per chunk.
        max_file_size_mb (float | None): Maximum allowed file size.
    """
    # Parent level allowance of file extensions supported in at least one concrete class
    ALLOWED_EXTENSIONS = [".pdf", ".docx"]
    
    # Extensions supported by a specific concreted class (to be overwritten by children)
    SUPPORTED_EXTENSIONS = []

    def __init__(
        self,
        file_path: str,
        chunk_size: int | None = SCANNER_DEFAULTS.CHUNK_SIZE,
        max_file_size_mb: float | None = SCANNER_DEFAULTS.MAX_FILE_SIZE_MB
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_file_size_mb = max_file_size_mb
        self._validate_file()
        check_file_extension(self.file_path, self.SUPPORTED_EXTENSIONS)

    def _validate_file(self):
        """Validate whether the file can be parsed by this parser.

        Raises:
            FileNotFoundError: Raised if the file cannot be found at file_path
            FileTooLargeError: Raised if the file exceeds the max_file_size_mb
        """
        # Confirm that the file exists in the given path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Check file size if a max size is specified
        if self.max_file_size_mb is not None:
            # Convert MB to bytes (1 MB = 1024 * 1024 bytes)
            max_size_bytes = self.max_file_size_mb * 1024 * 1024
            actual_size_bytes = os.path.getsize(self.file_path)
            
            if actual_size_bytes > max_size_bytes:
                raise FileTooLargeError(
                    max_size=max_size_bytes,
                    actual_size=actual_size_bytes
                )
        
    def _check_and_chunk_final_text(self, full_text) -> List[DocumentChunk]:
        """
        Validate the extracted text and split it into `DocumentChunk` objects.

        This method checks if the provided `full_text` contains any readable content.
        If it is empty or whitespace-only, a `FileEmptyError` is raised. Otherwise,
        the text is split into chunks based on the parser's configured `chunk_size`.

        Args:
            full_text (str): The complete text extracted from a file. Can be from any
                supported file type (PDF, DOCX, etc.).

        Returns:
            List[DocumentChunk]: A list of `DocumentChunk` objects containing the
            sequentially numbered chunks of the input text.

        Raises:
            FileEmptyError: If `full_text` is empty or contains only whitespace.
        """
        if not full_text.strip():
            raise FileEmptyError(self.file_path)

        return chunk_text(
            text = full_text,
            chunk_size = self.chunk_size,
        )

    @abstractmethod
    def parse(self) -> List[DocumentChunk]:
        """
        Parses the file located at `self.file_path` and return its content as a list of DocumentChunk objects.

        Returns:
            List[DocumentChunk]: Each DocumentChunk contains:
                - chunk_index: int
                - text: str
        """
        pass