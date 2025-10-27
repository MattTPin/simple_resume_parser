"""word_parser.py

Holds WordDocumentParser class using docx2txt for text extraction.
"""
from typing import List
import docx2txt

from src.models import DocumentChunk
from src.exceptions import FileOpenError, FileEmptyError
from src.parse_classes.file_parser.file_parser import FileParser


class WordDocumentParser(FileParser):
    """
    Concrete parser for Microsoft Word documents (.docx) with chunking support.

    This class extends the abstract ``FileParser`` and uses ``docx2txt`` to
    extract textual content (including from textboxes). The extracted text
    is then split into manageable chunks based on the configured ``chunk_size``.

    Args:
        file_path (str): Path to the Word document to parse.
        chunk_size (int | None, optional): Maximum number of characters per chunk.
            If None, returns the entire document as a single chunk.
        max_file_size_mb (float | None, optional): Maximum allowed file size in
            megabytes. If None, no file size limit is enforced.

    Attributes:
        SUPPORTED_EXTENSIONS (List[str]): File extensions supported by this parser
            (only ``.docx``).
    """

    SUPPORTED_EXTENSIONS = ['.docx']

    def parse(self) -> List[DocumentChunk]:
        """
        Parses the Word document and returns a list of ``DocumentChunk`` objects.

        Returns:
            List[DocumentChunk]: The chunked textual content extracted from the Word document.

        Raises:
            FileOpenError: If the file cannot be opened or read.
            FileEmptyError: If the document contains no readable text.
        """
        full_text = self._get_docx_contents()
        return self._check_and_chunk_final_text(full_text)

    def _get_docx_contents(self) -> str:
        """
        Opens the Word document using docx2txt and extracts all text content (including
        textboxes).

        Returns:
            str: The raw extracted text from the document.

        Raises:
            FileOpenError: If the Word document cannot be opened or read.
        """
        try:
            full_text = docx2txt.process(self.file_path)
        except Exception as e:
            raise FileOpenError(self.file_path, str(e))
        
        return full_text.strip()