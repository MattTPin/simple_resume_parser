"""pdf_parser.py

Holds PDFParser class.
"""
from typing import List

import pymupdf

from src.exceptions import FileOpenError
from src.models import DocumentChunk
from src.parse_classes.file_parser.helpers.chunk_text import chunk_text

from src.parse_classes.file_parser.file_parser import FileParser

class PDFParser(FileParser):
    """
    Concrete parser for PDF documents (.pdf) with chunking support.

    This class extends the abstract ``FileParser`` and uses PyMuPDF to extract
    textual content from PDF files. The extracted text is then split into
    manageable chunks based on the configured ``chunk_size``.

    Args:
        file_path (str): Path to the PDF file to parse.
        chunk_size (int | None, optional): Maximum number of characters per chunk.
            If None, returns the entire document as a single chunk.
        max_file_size_mb (float | None, optional): Maximum allowed file size in
            megabytes. If None, no file size limit is enforced.

    Attributes:
        SUPPORTED_EXTENSIONS (List[str]): List of file extensions supported by
            this parser (only ``.pdf``).
    """
    SUPPORTED_EXTENSIONS = ['.pdf']

    def parse(self) -> List[DocumentChunk]:
        """
        Parses the PDF document and returns a list of ``DocumentChunk`` objects.

        Returns:
            List[DocumentChunk]: The chunked textual content extracted from the PDF.

        Raises:
            FileOpenError: If the file cannot be opened or read by PyMuPDF.
            FileEmptyError: If the PDF contains no readable text.
        """
        full_text = self._get_pdf_contents()
        return self._check_and_chunk_final_text(full_text)
    
    def _get_pdf_contents(self) -> str:
        """
        Opens the PDF file using PyMuPDF, combines any pages, and returns
        its contents as a string.

        Returns:
            str: Full text inside pdf.

        Raises:
            FileOpenError: If the PDF file cannot be opened.
        """
        try:
            doc = pymupdf.open(self.file_path)
        except Exception as e:
            raise FileOpenError(self.file_path, str(e))
        
        full_text = ""
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            full_text += page.get_text("text") + "\n"

        doc.close()
        
        return full_text

