"""test_validate_document_chunk_list.py
Test validate_document_chunk_list
"""

import pytest
from src.models import DocumentChunk
from src.exceptions import FieldExtractionError, FieldExtractionConfigError
from src.parse_classes.field_extractor.helper_functions.validate_document_chunk_list import (
    validate_document_chunk_list
)


class TestValidateDocumentChunkList:
    # ----------------------
    # DocumentChunk validation
    # ----------------------
    
    def test_validate_document_chunk_list_type_and_empty(self):
        """Ensure invalid types and empty document_chunk_list raise appropriate errors."""
        # Non-list raises TypeError
        chunks = "not a list"
        with pytest.raises(TypeError):
            validate_document_chunk_list(chunks)

        # Empty list raises FieldExtractionConfigError
        chunks = []
        with pytest.raises(FieldExtractionConfigError):
            validate_document_chunk_list(chunks)

    def test_invalid_chunk_type(self):
        """Ensure invalid chunk objects in document_chunk_list raise TypeError."""
        chunks = [DocumentChunk(0, "text"), "not a chunk"]
        with pytest.raises(TypeError):
            validate_document_chunk_list(chunks)

    def test_invalid_chunk_attributes(self):
        """Ensure invalid chunk attributes (index/text) raise TypeError or ValueError."""
        # Non-integer chunk_index
        chunk = DocumentChunk(0, "text")
        chunk.chunk_index = "invalid"
        with pytest.raises(TypeError):
            validate_document_chunk_list([chunk])

        # Negative chunk_index
        chunk.chunk_index = -1
        with pytest.raises(ValueError):
            validate_document_chunk_list([chunk])

        # Missing text attribute
        chunk.chunk_index = 0
        del chunk.text
        with pytest.raises(TypeError):
            validate_document_chunk_list([chunk])

    def test_duplicate_chunk_indices(self):
        """Ensure duplicate chunk indices raise ValueError."""
        chunks = [DocumentChunk(0, "a"), DocumentChunk(0, "b")]
        with pytest.raises(ValueError):
            validate_document_chunk_list(chunks)

    def test_valid_chunks(self):
        """Ensure valid chunks pass without error."""
        chunks = [DocumentChunk(0, "a"), DocumentChunk(1, "b")]
        # Should not raise any exceptions
        validate_document_chunk_list(chunks)
