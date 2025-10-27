"""validate_document_chunk_list.py
Check that list of DocumentChunks is valid
"""
from typing import List

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError, FieldExtractionConfigError


def validate_document_chunk_list(document_chunk_list: List[DocumentChunk]) -> None:
    """
    Validate that self.document_chunk_list is a non-empty list of DocumentChunk
    objects and that each DocumentChunk has the expected attributes.

    Args:
        document_chunk_list (List[DocumentChunk]): Resume chunks (prepared by FileParser)
            contained in a list.

    Raises:
        TypeError: If document_chunk_list is not a list or contains invalid items.
        ValueError: If document_chunk_list is empty or contains invalid values.
    """
    if not isinstance(document_chunk_list, list):
        raise TypeError("document_chunk_list must be a list of DocumentChunk objects.")

    if not document_chunk_list:
        raise FieldExtractionConfigError(message="document_chunk_list must not be empty.")

    for idx, chunk in enumerate(document_chunk_list):
        if not isinstance(chunk, DocumentChunk):
            raise TypeError(
                f"document_chunk_list[{idx}] is not a DocumentChunk (got {type(chunk).__name__})."
            )

        # Basic attribute checks
        if not hasattr(chunk, "chunk_index") or not isinstance(chunk.chunk_index, int):
            raise TypeError(
                f"DocumentChunk at position {idx} must have an integer 'chunk_index'."
            )
        if chunk.chunk_index < 0:
            raise ValueError(
                f"DocumentChunk.chunk_index must be non-negative (got {chunk.chunk_index})."
            )

        if not hasattr(chunk, "text") or not isinstance(chunk.text, str):
            raise TypeError(
                f"DocumentChunk at position {idx} must have a string 'text' attribute."
            )

    # Ensure chunk_index values are unique to avoid ambiguous ordering
    chunk_indices = [b.chunk_index for b in document_chunk_list]
    if len(chunk_indices) != len(set(chunk_indices)):
        raise ValueError("Duplicate 'chunk_index' values found in document_chunk_list.")