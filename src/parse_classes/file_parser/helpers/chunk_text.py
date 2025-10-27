"""chunk_text.py
Used to chunk a continuous output of text into a list of DocumentChunks. 
"""

from typing import List, Optional 

from src.models import DocumentChunk

def chunk_text(
    text: str,
    chunk_size: Optional[int] = 0
) -> List[DocumentChunk]:
    """
    Split the input text into `DocumentChunk` objects without splitting words.

    Chunks are approximately `chunk_size` characters long. If a chunk would
    split a word, it is extended until the next whitespace character
    (space, tab, newline, etc.). This may cause chunks to slightly exceed
    `chunk_size`.

    If `chunk_size` is None or non-positive, the entire text is returned
    as a single chunk.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int | None, optional): Approximate maximum size of each chunk.
            Defaults to 0 (no chunking).

    Returns:
        List[DocumentChunk]: List of sequential chunks with `chunk_index` and `text`.
    """
    if chunk_size is None or chunk_size <= 0:
        return [DocumentChunk(chunk_index=1, text=text)]

    chunks = []
    start = 0
    chunk_index = 1
    text_length = len(text)

    while start < text_length:
        # Initial end position based on chunk_size
        end = min(start + chunk_size, text_length)

        # Extend to next whitespace to avoid splitting a word
        if end < text_length and not text[end].isspace():
            for i in range(end, text_length):
                if text[i].isspace():
                    end = i + 1  # include the whitespace
                    break
            else:
                end = text_length  # no whitespace found, take till end

        chunk_text_str = text[start:end]
        if chunk_text_str:  # avoid empty chunks
            chunks.append(DocumentChunk(chunk_index=chunk_index, text=chunk_text_str))
            chunk_index += 1

        start = end  # move start forward; no overlap

    return chunks


def join_document_chunk_text(
    chunks: List[DocumentChunk],
    char_limit: Optional[int] = None,
) -> str:
    """
    Reconstruct the original text from a list of `DocumentChunk` objects,
    aligning overlapping chunks and preserving newlines and tabs.

    Each chunk is assumed to have been produced by `chunk_text`, which may
    include overlaps. Overlapping text is removed to avoid duplication.
    Leading/trailing whitespace is trimmed only for overlap detection; internal
    formatting (newlines, tabs) is preserved.

    Args:
        chunks List[DocumentChunk]: Chunks to join.
        char_limit (int | None): Maximum number of characters to include in
            the final text. If None, all text is included.

    Returns:
        str: Combined sequential text from all chunks with overlaps removed.
    """
    # Sort chunks by index
    chunks_sorted = sorted(chunks, key=lambda c: c.chunk_index)
    combined_text = "".join(chunk.text for chunk in chunks_sorted)

    if char_limit is not None:
        combined_text = combined_text[:char_limit]

    return combined_text