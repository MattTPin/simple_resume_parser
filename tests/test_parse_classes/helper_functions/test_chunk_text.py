"""test_chunk_text.py
Run tests on chunking functions.
"""

from src.models import DocumentChunk

from src.parse_classes.file_parser.helpers.chunk_text import chunk_text, join_document_chunk_text
from src.test_helpers.dummy_classes import DummyExtractor

class TestChunkingAndJoining:
    def test_chunk_text_simple_space_split(self):
        """Chunks text at spaces, preserving word boundaries."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=4)
        
        # Each chunk should contain whole words
        for chunk in chunks:
            words = chunk.text.split()
            for word in words:
                assert word in text, f"Word '{word}' in chunk not in original text"

        # Joining should reconstruct original text
        joined = join_document_chunk_text(chunks)
        assert joined == text

    def test_chunk_text_none_chunk_size(self):
        """None chunk_size should return a single chunk."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=None)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_text_empty_string(self):
        """Empty text should return no chunks."""
        text = ""
        chunks = chunk_text(text, chunk_size=5)
        assert chunks == []

    def test_chunk_text_no_whitespace(self):
        """Text with no whitespace should not be chunked mid-word."""
        text = "Supercalifragilisticexpialidocious"
        chunks = chunk_text(text, chunk_size=5)
        # Should return single chunk since no whitespace to split
        assert len(chunks) == 1
        assert chunks[0].text == text
        
    def test_chunk_text_split_on_newline(self):
        """Chunks text properly when splitting at a newline."""
        text = "Hello world\nThis is a test\nAnother line"
        chunks = chunk_text(text, chunk_size=12)
        # Ensure newlines are preserved in chunks
        newline_present = any("\n" in c.text for c in chunks)
        assert newline_present, "Chunks should preserve newline characters"
        # Join should reconstruct original text
        joined = join_document_chunk_text(chunks)
        assert joined == text

    def test_join_document_chunk_text_char_limit(self):
        """Joining should respect a specified character limit."""
        chunks = [
            DocumentChunk(0, "12345"),
            DocumentChunk(1, "67890")
        ]
        joined = join_document_chunk_text(chunks, char_limit=7)
        assert joined == "1234567"
        
    def test_join_document_chunk_text_empty_list(self):
        """Joining an empty list should return an empty string."""
        joined = join_document_chunk_text([])
        assert joined == ""