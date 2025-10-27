"""dummy_classes.py
Holds dummy classes for abstract classes to test with
"""
from typing import Optional
from src.parse_classes.field_extractor.field_extractor import FieldExtractor, EXTRACTION_METHODS

# Dummy subclass for testing where needed
class DummyExtractor(FieldExtractor):
    """A dummy FieldExtractor subclass for testing."""
    SUPPORTED_EXTRACTION_METHODS = ["regex", "llm", "ner"]
    DEFAULT_EXTRACTION_METHOD = "regex"
    
    REQUIRED_MODELS = {
        "llm": {
            "spacy": ["en_core_web_sm"],
            "hf": ["dslim/bert-base-NER"]
        },
    }

    def extract(self) -> Optional[str]:
        # Minimal implementation for testing
        return "dummy"