"""conftest_helpers.py
Helper functions for `tests/conftest.py`
"""

from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor


# --------------------------------------------------------------
# SETUP MONKEYPATCH FIXTURES
# --------------------------------------------------------------
def apply_mock_llm_patch(monkeypatch):
    """
    Core patching logic for FieldExtractor and its subclasses.

    Forces all extractors to use mock LLM responses by default:
      - `force_mock_llm_response=True`
      - `llm_dummy_response` set per subclass:
        - NameExtractor -> {"full_name": "John Doe"}
        - EmailExtractor -> {"email_address": "john.doe@example.com"}
        - SkillsExtractor -> {"skills": ["SQL"]}

    Notes:
      - Intended to be called from a fixture to control scope.
      - Does not yield; directly applies the monkeypatch.
    """
    original_init = FieldExtractor.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("force_mock_llm_response", True)

        if isinstance(self, NameExtractor):
            kwargs.setdefault("llm_dummy_response", {"full_name": "John Doe"})
        elif isinstance(self, EmailExtractor):
            kwargs.setdefault("llm_dummy_response", {"email_address": "john.doe@example.com"})
        elif isinstance(self, SkillsExtractor):
            kwargs.setdefault("llm_dummy_response", {"skills": ["SQL"]})

        original_init(self, *args, **kwargs)

    monkeypatch.setattr(FieldExtractor, "__init__", patched_init)