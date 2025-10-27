"""config.py
Holds various defaults for different resume parser settings.
"""

from dataclasses import dataclass, field

# --------------------------------------------------------------
# SETUP DEFAULT VALUES
# --------------------------------------------------------------
@dataclass
class ScannerDefaults:
    """
    Default settings for parameters used across simple_resume_parser repo.
    """
    # ---- FileParser settings ----
    CHUNK_SIZE: int = field(
        default = 500,
        metadata = {
            "description": "Size of each chunk in characters"
    })
    MAX_FILE_SIZE_MB: float = field(
        default = 5.0,
        metadata = {
            "description": "Maximum allowed file size in MB"
    })
    
    # ---- ResumeExtractor settings ----
    MAX_THREADS: int = field(
        default = 3,
        metadata = {
            "description": "Maximum number of threads to use"
    })
    
    # ---- LLMClient settings ----
    LLM_PROVIDER: str = field(
        default = "anthropic",
        metadata = {
            "description": 'LLM provider: "anthropic"'
    })
    ANTHROPIC_MODEL_ID: str = field(
        default = "claude-haiku-4-5",
        metadata = {
            "description": "Anthropic model ID"
    })
    # NOT IMPLEMENTED
    # OPENAI_MODEL_ID: str = field(
    #     default = "gpt-4o-mini",
    #     metadata = {
    #         "description": "OpenAI model ID"
    # })
    # MISTRAL_MODEL_ID: str = field(
    #     default = "mistral-small-latest",
    #     metadata = {
    #         "description": "Mistral model ID"
    # })


# Import this where needed
SCANNER_DEFAULTS = ScannerDefaults()

