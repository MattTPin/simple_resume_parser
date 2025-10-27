"""llm_helpers.py
Functions to help with initiating a LLMClient class
"""

from typing import List, Optional

from src.parse_classes.field_extractor.helper_functions.llm.llm_client import LLMClient

def initialize_llm_if_needed(
    llm_client: Optional[LLMClient] = None,
    extraction_method: Optional[str] = "llm",
    extraction_class_list: Optional[List["src.parse_classes.field_extractor.field_extractor.FieldExtractor"]] = None,
) -> Optional[LLMClient]:
    """
    Initialize or validate an LLMClient if required by the extractors or extraction method.
    This helper ensures that an `LLMClient` is available only when necessary.

    Logic flow:
        1. If an existing `llm_client` is provided validates that it is an instance of `LLMClient`
        and returns it if it is.
        2. If no `extraction_method` is provide and or it is not `llm` returns `None`.
        3. If any extractor class or instance in `extraction_class_list` has an
            `extraction_method` of "llm" initialize and return a new LLMClient.
        4. Tests the client connection before returning.

    Args:
        llm_client (Optional[LLMClient]): Existing LLM client instance to use or validate.
        extraction_method (Optional[str]): The extraction method type (default: "llm").
        extraction_class_list (Optional[List[FieldExtractor]]): List of extractor
            classes or instances that may require LLM usage.

    Returns:
        Optional[LLMClient]: A ready-to-use or validated LLMClient instance, or None if not required.

    Raises:
        TypeError: If `llm_client` is provided but not an instance of `LLMClient`.
        ValueError: If an LLM is required but provider/model information is missing.
    """
    # Validate or update an existing LLMClient
    if llm_client is not None:
        if not isinstance(llm_client, LLMClient):
            raise TypeError("Provided llm_client must be an instance of LLMClient.")
        return llm_client

    # If extraction method was provided and is not LLM-based, skip
    if extraction_method != "llm":
        return None

    # If extraction_class_list was provided Check if any extractor class/instance requires LLM
    if isinstance(extraction_class_list, list):
        requires_llm = any(
            getattr(cls, "extraction_method", None) == "llm"
            for cls in extraction_class_list
        )
        if not requires_llm:
            return None

    # Initialize new LLM client if none exists and needed
    llm_client = LLMClient()
    llm_client.initialize_client()

    return llm_client
