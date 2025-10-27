"""extractor_map.py
Utilizes builds "extractor_map" dictionary utilized by ResumeExtractor
to determine which extraction steps are run.
"""

from typing import List, Dict, Optional

from src.models import ResumeData, DocumentChunk

from src.parse_classes.field_extractor.helper_functions.llm.llm_helpers import initialize_llm_if_needed

from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor


def build_default_extractor_map(
    llm_client: Optional["LLMClient"] = None,
    loaded_spacy_models: Optional[dict] = {},
    loaded_hf_models: Optional[dict] = {},
    document_chunk_list: Optional[List[DocumentChunk]] = None,
) -> dict:
    """
    Builds the default extractor map used by the resume parsing pipeline
    (i.e. ResumeExtractor).

    This map defines which extractor classes should handle each field type.
    Each field can now have multiple extractors, and each extractor entry
    can specify an `extraction_method` override.

    Args:
        loaded_spacy_models (Optional[dict], default={}):
            A cache of preloaded spaCy models shared across extractors.
        loaded_hf_models (Optional[dict], default={}):
            A cache of preloaded HuggingFace models.
        llm_client (Optional[LLMClient], default=None):
            Shared LLM client instance.
        document_chunk_list (Optional[List[DocumentChunk]], default=None):
            Document chunks to process. If None, extractors are initialized without it.

    Returns:
        dict:
            Mapping of field names -> list of extractor instances.

    Example:
        {
            "name": [
                NameExtractor(extraction_method="ner"),
                NameExtractor(extraction_method="llm")
            ],
            "email": [EmailExtractor()],
            "skills": [SkillsExtractor()],
        }
    """
    # Updated mapping: each field maps to a list of dicts with 'model' and optional 'extraction_method'
    default_extractor_classes_map = {
        "name": [
            {"model": NameExtractor, "extraction_method": "ner"},
            {"model": NameExtractor, "extraction_method": "llm"},
        ],
        "email": [
            {"model": EmailExtractor, "extraction_method": None}
        ],
        "skills": [
            {"model": SkillsExtractor, "extraction_method": None}
        ],
    }

    # Determine if LLM client is needed
    requires_llm = any(
        entry.get("extraction_method") == "llm"
        for field_list in default_extractor_classes_map.values()
        for entry in field_list
    )
    if requires_llm:
        llm_client = initialize_llm_if_needed(
            llm_client=llm_client,
            extraction_class_list=[
                entry["model"]
                for field_list in default_extractor_classes_map.values()
                for entry in field_list
            ]
        )

    # Shared defaults
    extractor_defaults = dict(
        loaded_spacy_models=loaded_spacy_models,
        loaded_hf_models=loaded_hf_models,
        llm_client=llm_client,
    )
    if document_chunk_list:
        extractor_defaults['document_chunk_list'] = document_chunk_list

    # Instantiate extractors
    extractor_map = {}
    for field, entries in default_extractor_classes_map.items():
        extractor_map[field] = []
        for entry in entries:
            model_cls = entry["model"]
            extraction_method = entry.get("extraction_method") or model_cls.DEFAULT_EXTRACTION_METHOD
            extractor_map[field].append(model_cls(extraction_method=extraction_method, **extractor_defaults))

    # Verify
    verify_extractor_map(extractor_map)

    return extractor_map


def verify_extractor_map(
    extractor_map: Optional[Dict[str, List[FieldExtractor]]]
):
    """
    Verifies the format and content of the extractor map.

    Args:
        extractor_map (Dict[str, List[FieldExtractor]]):
            Maps field names to extract to a list of extractor instances to try in order
            if the subsequent instance fails.

    This method performs validation checks on the extractor_map dictionary to ensure:
    1. The extractor_map is a dictionary
    2. All keys (fields) are strings
    3. All values are lists
    4. All items in the lists are FieldExtractor instances

    Raises:
        TypeError: If any of the following conditions are not met:
            - extractor_map is not a dictionary
            - field names are not strings
            - values are not lists
            - items in lists are not FieldExtractor instances
    """
    # Verification step: check format of extractor_map
    if not isinstance(extractor_map, dict):
        raise TypeError(
            f"extractor_map must be a dictionary, got {type(extractor_map).__name__}"
        )
    for field, extractors in extractor_map.items():
        if not isinstance(field, str):
            raise TypeError(
                f"Field names in extractor_map must be strings, got {type(field).__name__}"
            )
        if not isinstance(extractors, list):
            raise TypeError(
                f"Value for field '{field}' must be a list, got {type(extractors).__name__}"
            )
        for extractor in extractors:
            if not isinstance(extractor, FieldExtractor):
                raise TypeError(
                    f"All items in extractor list for field '{field}' must be "
                    f"FieldExtractor instances, got {type(extractor).__name__}"
                )
                

def unify_extractor_map_model_references(
    extractor_map: Dict[str, List[FieldExtractor]],
    document_chunk_list: Optional[List[DocumentChunk]] = None,
    llm_client: Optional["LLMClient"] = None,
    loaded_spacy_models: Optional[dict| None] = None,
    loaded_hf_models: Optional[dict| None] = None,
) -> dict:
    """
    Unifies pre-loaded model references to all extractors in the map.

    This method iterates over all FieldExtractor instances in `extractor_map`
    and assigns their self parameters to equivalents in the provided args.

    Returns:
        Optional[Dict[str, List[FieldExtractor]]]
    """
    for extraction_field_name, extractors in extractor_map.items():
        for extractor in extractors:
            # ALWAYS override document_chunk_list
            if document_chunk_list is not None:
                extractor.document_chunk_list = document_chunk_list
            
            # LLMCLient (only override if one isn't already assigned)
            if (
                getattr(extractor, "llm_client", None) is None
                and llm_client is not None
            ):
                extractor.llm_client = llm_client
            
            # SpaCy models cache. Only override empty loaded_spacy_models
            existing_spacy_models = getattr(extractor, "loaded_spacy_models", None)
            if (
                loaded_spacy_models is not None
                and (
                    existing_spacy_models is None
                    or len(existing_spacy_models) == 0
                )
            ):
                extractor.loaded_spacy_models = loaded_spacy_models

            # HuggingFace models cache. Only override empty loaded_hf_models
            existing_hf_models = getattr(extractor, "loaded_hf_models", None)
            if (
                loaded_hf_models is not None
                and (
                    existing_hf_models is None
                    or len(existing_hf_models) == 0
                )
            ):
                extractor.loaded_hf_models = loaded_hf_models