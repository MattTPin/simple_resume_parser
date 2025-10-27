"""spacy_loader.py
Used to load spacy models with optional cache.
"""
from typing import Dict, Optional, List, Tuple
import sys
import subprocess

import spacy
from spacy.util import is_package
from spacy.language import Language

def load_spacy_model(
    model_name: str,
    loaded_spacy_models: Optional[Dict[str, Language]] = None
) -> Language:
    """
    Load a SpaCy model, optionally using a provided cache dictionary.

    If the model is already present in `loaded_spacy_models`, it is returned
    directly. Otherwise, the model is loaded from SpaCy, and stored in the
    dictionary (if one was provided).

    Args:
        model_name (str): Name of the SpaCy model to load (e.g., "en_core_web_sm").
        loaded_spacy_models (Optional[Dict[str, Language]]): Optional dictionary
            to cache loaded models. Keys are model names, values are SpaCy
            Language objects.

    Returns:
        Language: Loaded SpaCy model.

    Raises:
        RuntimeError: If the model is not installed or fails to load.
    """
    # Return cached model if available
    if loaded_spacy_models is not None and model_name in loaded_spacy_models:
        return loaded_spacy_models[model_name]

    # Check if package is installed
    if not is_package(model_name):
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download SpaCy model '{model_name}': {e}")

    # Load model
    try:
        nlp_model = spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load SpaCy model '{model_name}': {e}")

    # Store in cache if provided
    if loaded_spacy_models is not None:
        loaded_spacy_models[model_name] = nlp_model

    return nlp_model


def preload_spacy_models(
    field_extractor: "FieldExtractor",  # must be a FieldExtractor child class
    loaded_spacy_models: Optional[Dict[str, "spacy.language.Language"]]
) -> None:
    """
    Preload all SpaCy models and assign them to loaded_spacy_models.

    Args:
        field_extractor: An instance of a FieldExtractor child class.
        loaded_spacy_models: Optional dictionary to cache loaded SpaCy models.

    Behavior:
        - Checks the field_extractor's self.extraction_method.
        - Reads REQUIRED_MODELS from the child class.
        - Loads any SpaCy models not already in loaded_spacy_models.
        - Returns the updated loaded_spacy_models dictionary.
    """
    # Get the extraction_method in provided field_extractor
    method = getattr(field_extractor, "extraction_method", None)
    if method is None:
        return loaded_spacy_models

    # Get list of REQUIRED_MODELS for the current field_extractor based on it's extraction_method
    required_models = getattr(field_extractor, "REQUIRED_MODELS", {}).get(method, {})
    spacy_models = required_models.get("spacy", [])

    # Load the model if it isn't already pre-loaded
    for model_name in spacy_models:
        if model_name not in loaded_spacy_models:
            loaded_spacy_models[model_name] = load_spacy_model(model_name, loaded_spacy_models)