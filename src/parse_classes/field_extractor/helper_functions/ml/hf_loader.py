"""
hf_loader.py
Used to load HuggingFace models with optional cache.
"""

from typing import Dict, Optional, Union, Tuple
import subprocess
import sys

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, logging
from transformers.pipelines import Pipeline, AggregationStrategy

# Suppress HuggingFace Warnings 
logging.set_verbosity_error()

def load_hf_model(
    model_name: str,
    loaded_hf_models: Optional[Dict[str, Pipeline]] = None
) -> Pipeline:
    """
    Load a HuggingFace NER pipeline, optionally using a provided cache dictionary.

    If the model is already present in `loaded_hf_models`, it is returned directly.
    Otherwise, it will automatically download from HuggingFace Hub if not found locally.

    Args:
        model_name (str): Name of the HuggingFace model to load.
        loaded_hf_models (Optional[Dict[str, Pipeline]], default=None): Optional dictionary
            to cache loaded models. Keys are model names, values are NER pipelines.

    Returns:
        Pipeline: HuggingFace NER pipeline with grouped entities.

    Raises:
        RuntimeError: If the model fails to load or download.
    """
    # Return cached model if already loaded
    if loaded_hf_models is not None and model_name in loaded_hf_models:
        return loaded_hf_models[model_name]

    try:
        # Try local load first
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
    except Exception:
        # Fallback to automatic download
        try:
            print(f"HuggingFace model '{model_name}' not found locally. Downloading from Hub...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download HuggingFace model '{model_name}': {e}"
            )

    # Build the pipeline
    try:
        nlp_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build HuggingFace pipeline for '{model_name}': {e}")

    # Cache for reuse
    if loaded_hf_models is not None:
        loaded_hf_models[model_name] = nlp_pipeline

    return nlp_pipeline


def preload_hf_models(
    field_extractor: "FieldExtractor",
    loaded_hf_models: Dict[str, "transformers.pipelines.Pipeline"]
) -> None:
    """
    Preload all HuggingFace models required for a FieldExtractor to run with its
    current extraction_method and assign them to loaded_hf_models.
        - Checks the field_extractor's self.extraction_method.
        - Reads REQUIRED_MODELS from the child class for HuggingFace model names.
        - Loads any HuggingFace models not already in loaded_hf_models.
        - Updates the provided loaded_hf_models dictionary.

    Args:
        field_extractor (FieldExtractor): An instance of a FieldExtractor child class.
        loaded_hf_models (Optional[Dict[str, "transformers.pipelines.Pipeline"]]): Optional
            dictionary to cache loaded HuggingFace models.
    """
    # Get the extraction_method in provided field_extractor
    method = getattr(field_extractor, "extraction_method", None)
    if method is None:
        return loaded_hf_models

    # Get list of REQUIRED_MODELS for the current field_extractor based on it's extraction_method
    required_models = getattr(field_extractor, "REQUIRED_MODELS", {}).get(method, {})
    hf_models = required_models.get("hf", [])

    # Load any required HuggingFace models (if not already in cache)
    for model_name in hf_models:
        if model_name not in loaded_hf_models:
            loaded_hf_models[model_name] = load_hf_model(model_name, loaded_hf_models)