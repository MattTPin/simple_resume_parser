"""
test_hf_loader.py
Comprehensive tests for load_hf_model() and preload_hf_models().
Mirrors test_spacy_loader.py.
"""

import pytest
import sys
import subprocess
from unittest.mock import MagicMock

from src.test_helpers.dummy_classes import DummyExtractor
from src.parse_classes.field_extractor.helper_functions.ml import hf_loader
from src.parse_classes.field_extractor.helper_functions.ml.hf_loader import (
    load_hf_model,
    preload_hf_models,
)

# ---------------------------------------------------------------------
# AUTO-DERIVED PATHS
# ---------------------------------------------------------------------
HF_LOADER_PATH = f"{hf_loader.__name__}"


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------
@pytest.fixture
def get_dummy_hf_pipeline():
    """Return a fake HuggingFace pipeline object."""
    mock_pipe = MagicMock(name="DummyHFPipeline")
    mock_pipe.model = MagicMock(name="DummyModel")
    mock_pipe.tokenizer = MagicMock(name="DummyTokenizer")
    return mock_pipe


@pytest.fixture
def mock_hf_tokenizer_load(mocker):
    """Mock AutoTokenizer.from_pretrained() for local model check."""
    return mocker.patch(f"{HF_LOADER_PATH}.AutoTokenizer.from_pretrained")


@pytest.fixture
def mock_hf_model_load(mocker):
    """Mock AutoModelForTokenClassification.from_pretrained()."""
    return mocker.patch(f"{HF_LOADER_PATH}.AutoModelForTokenClassification.from_pretrained")


@pytest.fixture
def mock_hf_pipeline(mocker, get_dummy_hf_pipeline):
    """Mock pipeline() to return dummy HuggingFace pipeline."""
    return mocker.patch(f"{HF_LOADER_PATH}.pipeline", return_value=get_dummy_hf_pipeline)


@pytest.fixture
def mock_hf_subprocess_run(mocker):
    """Mock subprocess.run() to prevent real installs."""
    return mocker.patch(f"{HF_LOADER_PATH}.subprocess.run")


# ---------------------------------------------------------------------
# TESTS — load_hf_model()
# ---------------------------------------------------------------------
class TestLoadHFModel:
    """Test suite for load_hf_model()."""

    def test_returns_cached_model_if_present(self, get_dummy_hf_pipeline):
        """Should return cached model immediately if already loaded."""
        cache = {"dslim/bert-base-NER": get_dummy_hf_pipeline}
        result = load_hf_model("dslim/bert-base-NER", cache)
        assert result is get_dummy_hf_pipeline

    def test_loads_existing_model_if_not_cached(
        self, mock_hf_tokenizer_load, mock_hf_model_load, mock_hf_pipeline, get_dummy_hf_pipeline
    ):
        """Should load tokenizer, model, and pipeline when available locally."""
        # local_files_only=True succeeds (no exception)
        mock_hf_tokenizer_load.return_value = MagicMock()

        cache = {}
        result = load_hf_model("dslim/bert-base-NER", cache)

        mock_hf_tokenizer_load.assert_any_call("dslim/bert-base-NER", local_files_only=True)
        mock_hf_model_load._

    def test_installation_fails_raises_runtimeerror(
        self, mock_hf_tokenizer_load, mock_hf_subprocess_run, mock_hf_model_load, mock_hf_pipeline, mocker
    ):
        """If subprocess.run() fails, raise RuntimeError."""
        mock_hf_tokenizer_load.side_effect = Exception("not found")
        mocker.patch("builtins.input", return_value="y")

        mock_hf_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["python", "-m", "transformers", "download", "distilbert-base-uncased"]
        )

        with pytest.raises(RuntimeError, match="Failed to download HuggingFace model"):
            load_hf_model("distilbert-base-uncased", {})

    def test_load_fails_raises_runtimeerror(
        self, mock_hf_tokenizer_load, mock_hf_model_load, mocker
    ):
        """If model loading fails after install, wrap in RuntimeError."""
        mock_hf_tokenizer_load.side_effect = [MagicMock(), MagicMock()]
        mock_hf_model_load.side_effect = OSError("load fail")

        with pytest.raises(RuntimeError, match="Failed to download HuggingFace model"):
            load_hf_model("distilbert-base-uncased", {})


# ---------------------------------------------------------------------
# TESTS — preload_hf_models()
# ---------------------------------------------------------------------
class TestPreloadHFModels:
    """Test suite for preload_hf_models()."""

    def test_preloads_all_REQUIRED_ML_MODELS(self, mocker, get_dummy_hf_pipeline):
        """Ensure all HF models for the extractor's method are loaded."""
        extractor = DummyExtractor(extraction_method="llm")
        cache = {}

        def side_effect(name, cache_dict):
            cache_dict[name] = get_dummy_hf_pipeline
            return get_dummy_hf_pipeline

        mock_load_hf = mocker.patch(
            "src.parse_classes.field_extractor.helper_functions.ml.hf_loader.load_hf_model",
            side_effect=side_effect,
        )

        preload_hf_models(extractor, cache)

        # Use the actual required model from DummyExtractor
        required_model = "dslim/bert-base-NER"
        assert required_model in cache
        assert cache[required_model] is get_dummy_hf_pipeline
        mock_load_hf.assert_called_once_with(required_model, cache)

    def test_skips_models_already_loaded(
        self, mock_hf_tokenizer_load, mock_hf_model_load, mock_hf_pipeline, get_dummy_hf_pipeline
    ):
        """Should not reload models already present in cache."""
        extractor = DummyExtractor(extraction_method="llm")
        cache = {"dslim/bert-base-NER": get_dummy_hf_pipeline}

        preload_hf_models(extractor, cache)

        mock_hf_pipeline.assert_not_called()
        assert cache["dslim/bert-base-NER"] == get_dummy_hf_pipeline

    def test_handles_no_extraction_method(self, mock_hf_pipeline):
        """If extractor has no extraction_method, function returns silently."""
        extractor = DummyExtractor()
        del extractor.extraction_method
        cache = {}

        preload_hf_models(extractor, cache)

        assert cache == {}
        mock_hf_pipeline.assert_not_called()

    def test_handles_empty_REQUIRED_ML_MODELS_dict(self, mock_hf_pipeline):
        """If REQUIRED_ML_MODELS empty or missing keys, should not crash."""
        extractor = DummyExtractor()
        extractor.REQUIRED_ML_MODELS = {}
        cache = {}

        preload_hf_models(extractor, cache)

        assert cache == {}
        mock_hf_pipeline.assert_not_called()

    def test_load_hf_model_called_for_each_missing_model(
        self, mocker, mock_hf_tokenizer_load, mock_hf_model_load, mock_hf_pipeline, get_dummy_hf_pipeline
    ):
        """Ensure load_hf_model() is called once per missing model."""
        extractor = DummyExtractor(extraction_method="llm")
        mock_load_hf = mocker.patch(
            f"{HF_LOADER_PATH}.load_hf_model",
            side_effect=lambda name, cache: get_dummy_hf_pipeline,
        )
        cache = {}

        preload_hf_models(extractor, cache)

        mock_load_hf.assert_called_once_with("dslim/bert-base-NER", cache)
        assert "dslim/bert-base-NER" in cache