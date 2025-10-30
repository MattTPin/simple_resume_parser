"""
test_spacy_loader.py
Comprehensive tests for load_spacy_model() and preload_spacy_models().
"""

import pytest
import sys
import subprocess
from unittest.mock import MagicMock

from src.test_helpers.dummy_classes import DummyExtractor

from src.parse_classes.field_extractor.helper_functions.ml import spacy_loader
from src.parse_classes.field_extractor.helper_functions.ml.spacy_loader import (
    load_spacy_model,
    preload_spacy_models,
)

SPACY_LOADER_PATH = f"{spacy_loader.__name__}"


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------
@pytest.fixture
def get_dummy_spacy_nlp():
    """Return a fake SpaCy Language-like object."""
    mock_nlp = MagicMock(name="DummySpacyModel")
    mock_nlp.pipe_names = ["tok2vec", "ner"]
    return mock_nlp


@pytest.fixture
def mock_is_spacy_package_installed(mocker):
    """Simulate SpaCy model being installed."""
    return mocker.patch(f"{SPACY_LOADER_PATH}.is_package", return_value=True)


@pytest.fixture
def mock_is_spacy_package_missing(mocker):
    """Simulate SpaCy model NOT being installed."""
    return mocker.patch(f"{SPACY_LOADER_PATH}.is_package", return_value=False)


@pytest.fixture
def mock_spacy_load(mocker, get_dummy_spacy_nlp):
    """Mock spacy.load() to return a fake model."""
    return mocker.patch(f"{SPACY_LOADER_PATH}.spacy.load", return_value=get_dummy_spacy_nlp)


@pytest.fixture
def mock_spacy_subprocess_run(mocker):
    """Mock subprocess.run() so no actual installs occur."""
    return mocker.patch(f"{SPACY_LOADER_PATH}.subprocess.run")


# ---------------------------------------------------------------------
# TESTS — load_spacy_model()
# ---------------------------------------------------------------------
class TestLoadSpacyModel:
    """Test suite for load_spacy_model()."""

    def test_returns_cached_model_if_present(self, get_dummy_spacy_nlp):
        """Should return cached model immediately if already loaded."""
        cache = {"en_core_web_sm": get_dummy_spacy_nlp}
        result = load_spacy_model("en_core_web_sm", cache)
        assert result is get_dummy_spacy_nlp

    def test_loads_existing_model_if_not_cached(
        self, mock_is_spacy_package_installed, mock_spacy_load, get_dummy_spacy_nlp
    ):
        """Should call spacy.load() when model is installed but not cached."""
        cache = {}
        result = load_spacy_model("en_core_web_sm", cache)
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
        assert cache["en_core_web_sm"] is get_dummy_spacy_nlp
        assert result is get_dummy_spacy_nlp

    def test_model_load_fails_raises_runtimeerror(
        self, mock_is_spacy_package_missing, mock_spacy_load, mocker
    ):
        """If model is not installed, raise RuntimeError."""
        mocker.patch("builtins.input", return_value="y")

        # Simulate model not being installed
        mock_is_spacy_package_missing.return_value = False  # Simulate that is_package() returns False

        with pytest.raises(RuntimeError, match="SpaCy model 'en_core_web_sm' is not installed"):
            load_spacy_model("en_core_web_sm", {})

    def test_load_fails_raises_runtimeerror(self, mock_is_spacy_package_installed, mocker):
        """If spacy.load() raises an exception, wrap in RuntimeError."""
        mocker.patch(f"{SPACY_LOADER_PATH}.spacy.load", side_effect=OSError("Bad load"))
        with pytest.raises(RuntimeError, match="Failed to load SpaCy model"):
            load_spacy_model("en_core_web_sm", {})



# ---------------------------------------------------------------------
# TESTS — preload_spacy_models()
# ---------------------------------------------------------------------

class TestPreloadSpacyModels:
    """Test suite for preload_spacy_models()."""

    def test_preloads_all_REQUIRED_ML_MODELS(self, mock_spacy_load, mock_is_spacy_package_installed, get_dummy_spacy_nlp):
        """Ensure all SpaCy models for the extractor's method are loaded."""
        extractor = DummyExtractor(extraction_method="llm")
        cache = {}

        preload_spacy_models(extractor, cache)

        # ✅ Only one model is required by DummyExtractor
        assert "en_core_web_sm" in cache
        assert isinstance(cache["en_core_web_sm"], MagicMock)
        mock_spacy_load.assert_called_once_with("en_core_web_sm")

    def test_skips_models_already_loaded(self, mock_spacy_load, mock_is_spacy_package_installed, get_dummy_spacy_nlp):
        """Should not reload models already present in cache."""
        extractor = DummyExtractor(extraction_method="llm")
        cache = {"en_core_web_sm": get_dummy_spacy_nlp}

        preload_spacy_models(extractor, cache)

        # ✅ No reloading should occur since the model is already cached
        mock_spacy_load.assert_not_called()
        assert cache["en_core_web_sm"] == get_dummy_spacy_nlp

    def test_handles_no_extraction_method(self, mock_spacy_load):
        """If extractor has no extraction_method, function returns silently."""
        extractor = DummyExtractor()
        del extractor.extraction_method  # simulate missing attribute
        cache = {}

        preload_spacy_models(extractor, cache)

        assert cache == {}
        mock_spacy_load.assert_not_called()

    def test_handles_empty_REQUIRED_ML_MODELS_dict(self, mock_spacy_load):
        """If REQUIRED_ML_MODELS empty or missing keys, should not crash."""
        extractor = DummyExtractor()
        extractor.REQUIRED_ML_MODELS = {}
        cache = {}

        preload_spacy_models(extractor, cache)

        assert cache == {}
        mock_spacy_load.assert_not_called()

    def test_load_spacy_model_called_for_each_missing_model(
        self, mocker, mock_spacy_load, mock_is_spacy_package_installed, get_dummy_spacy_nlp
    ):
        """Ensure load_spacy_model() is called once per missing model."""
        extractor = DummyExtractor(extraction_method="llm")

        mock_load_spacy = mocker.patch(
            f"{SPACY_LOADER_PATH}.load_spacy_model",
            side_effect=lambda name, cache: get_dummy_spacy_nlp,
        )

        cache = {}
        preload_spacy_models(extractor, cache)

        # ✅ Only one model expected for DummyExtractor
        assert mock_load_spacy.call_count == 1
        mock_load_spacy.assert_called_with("en_core_web_sm", cache)
        assert "en_core_web_sm" in cache
