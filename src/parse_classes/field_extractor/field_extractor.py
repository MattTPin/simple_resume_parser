"""field_extractor.py
Holds abstract FileParser class inherited by filetype-specific parsers.
"""
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Union, Any

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError, FieldExtractionConfigError

from src.parse_classes.field_extractor.helper_functions.validate_document_chunk_list import (
    validate_document_chunk_list
)
from src.parse_classes.file_parser.helpers.chunk_text import join_document_chunk_text
from src.parse_classes.field_extractor.helper_functions.llm.llm_helpers import initialize_llm_if_needed

# Define allowed extraction methods (if implemented)
EXTRACTION_METHODS = Literal[
    "regex",
    "ner",
    "llm",
    "rule"
]

class FieldExtractor(ABC):
    """
    Abstract base class for extracting a specific field from a resume.
    Concrete extractors must implement the `extract` method.

    Extraction Methods:
        - regex: Uses regular expressions to identify patterns in text.
        - ner: Uses a Named Entity Recognition model to extract entities.
        - ml: Uses a trained machine learning model for extraction.
        - llm: Uses a large language model (via LangChain) for extraction.
        - rule: Uses simple rule-based logic, keyword matching, or heuristics.
    """
    # Define supported methods and a default method in each subclass (define in each child)
    SUPPORTED_EXTRACTION_METHODS: List[str] = []
    DEFAULT_EXTRACTION_METHOD = None
    
    # Define which models are required to run different extraction methods (define in each child)
    # Retrieved when pre-loading models.
    REQUIRED_ML_MODELS = {}
    # Example:
        # REQUIRED_ML_MODELS = {
        #     "ner": {
        #         "spacy": ["en_core_web_sm"],
        #         "hf": [],
        #     }
        # }
    
    # Define commmon regex queries that might be used in different subclasses
    COMMON_REGEX: dict = {
        # Email address: Covers standardized email format
        "email_address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # Phone Number: Covers common phone formats:
        # -> `+1 123-456-7890`, `(123) 456-7890`, `123-456-7890`, `123.456.7890`, `1234567890`
        "phone_number":  (
            r"(\+?\d{1,3}[\s.-]?)?"           # Optional country code
            r"(\(?\d{3}\)?[\s.-]?)"           # Area code with optional parentheses
            r"\d{3}[\s.-]?\d{4}"              # Local number
        )
    }


    def __init__(
        self,
        document_chunk_list: List[DocumentChunk] = None,
        extraction_method: Optional[EXTRACTION_METHODS] = None,
        llm_client: Optional["LLMClient"] = None,
        force_mock_llm_response: Optional[bool] = False,
        llm_dummy_response: Optional[Any] = None,
        loaded_spacy_models: Optional[Dict[str, "spacy.language.Language"]] = None,
        loaded_hf_models: Optional[Dict[str, "transformers.pipelines.Pipeline"]] = None
    ):
        """
        Args:
            document_chunk_list (Optional[List[DocumentChunk]] | None): Parsed resume chunks.
            extraction_method (EXTRACTION_METHODS | None): Which extraction strategy to use.
                Defaults to the subclass's default method.
            llm_client (LLMClient | None): Pre-initialized LLMClient to use for LLM queries.
            force_mock_llm_response (bool | False): Don't run a live LLM query and return llm_dummy_response
                instead.
            llm_dummy_response (str | None): Force LLM querying tests to return a preset response (for
                testing).
            loaded_spacy_models (Optional[Dict[str, spacy.language.Language]] default = None): Optional
                cache of SpaCy models to use instead of loading from scratch. If not provided then
                models will be loaded every time they are needed.
            loaded_hf_models (Optional[Dict[str, transformers.pipelines.Pipeline]] default = None): Optional
                cache of HuggingFace NER pipelines to use instead of loading from scratch. If not provided
                then models will be loaded every time they are needed.
        """
        self.document_chunk_list: List[DocumentChunk] = document_chunk_list or []
        self.extraction_method = extraction_method
        
        # Hold LLM specific variables
        self.llm_client = llm_client
        self.force_mock_llm_response = force_mock_llm_response
        self.llm_dummy_response = llm_dummy_response
        
        # Use dicts to hold onto previously loaded models
        self.loaded_spacy_models = loaded_spacy_models
        self.loaded_hf_models = loaded_hf_models
        
        # Check that the current extraction method is valid (for subclass)
        self._validate_extraction_method()

    @staticmethod
    def _requires_document_chunks(func):
        """Decorator to ensure `self.document_chunk_list` exists before execution."""
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "document_chunk_list", None):
                raise FieldExtractionConfigError(
                    message = f"{func.__name__} requires self.document_chunk_list to be set before running."
                )
            validate_document_chunk_list(self.document_chunk_list)
            return func(self, *args, **kwargs)
        return wrapper
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "extract" in cls.__dict__:
            cls.extract = cls._requires_document_chunks(cls.extract)
            

    def _validate_extraction_method(self) -> None:
        """
        Validate and set the extraction method for the FieldExtractor instance.

        This method ensures that the specified `extraction_method` is supported
        by the subclass. If no method is provided, it defaults to the class's
        `DEFAULT_EXTRACTION_METHOD`. Additionally, it verifies that a dummy LLM
        response is provided if `force_mock_llm_response` is True.

        Raises:
            NotImplementedError: If `extraction_method` is not in
                `SUPPORTED_EXTRACTION_METHODS`.
            ValueError: If no `SUPPORTED_EXTRACTION_METHODS` are defined in the
                subclass, or if `force_mock_llm_response` is True but
                `llm_dummy_response` is not provided.
        """
        if self.extraction_method:
            # Non acceptable extracted method for specific FieldExtractor Subclass
            if self.extraction_method not in self.SUPPORTED_EXTRACTION_METHODS:
                raise NotImplementedError(
                    f"Unsupported extraction_method '{self.extraction_method}' for {self.__class__.__name__}"
                )
            self.extraction_method = self.extraction_method
        else:
            # No SUPPORTED_EXTRACTION_METHODS defined in subclass
            if not self.SUPPORTED_EXTRACTION_METHODS:
                raise ValueError(f"{self.__class__.__name__} must define SUPPORTED_EXTRACTION_METHODS")
            self.extraction_method = self.DEFAULT_EXTRACTION_METHOD

    @abstractmethod
    def extract(self) -> Optional[str]:
        """
        Extract the field from `document_chunk_list` using the chosen `extraction_method`.

        Raises an error if extraction fails. In production, calls to this method
        should be wrapped in try/except blocks to handle failures gracefully.

        Returns:
            Any: The extracted field value.

        Raises:
            NotImplementedError: If the subclass has not implemented this method.
            FieldExtractionError: If extraction fails despite being implemented (e.g., no value found).
            FieldExtractionConfigError: If extraction fails due to invalid configuration.
            ValueError: If the provided extraction_method is invalid or misconfigured.
            LLMConfigError: If an llm model is to be initiated but has invalid config.
            LLMError: Misc error when running an LLM query.
            LLMInitializationError: Raised when the LLM client fails to initialize
            LLMQueryError: Raised when a query to the LLM fails
            LLMEmptyResponse: Raised when the LLM returns an empty response
        """
        pass
        
    # ----------------------
    # REGEX HANDLING
    # ----------------------
    @_requires_document_chunks
    def _regex_extract_term(
        self,
        pattern: str,
        search_batch_size: int | None = 1
    ) -> str:
        """
        Extract the first match of a given regex pattern from the document.
        
        The document is processed in sequential batches of chunks. Each batch is 
        combined into text and searched for the pattern. Returns the first match 
        found. If `search_batch_size` is None, all chunks are combined and treated as a 
        single batch.

        Args:
            pattern (str): The regex pattern to search for.
            search_batch_size (int | None): Number of DocumentChunk entries to combine per batch.
                If None, all chunks are treated as a single batch.

        Returns:
            str: The first regex match found.

        Raises:
            ValueError: If no match is found in any batch.
        """
        num_chunks = len(self.document_chunk_list)
        search_batch_size = num_chunks if search_batch_size is None else search_batch_size

        # Process in sequential batches
        for start_idx in range(0, num_chunks, search_batch_size):
            batch_chunks = self.document_chunk_list[start_idx : start_idx + search_batch_size]
            text = join_document_chunk_text(chunks=batch_chunks)
            match = re.search(pattern, text)
            if match:
                return match.group(0)  # always return first match

        raise FieldExtractionError(
            "No regex match could be found in any chunk batch. "
            f"pattern: `{pattern}`",
            document_chunk_list=self.document_chunk_list,
        )
    
    def _get_chunks_with_matching_regex(
        self,
        document_chunk_list: List[DocumentChunk],
        patterns: Union[str, List[str]],
        ignore_case: bool = True,
        backward_buffer: int = 0,
        forward_buffer: int = 0
    ) -> List[DocumentChunk]:
        """
        Search through a list of document chunks for matches to one or more regex patterns,
        and return the matching chunks, optionally including surrounding chunks as buffers.

        Args:
            document_chunk_list (List[DocumentChunk]): List of chunks to search through.
            patterns (str | List[str]): Regex pattern or list of patterns to search for. 
                                        Patterns are tried in order until one matches.
            ignore_case (bool): Whether to ignore case in matching. Defaults to True.
            backward_buffer (int): Number of preceding chunks to include with each match.
            forward_buffer (int): Number of following chunks to include with each match.

        Returns:
            List[DocumentChunk]: All chunks where a matching pattern was found, including buffers.
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        flags = re.IGNORECASE if ignore_case else 0
        matching_indices = set()

        # Aggregate matches for all patterns
        for pattern in patterns:
            for i, chunk in enumerate(document_chunk_list):
                if re.search(pattern, chunk.text, flags=flags):
                    start_idx = max(0, i - backward_buffer)
                    end_idx = min(len(document_chunk_list), i + forward_buffer + 1)
                    matching_indices.update(range(start_idx, end_idx))

        if matching_indices:
            # Return chunks in their original order
            return [document_chunk_list[i] for i in sorted(matching_indices)]

        # No matches found
        raise FieldExtractionError(
            message=(
                f"Regex could not find a match for the provided patterns: {patterns} "
                "in any of the provided entries in document_chunk_list."
            ),
            document_chunk_list=self.document_chunk_list,
        )
    
    # ----------------------
    # SPACY HANDLING
    # ----------------------
    @_requires_document_chunks
    def _spacy_search(
        self,
        model_name: str,
        ner_label: Optional[str] = None,
        token_attr: Optional[str] = None,
        search_batch_size: Optional[int] = 1,
    ) -> Optional[str]:
        """
        Run SpaCy extraction on the document text in sequential batches of chunks.
        Runs `load_spacy_model` to either load a model or utilize the `self.loaded_spacy_models`
        cache to retrieve the model if it was pre-loaded.

        Supports:
            1. NER-based search for entities matching `ner_label`.
            2. Rule-based token attribute search using `token_attr`.

        Args:
            ner_label (str | None): The target NER entity label to search for (e.g., "PERSON").
            token_attr (str | None): SpaCy token attribute to check (e.g., "like_email", "like_url").
            search_batch_size (int | None): Number of DocumentChunk entries to combine per batch.
                If None, all chunks are combined and treated as a single batch.
            model_name (str): SpaCy model to load.

        Returns:
            Optional[str]: The first matching entity or token text, or None if not found.

        Notes:
            - Only the first match encountered is returned.
            - Useful for early-exit searches if the desired entity is expected near the top.
        """    
        if not ner_label and not token_attr:
            raise FieldExtractionConfigError(
                message = (
                    f"Either `ner_label` or `token_attr` is required to run a spacy search"
                )
            )
        from src.parse_classes.field_extractor.helper_functions.ml.spacy_loader import load_spacy_model

        # Load the model (either fresh or from self.loaded_spacy_models cache)
        nlp = load_spacy_model(model_name, self.loaded_spacy_models)
        
        # Count chunks and determine batch size for search
        num_chunks = len(self.document_chunk_list)
        search_batch_size = num_chunks if search_batch_size is None else search_batch_size

        # Loop through batched chunks and run model search
        for start_idx in range(0, num_chunks, search_batch_size):
            batch_chunks = self.document_chunk_list[start_idx : start_idx + search_batch_size]
            text = join_document_chunk_text(chunks=batch_chunks)
            doc = nlp(text)

            # NER-based search
            if ner_label:
                for ent in doc.ents:
                    if ent.label_.upper() == ner_label.upper():
                        return ent.text

            # Rule-based token attribute search
            if token_attr:
                for token in doc:
                    if getattr(token, token_attr, False):
                        return token.text

        raise FieldExtractionError(
            message=(
                "No spacy match could be found in any chunk batch. "
                f"model_name: `{model_name}`, ner_label: `{ner_label}`, token_attr: {token_attr}\n"
            ),
            document_chunk_list=self.document_chunk_list,
        )

    # ----------------------
    # HUGGINGFACE HANDLING
    # ----------------------    
    @_requires_document_chunks
    def _hf_search(
        self,
        model_name: str,
        search_batch_size: Optional[int] = 1,
        target_label: Optional[str] = None,
        return_target: Literal['ner_results', 'target_label'] = 'ner_results'
    ) -> Optional[str]:
        """
        Run a HuggingFace NER pipeline over the document chunks and return entity information.

        This function processes the document in sequential batches, passes the text through
        a HuggingFace token-classification (NER) pipeline, and either returns the raw NER results
        or extracts the first entity matching a specified target label. It is intended to be
        generic for any entity type but can be used specifically for extracting names, organizations, etc.

        Args:
            model_name (str): Name of the HuggingFace model to load (resume-optimized models recommended).
            search_batch_size (int | None, default=1): Number of DocumentChunk entries to combine per batch.
                If None, the entire document is processed as a single batch.
            target_label (str | None, default=None): The entity label to extract (e.g., 'PER', 'ORG').
                Required if `return_target` is 'target_label'.
            return_target (Literal['ner_results', 'target_label'], default='ner_results'):
                - 'ner_results': return the raw HuggingFace NER output.
                - 'target_label': return the first entity matching `target_label` as a string.

        Returns:
            Optional[str or list[dict]]: 
                - If `return_target='ner_results'`, returns the full list of NER predictions as produced by the pipeline.
                - If `return_target='target_label'`, returns the first entity text matching `target_label`.
                Returns None if no match is found.

        Raises:
            FieldExtractionError: If no entity matching `target_label` is found in any batch
                when `return_target='target_label'`.
            ValueError: If `return_target='target_label'` is set but `target_label` is None.
        """
        from src.parse_classes.field_extractor.helper_functions.ml.hf_loader import load_hf_model

        if return_target == "target_label" and not target_label:
            raise ValueError("target_label must be provided when return_target='target_label'")

        # Load the model (either fresh or from self.loaded_hf_models cache)
        nlp = load_hf_model(
            model_name=model_name,
            loaded_hf_models=self.loaded_hf_models
        )

        # Count chunks and determine batch size
        num_chunks = len(self.document_chunk_list)
        batch_size = num_chunks if search_batch_size is None else search_batch_size

        # Loop through batches
        for start_idx in range(0, num_chunks, batch_size):
            batch_chunks = self.document_chunk_list[start_idx : start_idx + batch_size]
            text = join_document_chunk_text(chunks=batch_chunks)

            results = nlp(text)

            if return_target == "ner_results":
                return results

            # Accumulate consecutive tokens with the same target_label
            entity_tokens = []
            current_entity = []
            for ent in results:
                label = ent.get('entity_group') or ent.get('entity')
                word = ent.get('word') or ent.get('token')
                if not word or not label:
                    continue
                if label.upper() == target_label.upper():
                    current_entity.append(word)
                else:
                    if current_entity:
                        entity_tokens.append(" ".join(current_entity))
                        current_entity = []
            if current_entity:
                entity_tokens.append(" ".join(current_entity))

            if entity_tokens:
                return entity_tokens[0]

        # Nothing found
        raise FieldExtractionError(
            message=(
                "No HF NER match could be found in any chunk batch. "
                f"model_name: `{model_name}`, target_label: `{target_label}`\n"
            ),
            document_chunk_list=self.document_chunk_list,
        )
    
    # ----------------------
    # LLM HANDLING
    # ----------------------
    def _initiate_llm(self) -> None:
        """
        Initialize and test the LLM client if one isn't already defined

        This method sets up `self.llm_client` with the specified provider and model ID
        only when `self.extraction_method` is set to "llm".
        
        Raises:
            LLMInitializationError: if LLMClient().test_connection() fails with the current settings. 
        """
        # If we're in "mock" mode then don't initiate the client.
        if self.force_mock_llm_response:
            self.llm_client = None
            return
        
        self.llm_client = initialize_llm_if_needed(
            llm_client = self.llm_client,
            extraction_method = self.extraction_method,
        )
    
    @_requires_document_chunks
    def _query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[str]:
        """
        Query the LLM client using the provided system and user prompts.

        Combines the text from `self.document_chunk_list` and passes it to the LLM.
        If `force_mock_llm_response` is True, returns a preset dummy response
        instead of making a live query.

        Args:
            system_prompt (str): The system prompt to guide the LLM's behavior.
            user_prompt (str): The user prompt, which can include `{text}` to inject combined chunk text.

        Returns:
            Optional[str]: The value returned by the LLM, or a dummy response in test mode.
        """
        # Initialize LLM client if needed
        if self.llm_client is None and not self.force_mock_llm_response:
            self._initiate_llm()

        # Combine all chunk texts
        combined_text = "\n".join(chunk.text for chunk in self.document_chunk_list)

        if (
            self.force_mock_llm_response == True
            and not self.llm_dummy_response
        ):
            # Make sure we have dummy response to return if self.force_mock_llm_response
            raise FieldExtractionConfigError(
                f"if self.force_mock_llm_response is True a "
                "`llm_dummy_response` string is required to run `self_query_llm()`"
                f"force_mock_llm_response : `{self.force_mock_llm_response}`"
                f"llm_dummy_response : `{self.llm_dummy_response}`"
            )

        if self.force_mock_llm_response:
            # Return preset dummy response for testing
            result = self.llm_dummy_response
        else:
            # Run live LLM query
            result = self.llm_client.query(
                system_prompt=system_prompt,
                user_prompt=user_prompt.format(text=combined_text),
                temperature=0.0,  # Minimize randomness
                expect_json=True,  # Always expect JSON
            )

        return result
    
    def _verify_llm_dict_output(
        self,
        llm_response: dict,
        field_name: str,
        expected_type: type,
        user_prompt: str = None,
    ) -> None:
        """
        Verify that an LLM response contains the expected field with the correct type.

        Args:
            llm_response (dict): The response returned by the LLM.
            field_name (str): The key we expect in the LLM response.
            expected_type (type): The expected type of the field value.
            user_prompt (str | None): Optional query that was passed to the LLM.

        Raises:
            FieldExtractionError: If the response is invalid or the field is missing/incorrect type.
        """
        if not isinstance(llm_response, dict):
            raise FieldExtractionError(
                field_name=field_name,
                message=(
                    f"LLM did not return a valid JSON (got {type(llm_response)}): `{llm_response}`. "
                    f"LLM user_prompt: `{user_prompt}`"
                )
            )

        if field_name not in llm_response:
            raise FieldExtractionError(
                field_name=field_name,
                message=(
                    f"LLM JSON missing expected field `{field_name}`: `{llm_response}`. "
                    f"LLM user_prompt: `{user_prompt}`"
                ),
            )

        if not isinstance(llm_response[field_name], expected_type):
            raise FieldExtractionError(
                field_name=field_name,
                message=(
                    f"LLM field `{field_name}` has wrong type "
                    f"(expected {expected_type.__name__}, got {type(llm_response[field_name]).__name__}): "
                    f"LLM response: `{llm_response}`,\n"
                    f"LLM user_prompt: `{user_prompt}`"
                )
            )