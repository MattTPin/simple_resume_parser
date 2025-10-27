"""resume_parse_framework.py
Holds framework to orchestrate operation of FileParser() and ResumeExtractor()
and return ResumeData.
"""

from typing import Optional, Dict, List

from src.config import SCANNER_DEFAULTS
from src.exceptions import ResumeParserFrameworkConfigError
from src.models import ResumeData, DocumentChunk

from src.parse_classes.file_parser.helpers.check_file_extension import check_file_extension
from src.parse_classes.file_parser.pdf_parser import PDFParser
from src.parse_classes.file_parser.word_document_parser import WordDocumentParser

from src.parse_classes.field_extractor.helper_functions.llm.llm_client import LLMClient
from src.parse_classes.field_extractor.helper_functions.llm.llm_helpers import initialize_llm_if_needed
from src.parse_classes.field_extractor.helper_functions.ml.spacy_loader import preload_spacy_models
from src.parse_classes.field_extractor.helper_functions.ml.hf_loader import preload_hf_models
from src.parse_classes.field_extractor.field_extractor import FieldExtractor

from src.parse_classes.resume_extractor.helpers.extractor_map import (
    verify_extractor_map,
    build_default_extractor_map,
    unify_extractor_map_model_references
)
from src.parse_classes.resume_extractor.resume_extractor import ResumeExtractor


class ResumeParserFramework:
    """
    Orchestrates the complete resume parsing process — from raw file to structured data.

    Combines:
        - ``FileParser`` (e.g., :class:`PDFParser`, :class:`WordDocumentParser`)
        - ``ResumeExtractor`` (e.g., :class:`NameExtractor`, :class:`EmailExtractor`)

    This class supports dependency overrides to simplify testing. During tests, 
    you can inject:
        * ``forced_document_chunk_list_output`` — to bypass file parsing.

    Parameters
    ----------
    chunk_size : int, optional
        The maximum number of characters per text chunk when parsing the file. 
        Defaults to ``SCANNER_DEFAULTS.CHUNK_SIZE``.
    max_file_size_mb : float, optional
        The maximum file size (in megabytes) allowed for parsing. 
        Defaults to ``SCANNER_DEFAULTS.MAX_FILE_SIZE_MB``.
    extractor_map : dict[str, list[FieldExtractor]], optional
        A mapping of field names to lists of extractor instances used by the 
        :class:`ResumeExtractor`. Enables multiple strategies for each field.
    llm_client : LLMClient, optional
        A language model client instance (e.g. Anthropic) used by 
        LLM-based extractors.
    loaded_spacy_models : dict[str, spacy.language.Language], optional
        A dictionary of pre-loaded spaCy models keyed by model name. 
        Used to avoid repeated initialization across multiple extractions.
    loaded_hf_models : dict[str, transformers.pipelines.Pipeline], optional
        A dictionary of pre-loaded Hugging Face pipelines keyed by task name. 
        Used for transformer-based extractors.
    forced_document_chunk_list_output : list[DocumentChunk], optional
        When provided, overrides file parsing and directly supplies a list of 
        :class:`DocumentChunk` objects (useful for testing).

    Example
    -------
    >>> framework = ResumeParserFramework()
    >>> resume_data = framework.parse_resume("path/to/resume.pdf")
    """

    FILETYPE_PARSER_MAP = {
        ".pdf": PDFParser,
        ".docx": WordDocumentParser,
    }

    def __init__(
        self,
        chunk_size: int = SCANNER_DEFAULTS.CHUNK_SIZE,
        max_file_size_mb: Optional[float] = SCANNER_DEFAULTS.MAX_FILE_SIZE_MB,
        extractor_map: Optional[Dict[str, List[FieldExtractor]]] = None,
        llm_client: Optional["LLMClient"] = None,
        max_threads: int = SCANNER_DEFAULTS.MAX_THREADS,
        loaded_spacy_models: Optional[Dict[str, "spacy.language.Language"]] = None,
        loaded_hf_models: Optional[Dict[str, "transformers.pipelines.Pipeline"]] = None,
        forced_document_chunk_list_output: Optional[List[DocumentChunk]] = None,
    ):
        """
        Initialize the ResumeParserFramework.

        Args:
            chunk_size (int): Maximum number of characters per text chunk.
            max_file_size_mb (float | None): Maximum allowed file size in MB.
            extractor_map (dict[str, list[FieldExtractor]] | None): Optional map of
                field names to extractor instances.
            max_threads (int): Maximum parallel extraction threads. Defaults to 1.
            llm_client (LLMClient | None): Optional pre-initialized LLM client.
            loaded_spacy_models (dict[str, spacy.language.Language] | None):
                Optional cache of preloaded SpaCy models.
            loaded_hf_models (dict[str, transformers.pipelines.Pipeline] | None):
                Optional cache of preloaded Hugging Face pipelines.
            forced_document_chunk_list_output (list[DocumentChunk] | None):
                When provided, bypasses the file parsing step and uses this list
                as the parsed document chunks (useful for testing).
        """
        # FileParser parameters
        self.chunk_size = chunk_size
        self.max_file_size_mb = max_file_size_mb
        
        # Store references to passed cache/client instances
        self.llm_client = llm_client
        self.loaded_spacy_models = loaded_spacy_models or {}
        self.loaded_hf_models = loaded_hf_models or {}
        
        # ResumeExtractor parameters
        self.max_threads = max_threads
        
        # Setup `self.extractor_map` and `self.llm_client` (default or provided values)
        self._setup_llm_client_and_extractor_map(extractor_map)
        
        # Preload any models and ensure they're shared in all instances
        self._preload_spacy_models()
        self._preload_hf_models()

        # Testing hooks
        self.forced_document_chunk_list_output = forced_document_chunk_list_output

    def _setup_llm_client_and_extractor_map(
        self,
        extractor_map: (Dict[str, List[FieldExtractor]]| None) = None
    ) -> None:
        """
        Initialize or unify the LLM client and extractor map for the framework. Used
        to set `self.extractor_map` and `self.llm_client` in self AND in all
        `FieldExtractor` instances in self.extractor_map

        This method ensures that all `FieldExtractor` instances (in the final
        self.extractor_map) share the same `LLMClient`, `loaded_spacy_models`,
        and `loaded_hf_models` references to provide a cache or pre-initiated client.

        If an extractor_map IS provided then its structure is verified.

        If an extractor_map is NOT provided then one (which uses the default methods in
        all FieldExtractor instances) is created using build_default_extractor_map().
        
        Args:
            extractor_map (Optional[Dict[str, List[FieldExtractor]]]): 
                A mapping of field names to lists of extractor instances. If `None`,
                a default extractor map will be created.

        Returns:
            None
        """
        if extractor_map is not None:
            # Verify that passed extractor_map is valid
            verify_extractor_map(extractor_map)
            self.extractor_map = extractor_map
            
            # Load LLMClient if needed (stored in `self.llm_client`)
            self._preload_llm_client()
            
            # Unify any existing loaded_spacy_models or loaded_hf_models in ALL
            # extractor_map entries so that they're shared.
            unify_extractor_map_model_references(
                extractor_map=extractor_map,
                llm_client=self.llm_client,
                loaded_spacy_models=self.loaded_spacy_models,
                loaded_hf_models=self.loaded_hf_models
            )
        else:
            # Build an extractor_map with unified LLMClient, loaded_spacy_models, and
            # loaded_hf_models references
            extractor_map = build_default_extractor_map(
                llm_client=self.llm_client,
                loaded_spacy_models=self.loaded_spacy_models,
                loaded_hf_models=self.loaded_hf_models
            )
            
            # Store assigned values
            self.extractor_map = extractor_map

    def _preload_llm_client(self) -> None:
        """
        Iterate over all FieldExtractor instances in `self.extractor_map` and
        if any of them are running in `llm` mode then preload a LLMClient() and
        save it to `self.llm_client`

        Returns:
            None
        """
        # If LLMClient is already loaded don't bother loading a new one
        if isinstance(self.llm_client, LLMClient):
            return
        
        for field, extractor_list in self.extractor_map.items():
            for extractor in extractor_list:
                # Only any extractor has a "llm" extraction method load the LLMClient
                if extractor.extraction_method == "llm":
                    self.llm_client = initialize_llm_if_needed(
                        llm_client = self.llm_client
                    )
                    # No reason to load models more than once
                    return


    def _preload_spacy_models(self) -> None:
        """
        Iterate over all FieldExtractor instances in `self.extractor_map` and
        preload required SpaCy models for each. Models are cached in
        `self.loaded_spacy_models` to avoid redundant loads and modified in place.
        Already cached models are not re-loaded.

        Returns:
            None
        """
        for field, extractor_list in self.extractor_map.items():
            for extractor in extractor_list:
                preload_spacy_models(
                    field_extractor=extractor,
                    loaded_spacy_models=self.loaded_spacy_models
                )

    def _preload_hf_models(self) -> None:
        """
        Iterate over all FieldExtractor instances in `self.extractor_map` and
        preload required HuggingFace models for each. Models are cached in
        `self.loaded_hf_models` to avoid redundant loads and modified in place.
        Already cached models are not re-loaded.

        Returns:
            None
        """
        for field, extractor_list in self.extractor_map.items():
            for extractor in extractor_list:
                # Load any HuggingFace models (if required for current config)
                preload_hf_models(
                    field_extractor=extractor,
                    loaded_hf_models=self.loaded_hf_models
                )

    def parse_resume(self, file_path: str) -> ResumeData:
        """
        Full pipeline: parse file → extract structured data → return ``ResumeData``.

        Args:
            file_path (str): Path to the resume file (.pdf or .docx).

        Returns:
            ResumeData: Structured extracted data (name, email, skills, etc.).
        """
        # Run parse function
        document_chunk_list = self._parse_file(file_path)

        # Run ResumeExtractor
        try:
            resume_data: ResumeData = ResumeExtractor(
                document_chunk_list=document_chunk_list,
                extractor_map=self.extractor_map,
                llm_client=self.llm_client,
                max_threads=self.max_threads,
                loaded_spacy_models=self.loaded_spacy_models,
                loaded_hf_models=self.loaded_hf_models,
            ).extract()
        except:
            # If extraction fails then simply return an empty ResumeData
            resume_data = ResumeData()

        return resume_data

    def _parse_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Internal helper that selects and executes the appropriate ``FileParser`` subclass.

        Determines the correct parser based on the file extension (e.g., .pdf, .docx),
        initializes it with the framework's parsing configuration, and returns the
        resulting list of ``DocumentChunk`` objects.
        
        If ``self.forced_document_chunk_list_output`` exists then the output of the
        file parser is overwritten with the explicitly provided output

        Args:
            file_path (str): Path to the resume file to be parsed.

        Returns:
            list[DocumentChunk]: A list of structured text chunks extracted from
            the input file.

        Raises:
            FileNotSupportedError: From check_file_extension if supported filetype
                not in self.FILETYPE_PARSER_MAP
            ResumeParserFrameworkConfigError: If no compatible parser is found for
                the file type.
        """
        # Validate file extension is supported
        ext = check_file_extension(
            file_path=file_path,
            supported_extensions=self.FILETYPE_PARSER_MAP.keys()
        )
        parser_class = self.FILETYPE_PARSER_MAP.get(ext)
        if parser_class is None:
            raise ResumeParserFrameworkConfigError(
                message=(
                    f"Invalid extension '{ext}'. "
                    f"Supported extensions are: {list(self.FILETYPE_PARSER_MAP.keys())}. "
                    "Ensure that FILETYPE_PARSER_MAP in ResumeParserFramework contains your extension "
                    "and a matching FileParser, e.g., {'.pdf': PDFParser}."
                )
            )

        # Setup Parse function
        parser = parser_class(
            file_path=file_path,
            chunk_size=self.chunk_size,
            max_file_size_mb=self.max_file_size_mb,
        )
        
        if self.forced_document_chunk_list_output is not None:
            # Return (fake) response for testing
            document_chunk_list = self.forced_document_chunk_list_output
        else:
            # Run actual parse function
            document_chunk_list: List[DocumentChunk] = parser.parse()
            
        return document_chunk_list
