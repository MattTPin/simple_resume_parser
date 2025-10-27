"""resume_extractor.py
Utilizes FieldExtractor subclasses to extract fields from a list of DocumentChunks
(resume FileParser output).
"""
import sys
import warnings
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from src.logging import LoggerFactory 
from src.models import ResumeData, DocumentChunk
from src.config import SCANNER_DEFAULTS

from src.parse_classes.field_extractor.helper_functions.validate_document_chunk_list import (
    validate_document_chunk_list
)
from src.parse_classes.field_extractor.field_extractor import FieldExtractor

from src.parse_classes.field_extractor.name_extractor import NameExtractor
from src.parse_classes.field_extractor.email_extractor import EmailExtractor
from src.parse_classes.field_extractor.skills_extractor import SkillsExtractor

from src.parse_classes.resume_extractor.helpers.extractor_map import (
    build_default_extractor_map,
)

# Load field extraction specific logger
logger_factory = LoggerFactory()
extractor_failure_logger = logger_factory.get_logger(
    name="extractor_failures",
    logger_type="extractor"
)

class ResumeExtractor:
    """
    Orchestrates extraction of resume fields using configurable field extractors.

    The extractor_map allows multiple "backup" extractors per field. If one
    extractor raises an exception or fails, the next in the list is attempted.

    Supports parallel extraction using threads. By default, extraction runs
    sequentially. max_threads specifies how many fields can be extracted
    concurrently (cannot exceed available threads on current machine).

    Attributes:
        extractor_map (Dict[str, List[FieldExtractor]]):
            Maps field names to extract to a list of extractor instances to try in order
            if the subsequent instance fails.
        max_threads (int): Maximum threads to use for parallel extraction.
    """
    def __init__(
        self,
        document_chunk_list: List[DocumentChunk],
        extractor_map: Optional[Dict[str, List[FieldExtractor]]] = None,
        max_threads: int = SCANNER_DEFAULTS.MAX_THREADS,
        llm_client: Optional["LLMClient"] = None,
        loaded_spacy_models: Optional[Dict[str, "spacy.language.Language"]] = {},
        loaded_hf_models: Optional[Dict[str, "transformers.pipelines.Pipeline"]] = {}
    ):
        """
        Args:
            document_chunk_list (List[DocumentChunk]): Resume chunks (prepared by FileParser)
                to run extractor functions on.
            extractor_map (Optional[Dict[str, List[FieldExtractor]]]): 
                Map of field names to lists of extractor instances. Each list represents
                fallback extractors to try if the previous one fails. Expects keys for
                "name", "email", and "skills". If None, default instances of each extractor
                are created using standard `FieldExtractor` defaults.

                Example:
                    {
                        "name": [NameExtractor(...), BackupNameExtractor(...)],
                        "email": [EmailExtractor(...)],
                        "skills": [SkillsExtractor(...)]
                    }
            max_threads (int): Maximum parallel extraction threads. Defaults to 1.
            llm_client (Optional[LLMClient]): Optional pre-initialized LLM client.
            loaded_spacy_models (Optional[Dict[str, spacy.language.Language]]): Optional
                cache of SpaCy models to share across extractors.
            loaded_hf_models (Optional[Dict[str, transformers.pipelines.Pipeline]]): Optional
                cache of HuggingFace NER pipelines to share across extractors.
        """
        validate_document_chunk_list(document_chunk_list)
        self.document_chunk_list = document_chunk_list
        self.llm_client = llm_client
        self.loaded_spacy_models = loaded_spacy_models if loaded_spacy_models else {}
        self.loaded_hf_models = loaded_hf_models if loaded_hf_models else {}

        if extractor_map is None:
            # Use default extractors if none provided
            extractor_map = build_default_extractor_map(
                document_chunk_list=self.document_chunk_list,
                llm_client=self.llm_client,
                loaded_spacy_models=self.loaded_spacy_models,
                loaded_hf_models=self.loaded_hf_models
            )
            
        # ExtractorMap should've been verified in ResumeParseFramework
        self.extractor_map = extractor_map
        
        # Determine and set max available threads (to parallelize extraction methods)
        self._determine_max_threads(max_threads)
        

    def _determine_max_threads(self, max_threads: int) -> None:
        """
        Validate and set the `self.max_threads` for parallel extraction.

        Ensures that the requested `max_threads` does not exceed:
        - the number of available CPU cores,
        - the number of extraction fields in `self.extractor_map`.

        Args:
            max_threads (int): The requested maximum number of concurrent threads.

        Warnings:
            - Issues a warning if `max_threads` exceeds the available CPU cores.
            - Issues a warning if `max_threads` exceeds the number of extraction fields.
        """
        # Determine allowed max threads
        available_cores = multiprocessing.cpu_count()
        num_fields = len(self.extractor_map) if hasattr(self, "extractor_map") else 1

        if max_threads <= 0:
            warnings.warn(f"Requested max_threads={max_threads} is invalid. Defaulting to 1 thread.")
            max_threads=1

        # Don't exceed available core amount 
        if max_threads > available_cores:
            warnings.warn(
                f"Requested max_threads={max_threads} exceeds available cores "
                f"({available_cores}). Using {available_cores} instead."
            )
            max_threads = available_cores

        # Don't exceed number of fields we're running extractors for
        if max_threads > num_fields:
            warnings.warn(
                f"Requested max_threads={max_threads} exceeds the number of extraction fields "
                f"({num_fields}). Using {num_fields} instead."
            )
            max_threads = num_fields

        self.max_threads = max_threads


    def _extract_field_with_fallback(
        self, 
        field_name: str
    ) -> object:
        """
        Attempt to extract a single field using all configured extractors.

        Extraction is attempted in the order defined in self.extractor_map[field_name].
        - If an extractor succeeds, its value is returned immediately.
        - If all extractors fail, returns the default value from ResumeData.

        Logs extractor failures to the extractor-specific logger, unless running under pytest.

        Args:
            field_name (str): The field to extract (e.g., "name").

        Returns:
            object: Extracted value, or default from ResumeData if all extractors fail.
        """
        extractors = self.extractor_map.get(field_name, [])

        for extractor in extractors:
            try:
                # Ensure extractor has the latest document chunks
                extractor.document_chunk_list = self.document_chunk_list
                result = extractor.extract()
                return result
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                if not any("pytest" in arg for arg in sys.argv):
                    extractor_failure_logger.warning(
                        f"Field '{field_name}' failed in extractor '{type(extractor).__name__}': {str(e)}"
                    )
                # Continue to next extractor

        # If all extraction attempts failed for this field then return ResumeData() default
        default_value = getattr(ResumeData(), field_name)
        return default_value

    def extract(self) -> ResumeData:
        """
        Extract all fields outlined in self.extractor_map and return a ResumeData
        instance.

        Fields can be extracted sequentially (max_threads=1) or in parallel (max_threads>1)
        using ThreadPoolExecutor. Parallelization is limited to a self.max_threads.

        Returns:
            ResumeData: Object containing extracted fields.
        """
        resume_data = ResumeData()

        if self.max_threads == 1:
            # Sequential extraction
            for extraction_field in self.extractor_map:
                setattr(resume_data, extraction_field, self._extract_field_with_fallback(extraction_field))
        elif self.max_threads > 1:
            # Parallel extraction
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit each field extraction task to the thread pool and map each
                # returned Future to its corresponding field name
                future_to_field = {
                    executor.submit(self._extract_field_with_fallback, field): field
                    for field in self.extractor_map
                }
                # Collect results as each Future completes
                for future in as_completed(future_to_field):
                    # Retrieve the field name and store the extraction result.
                    extraction_field = future_to_field[future]
                    setattr(resume_data, extraction_field, future.result())

        return resume_data