"""email_extractor.py
Extracts email addresses from parsed resume chunks.
"""
from typing import Optional
import re

from src.parse_classes.file_parser.helpers.chunk_text import join_document_chunk_text
from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.exceptions import FieldExtractionError

class EmailExtractor(FieldExtractor):
    """
    Extracts the candidate's email address from list of DocumetChunks.

    Supports:
        - 'regex': Pattern-based extraction using COMMON_REGEX['email_address'].
        - 'rule': Token-level SpaCy heuristic using `token.like_email`.
    """

    SUPPORTED_EXTRACTION_METHODS = ["regex", "rule"]
    DEFAULT_EXTRACTION_METHOD = "regex"
    
    REQUIRED_ML_MODELS = {
        "rule": {
            "spacy": ["en_core_web_sm"],
            "hf": [],
        }
    }

    def extract(self) -> Optional[str]:
        """
        Extract the email address from the document chunks using the chosen extraction method.

        Returns:
            str: The first detected email address.

        Raises:
            NotImplementedError: If extraction method is unsupported.
            FieldExtractionError: If no email could be extracted.
        """
        if self.extraction_method == "regex":
            email = self._regex_extract()
        elif self.extraction_method == "rule":
            print("Running rule extraction...")
            email = self._rule_extract()
        else:
            raise NotImplementedError(
                f"Extraction method '{self.extraction_method}' is not implemented for EmailExtractor."
            )

        if not email:
            raise FieldExtractionError(
                field_name="email",
                message=(
                    f"Could not extract an email address from the resume using the "
                    f"`{self.extraction_method}` extraction method."
                ),
                document_chunk_list=self.document_chunk_list,
            )
        return email

    def _regex_extract(self) -> str:
        """
        Extract the first email address found in the `self.document_chunk_list` using regex.
        
        This method delegates to `_regex_extract_term()` internally.

        Returns:
            str: The first email address match found.
            
        Raises:
            FieldExtractionError: If no match is found in any batch.
        """
        return self._regex_extract_term(
            pattern=self.COMMON_REGEX['email_address'],
            search_batch_size=3
        )

    def _rule_extract(self) -> str:
        """
        Extract an email address from the resume text using SpaCy's
        rule-based token-level heuristic.

        This method leverages SpaCy's `token.like_email` attribute,
        which identifies tokens resembling valid email addresses.
        relies on `_spacy_search` and searches with a chunk batch
        size of 3 (3 chunks at a time).

        Returns:
            Optional[str]: The first detected email address, or ``None`` if
            no email address is found.
        """
        return self._spacy_search(
            ner_label=None, # Don't run NER
            token_attr="like_email", # Run RULE search
            search_batch_size=3,
            model_name="en_core_web_sm"
        )
