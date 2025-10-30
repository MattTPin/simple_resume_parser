"""name_extractor.py
Utilizes abstract FileParser to parse a name out of a parsed resume output.
"""
import re
from typing import Optional, List, Dict, Any

from src.models import DocumentChunk
from src.exceptions import FieldExtractionError

from src.parse_classes.file_parser.helpers.chunk_text import join_document_chunk_text
from src.parse_classes.field_extractor.field_extractor import FieldExtractor


class NameExtractor(FieldExtractor):
    """
    Extracts the candidate's name from list of DocumetChunks.

    Supports:
        - 'ner': SpaCy Named Entity Recognition (PERSON entity).
        - 'llm': (Optional) LLM-based extraction.
    """

    SUPPORTED_EXTRACTION_METHODS = ["ner", "llm"]
    DEFAULT_EXTRACTION_METHOD = "ner"
    
    REQUIRED_ML_MODELS = {
        "ner": {
            "spacy": [],
            "hf": ["dslim/bert-base-NER"],
        }
    }

    def extract(self) -> Optional[str]:
        """
        Extract the name from the document chunks using the chosen extraction method.
        Currently supports NER and LLM extraction methods.
        """
        if self.extraction_method in ["llm"]:
            name = self._llm_extract()
        elif self.extraction_method == "ner":
            name = self._ner_extract()
        else:
            raise NotImplementedError(
                f"Extraction method '{self.extraction_method}' is not implemented for NameExtractor."
            )
        if not name:
            raise FieldExtractionError(
                field_name="name",
                message="Could not extract a PERSON entity from the resume",
                document_chunk_list=self.document_chunk_list,
            )

        return name
    
    
    def _llm_extract(self) -> str:
        """
        Works by searching for a chunk in the resume with a phone number of email address
        (since those are typically near the name) and then passing the matching chunk(s) to
        an LLM and asking it isolate the name.

        Returns:
            str: the name in the resume (if one could be found)
        """
        # Get chunks of text containing the phone number (likely next to the name)
        user_prompt_text = self._find_chunks_with_phone_number()
        if not user_prompt_text:
            # use first 2000 characters of all text as backup if phone number can't be found
            # (that's where you usually find name)
            user_prompt_text = join_document_chunk_text(
                chunks=self.document_chunk_list,
                char_limit=2000,
            )
        
        llm_system_prompt = (
            "You are a resume parsing AI. Your task is to extract the candidate's full name "
            "from the portion of a resume provided to you.\n\n"
            "Instructions:\n"
            "1. Extract only the full name of the candidate.\n"
            "2. Do not infer, guess, or split the name incorrectly.\n"
            "3. Return the name exactly as it appears in the resume.\n"
            "4. Return your output strictly as a valid JSON object.\n"
            "5. Use the following format:\n\n"
            "{\n"
            "  \"full_name\": \"John Doe\"\n"
            "}\n\n"
            "6. If no name is found, return an empty JSON object:\n\n"
            "{\n"
            "  \"full_name\": \"\"\n"
            "}\n\n"
            "Do not include any additional text, explanations, or formatting outside the JSON."
        )

        # Use the base class helper to query the LLM
        llm_response = self._query_llm(
            system_prompt=llm_system_prompt,
            user_prompt=user_prompt_text,
        )
        
        # Validate that the LLM response contains the fields we are looking for.
        self._verify_llm_dict_output(
            llm_response=llm_response,
            field_name="full_name",
            expected_type=str,
            user_prompt=user_prompt_text
        )
        
        return llm_response['full_name']
    
    def _find_chunks_with_phone_number(self) -> str | None:
        """
        Performs a regex-based search over self.document_chunk_list using
        self.COMMON_REGEX['phone_number'] to locate chunks that that likely
        contain phone numbers.
        
        Returns:
            str: Text containing an email address
            None: If no regex matching document chunks can be found
        """
        try:
            email_chunk_list = self._get_chunks_with_matching_regex(
                document_chunk_list=self.document_chunk_list,
                patterns=self.COMMON_REGEX['phone_number'],
                ignore_case=True,
                backward_buffer=2,
                forward_buffer=2
            )
        except:
            # If no matches are found, return None
            return None
        
        # Join skills into one text string
        return join_document_chunk_text(
            chunks=email_chunk_list,
            char_limit=3000,
        )
    
    
    # -------------------
    # NER extraction
    # -------------------
    
    def _ner_extract(self) -> Optional[str]:
        """
        Extract a candidate's name from resume text using a HuggingFace NER model.

        Uses the 'dslim/bert-base-NER' model to identify PERSON entities and applies
        post-processing heuristics to merge subword tokens and select a single
        best candidate.

        Returns:
            Optional[str]: The first detected full name, or None if no valid
            name is found.
        """
        ner_results = self._hf_search(
            model_name="dslim/bert-base-NER",
            target_label="PER",
            search_batch_size=1,
            return_target="ner_results"
        )

        name_search_result = self._merge_and_select_bert_base_NER_person(
            results=ner_results,
            text=join_document_chunk_text(self.document_chunk_list),
            target_label="PER"
        )

        # Only return if it's more than one word and long enough
        if (
            name_search_result and len(name_search_result.split()) > 1
            and len(name_search_result) > 4
        ):
            return name_search_result
                
        return None

    # -------------------
    # NER extraction helper functions
    # -------------------
    def _merge_and_select_bert_base_NER_person(
        self,
        results: List[Dict[str, Any]],
        text: str,
        target_label: str = "PER",
        max_gap: int = 3
    ) -> Optional[str]:
        """
        Post-process HuggingFace NER output to extract the (our best guess)
        at the first PERSON entity as a candidate full name.

        Merges subword tokens, applies heuristics to expand single-word candidates,
        and selects the first candidate in the text.

        Args:
            results (List[Dict[str, Any]]): Output from HuggingFace NER pipeline.
            text (str): Original text to extract the name from.
            target_label (str, optional): Entity label to look for. Defaults to 'PER'.
            max_gap (int, optional): Maximum allowed character gap between consecutive
                PERSON tokens to merge. Defaults to 3.

        Returns:
            Optional[str]: The first full name candidate found, or None if no candidate exists.
        """
        candidates: List[tuple[str, float]] = []
        current_tokens: List[str] = []
        current_scores: List[float] = []
        last_end: Optional[int] = None

        for ent in results:
            label = ent.get('entity_group') or ent.get('entity')
            word = ent.get('word') or ent.get('token')
            score = ent.get('score', 1.0)
            start = ent.get('start')
            end = ent.get('end')

            if not word or not label or start is None or end is None:
                continue

            # Clean subword marker
            word_clean = word.replace("##", "").strip()

            if label.upper() == target_label.upper():
                # Start a new candidate if gap is too large
                if last_end is not None and start - last_end > max_gap:
                    if current_tokens:
                        merged_name = " ".join(current_tokens)
                        avg_score = sum(current_scores) / len(current_scores)
                        candidates.append((merged_name, avg_score))
                        current_tokens, current_scores = [], []

                # Merge subword tokens: if it starts with ##, append to previous token
                if word.startswith("##") and current_tokens:
                    current_tokens[-1] += word_clean
                else:
                    current_tokens.append(word_clean)

                current_scores.append(score)
                last_end = end
            else:
                # End current candidate if non-target encountered
                if current_tokens:
                    merged_name = " ".join(current_tokens)
                    avg_score = sum(current_scores) / len(current_scores)
                    candidates.append((merged_name, avg_score))
                    current_tokens, current_scores = [], []
                last_end = None

        # Add last candidate if exists
        if current_tokens:
            merged_name = " ".join(current_tokens)
            avg_score = sum(current_scores) / len(current_scores)
            candidates.append((merged_name, avg_score))

        if not candidates:
            return None

        # Always take the first candidate
        top_candidate = candidates[0][0]

        # Expand if the last token looks like a partial word (does not end with full letters)
        top_candidate = self._expand_name_from_text(candidate=top_candidate, text=text)
        return top_candidate

    def _expand_name_from_text(self, candidate: str, text: str) -> str:
        """
        Expand a candidate name by finding it in the text and appending
        characters immediately following it until a whitespace, tab, or newline
        is reached.

        This method only extends the *partial last token* in the candidate.
        It does NOT cross whitespace boundaries.

        Example:
            candidate = "John D"
            text = "John Doe is a great engineer"
            -> returns "John Doe"

            candidate = "John"
            text = "John Doe is a great engineer"
            -> returns "John" (stop at space)

        Args:
            candidate (str): The initial candidate name (partial or single word).
            text (str): The full text to search within.

        Returns:
            str: The expanded candidate name if possible, otherwise the original candidate.
        """
        # Escape candidate for regex
        escaped_candidate = re.escape(candidate)

        # Find candidate in text
        match = re.search(escaped_candidate, text)
        if match:
            start_idx = match.start()
            end_idx = match.end()

            # Append characters until whitespace, tab, or newline
            while end_idx < len(text) and not text[end_idx].isspace():
                end_idx += 1

            expanded_candidate = text[start_idx:end_idx]
            return expanded_candidate.strip()

        # fallback: candidate not found in text
        return candidate