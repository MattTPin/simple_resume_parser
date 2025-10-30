"""skills_extractor.py
Extracts skills from parsed resume chunks.
"""
from typing import Optional, List
import re

from src.models import DocumentChunk

from src.parse_classes.file_parser.helpers.chunk_text import join_document_chunk_text
from src.parse_classes.field_extractor.field_extractor import FieldExtractor
from src.exceptions import FieldExtractionError

class SkillsExtractor(FieldExtractor):
    """
    Extracts the candidate's skills from list of DocumetChunks.

    Supports:
        - 'llm': Pass combined DocumetChunks to an LLM and have it isolate skills. 
    """

    SUPPORTED_EXTRACTION_METHODS = ["llm"]
    DEFAULT_EXTRACTION_METHOD = "llm"
    
    # No required ML models
    REQUIRED_ML_MODELS = {}
    
    # Regex to search for "skills" section with
    SKILLS_REGEX = [
        r"\bskills\b",
        r"\btools\b",
        r"\bexpertise\b",
        r"\btechnologies\b",
        r"\bstrengths\b",
        r"\bsoftware\b",
        r"\bprogramming languages\b",
    ]

    def extract(self) -> List[str]:
        """
        Extract the skills from the document chunks using the chosen extraction method.

        Returns:
            List[str]: List of skills in the resume. Will be an empty list if no skills
            could be found.

        Raises:
            NotImplementedError: If extraction method is unsupported.
            FieldExtractionError: If no skills could be extracted.
        """
        if self.extraction_method == "llm":
            skills = self._llm_extract()
        else:
            raise NotImplementedError(
                f"Extraction method '{self.extraction_method}' is not implemented for SkillsExtractor."
            )

        if not skills or type(skills) is not list:
            # return empty list
            return list()
            
        return skills

    def _find_skills_text(self) -> List[DocumentChunk]:
        """
        Performs a regex-based search over self.document_chunk_list using
        self.SKILLS_REGEX to locate chunks that likely represent skills
        headings or content.
        
        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects representing
                the consolidated skills-related text chunks.
        """
        skills_chunk_list = self._get_chunks_with_matching_regex(
            document_chunk_list=self.document_chunk_list,
            patterns=self.SKILLS_REGEX,
            ignore_case=True,
            backward_buffer=1,
            forward_buffer=1
        )
        
        # Join skills into one text string
        return join_document_chunk_text(
            chunks=skills_chunk_list,
            char_limit=3000,
        )

    def _llm_extract(self) -> list:
        """
        Extract skills from a resume section using an LLM.
        A "skills" (or skills-like) portion of a resume (using regex) and then
        queries an LLM to try and extract the skills held within it.

        Returns:
            list[str]: The list of extracted skills (may be empty). On a
            successful extraction the method returns the value of the "skills"
            key from the LLM's JSON response. If the method's declared annotation
            differs, callers should expect a list when extraction succeeds.
        Raises:
            FieldExtractionError: Raised when:
                - No skills section can be found in the resume.
                - The LLM response is not a dict/JSON object.
                - The LLM JSON does not contain the required "skills" key.
            Each error includes contextual information (e.g. the LLM response or its type).
        Behavior and expectations:
            - Uses a regex-based helper to find relevant resume text to pass to the LLM.
            - Instructs the LLM to return only a strict JSON object of the form:
                {"skills": ["skill1", "skill2", ...]}
              or {"skills": []} when no skills are found.
            - Validates the LLM output type and presence of the "skills" field before
              returning the extracted list.
        """
        # First: Use regex to pull out chunks (and their surrounding chunks) that
        # contain a match for "skills" regex
        skills_text: str = self._find_skills_text()
        
        if not skills_text:
            raise FieldExtractionError(
                field_name="skills",
                message="No 'skills' section could be found.",
                document_chunk_list=self.document_chunk_list,
            )
        
        # Setup prompt to extract the skills as a list
        llm_system_prompt = (
            "You are a resume parsing AI. Your task is to extract all skills from the portion "
            "of a resume provided to you. \n\n"
            "Instructions:\n"
            "1. Extract only information from a skills section or a skills-like section (i.e. 'strengths').\n"
            "2. Extract only bullet-point or list-like skills ignoring and categories (i.e. 'programming languages').\n"
            "3. Do not repeat skills in your output.\n"
            "4. Copy the skills exactly as they appear in the resume (e.g. don't alter case).\n"
            "5. Return your output strictly as a valid JSON object.\n"
            "6. Use the following format:\n\n"
            "{\n"
            "  \"skills\": [\"python\", \"front crawl\", \"Team Player\"]\n"
            "}\n\n"
            "4. If no skills are found, return an empty JSON object:\n\n"
            "{\n"
            "  \"skills\": []\n"
            "}\n\n"
            "Do not include any additional text, explanations, or formatting outside the JSON."
        )
        
        # Query the LLM
        user_prompt = f"Resume Portion: {skills_text}"
        llm_response = self._query_llm(
            system_prompt = llm_system_prompt,
            user_prompt = f"Resume Portion: {skills_text}",
        )
        
        # Validate that the LLM response contains the fields we are looking for.
        self._verify_llm_dict_output(
            llm_response=llm_response,
            field_name="skills",
            expected_type=list,
            user_prompt=user_prompt
        )
            
        return llm_response['skills']