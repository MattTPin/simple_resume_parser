"""models.py
Holds standardized data models used across various functions.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of text extracted from a parsed resume.
    Typically stored in a list to preserve the order of chunks.

    Attributes:
        chunk_index (int): Index of the chunk within the resume, 
            counting sequentially. Starts at 1.
        text (str): Text content of the chunk.
    """
    chunk_index: int
    text: str
    

@dataclass
class ResumeData:
    """
    Stores structured information extracted from a resume.

    Attributes:
        name (Optional[str]): Full name of the individual represented by the resume.
        email (Optional[str]): Email address of the individual.
        skills (Optional[List[str]]): List of skills listed in the resume's skills section.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    
    # ---- STUBS FOR FUTURE FEATURES ----
    # phone_number: Optional[str] = None
    # location: Optional[str] = None
    # linkedin: Optional[str] = None
    # github: Optional[str] = None
    # languages: List[str] = field(default_factory=list)
    # summary: Optional[str] = None
    
    # work_experience: List[Dict[str, Optional[str]]] = field(default_factory=list)
    # projects: List[Dict[str, Optional[str]]] = field(default_factory=list)
    # ...

