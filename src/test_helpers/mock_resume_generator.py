"""parser_output_generator.py
Outputs lists of DocumentChunks that simulate FileParser.parse() outputs.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import copy

from src.models import DocumentChunk
from src.parse_classes.file_parser.helpers.chunk_text import chunk_text

# SET DEFAULT CHUNK SIZE TO TEST WITH
DEFAULT_TEST_CHUNK_SIZE = 500
DEFAULT_CHUNK_ORDER = [
    "contact_info",
    "work_experience",
    "education",
    "projects",
    "skills"
]

# -------------------------------------------------------------------------
# DUMMY RESUME CHUNKS (to construct resumes from)
# -------------------------------------------------------------------------

DUMMY_RESUME_BLOCKS = {
    # ---------------------------------------------------------
    # CONTACT INFO CHUNKS
    # First example is the default used
    # ---------------------------------------------------------
    "contact_info": [
        """{name}  Greater New York  {phone}  {email}  linkedin.com/in/profile1""",

        """{name}
        Greater New York Area | {phone} | {email} | linkedin.com/in/profile2
        """,

        """{name}
        📍 Greater New York - 📞 {phone} - ✉️ {email} - 🔗 linkedin.com/in/profile3
        """,
    ],

    # ---------------------------------------------------------
    # WORK EXPERIENCE CHUNKS
    # ---------------------------------------------------------
    "work_experience": [
        """WORK EXPERIENCE
        Director of Product Management
        {company_name}
        May 2018 - current Colorado Springs, CO
        • Streamlined customer support process by using SysAid for
        ticket management, boosting satisfaction ratings by 27%.
        • Upsold Comcast products and services to 20% of inbound
        callers, contributing to a 7% increase in quarterly sales.
        """,

        """WORK EXPERIENCE
        MARCH 2021 - CURRENT
        Data Scientist | {company_name} | San Diego, CA
        ● Pioneered segmentation in Google Analytics 4, leading to 3 successful campaigns.
        """,

        """Career Summary
        {company_name}
        Four years experience in early childhood development...
        """,
    ],

    # ---------------------------------------------------------
    # EDUCATION CHUNKS
    # ---------------------------------------------------------
    "education": [
        """EDUCATION
        M.S. Computer Science, San Diego State University
        February 2016 - June 2018
        """,

        """EDUCATION
        The Collegiate School – High school diploma
        2020 - current Richmond, VA
        """,

        """EDUCATION
        M.A. English, University of Texas at San Antonio
        January 2021 - May 2023
        """,
    ],

    # ---------------------------------------------------------
    # PROJECTS / ACHIEVEMENTS CHUNKS
    # ---------------------------------------------------------
    "projects": [
        """PROJECTS
        h2oFiltration – Group Member 2021
        • Designed a water filtration system...
        """,

        """PROJECTS
        OCTOBER 2019 - FEBRUARY 2020
        Editor / Cultural Studies / San Antonio, TX
        """,

        """ACHIEVEMENTS
        2022 GREW ANNUAL REVENUE BY 11%
        """,
    ],

    # ---------------------------------------------------------
    # SKILLS CHUNKS (FILLED)
    # ---------------------------------------------------------
    "skills_filled": [
        """SKILLS
        Google Analytics
        Data Cleaning
        SQL
        Custom Funnels
        Power BI
        """,

        """SKILLS
        Pet footwear; Partial Budget; Water Flow Meters
        """,

        """Skills:
        • Zendesk
        • Intercom
        • Skype
        """,
    ],

    # ---------------------------------------------------------
    # SKILLS CHUNKS (FILLABLE)
    # ---------------------------------------------------------
    "skills_fillable": [
        """SKILLS
        {skills}
        """,
        
        """Skills: {skills}""",

        """Expertise - {skills}""",
    ],
}


# -------------------------------------------------------------------------
# MockResumeGenerator INPUT DATA MODELS
# -------------------------------------------------------------------------
@dataclass
class ChunkValues:
    """
    Holds fillable field values that can be overridden when
    generating a mock resume. These represent the variable parts
    of the text templates (e.g., name, email, skills).

    Attributes:
        name: Default person name.
        email: Default email address.
        phone: Default phone number.
        linkedin_name: LinkedIn profile slug or username.
        skills: Default skills text for insertion into the skills chunk.
        company_name: Default company name for work experience.
    """
    name: str = "John Doe"
    email: str = "john.doe@example.com"
    phone: str = "123-456-7890"
    linkedin_name: str = "john_doe23"
    skills: str = DUMMY_RESUME_BLOCKS["skills_filled"][0]
    company_name: str = "Comcast"
    
@dataclass
class ChunkTemplates:
    """
    Defines the text templates used to render each section of the
    resume. Each template can include placeholders compatible with
    Python's `str.format()` syntax, such as `{name}` or `{skills}`.

    Attributes:
        contact_info: Template for the contact information chunk.
        work_experience: Template for the work experience chunk.
        education: Template for the education chunk.
        projects: Template for the projects or achievements chunk.
        skills: Template for the skills chunk.
    """
    contact_info: str = DUMMY_RESUME_BLOCKS["contact_info"][0]
    work_experience: str = DUMMY_RESUME_BLOCKS["work_experience"][0]
    education: str = DUMMY_RESUME_BLOCKS["education"][0]
    projects: str = DUMMY_RESUME_BLOCKS["projects"][0]
    skills: str = DUMMY_RESUME_BLOCKS["skills_fillable"][0]
    other: Optional[str] = None  # optional, only used if provided

# -------------------------------------------------------------------------
# MOCK RESUME GENERATOR
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Main generator class
# -------------------------------------------------------------------------
class MockResumeGenerator:
    """
    Generate realistic mock resumes for testing purposes.

    This class builds a resume from predefined templates and values,
    substitutes fillable fields like name, email, phone, LinkedIn username,
    and skills, and then chunks the resulting text into DocumentChunk objects.

    Attributes:
        chunk_values (ChunkValues): Fillable field values for substitution.
        chunk_templates (ChunkTemplates): Templates for each resume section.
        chunk_order (List[str]): The sequence of sections to include.
        chunk_size (int): Maximum number of characters per chunk.
    """

    def __init__(
        self,
        chunk_values: Optional[ChunkValues] = None,
        chunk_templates: Optional[ChunkTemplates] = None,
        chunk_order: Optional[List[str]] = None,
        chunk_size: Optional[int] = DEFAULT_TEST_CHUNK_SIZE,
    ):
        self.chunk_values = chunk_values or ChunkValues()
        self.chunk_templates = chunk_templates or ChunkTemplates()
        self.chunk_order = DEFAULT_CHUNK_ORDER if chunk_order is None else chunk_order
        self.chunk_size = chunk_size

    # ----------------------
    # Section generators
    # ----------------------
    def _generate_contact_info(self) -> str:
        """Render the contact information section."""
        v = self.chunk_values
        t = self.chunk_templates

        # Ensure linkedin_name is in template
        template_text = t.contact_info
        if "{linkedin_name}" not in template_text:
            # Replace last part with linkedin_name placeholder
            template_text = template_text.rsplit("linkedin.com/in/profile1", 1)[0] + "linkedin.com/in/{linkedin_name}"

        return template_text.format(
            name=v.name,
            email=v.email,
            phone=v.phone,
            linkedin_name=v.linkedin_name,
        )

    def _generate_work_experience(self) -> str:
        """Render the work experience chunk text."""
        v = self.chunk_values
        t = self.chunk_templates
        return t.work_experience.format(company_name=v.company_name)

    def _generate_education(self) -> str:
        """Return the education chunk text."""
        return self.chunk_templates.education

    def _generate_projects(self) -> str:
        """Return the projects chunk text."""
        return self.chunk_templates.projects

    def _generate_skills(self) -> str:
        """Render the skills section."""
        return self.chunk_templates.skills.format(skills=self.chunk_values.skills)

    def _generate_other(self) -> str:
        """Render the optional 'other' chunk if defined."""
        if not self.chunk_templates.other:
            return ""
        # Allow format() to use same value fields if needed
        try:
            return self.chunk_templates.other.format(**vars(self.chunk_values))
        except KeyError:
            # If unknown placeholders are present, just return raw text
            return self.chunk_templates.other

    # ----------------------
    # Public interface
    # ----------------------
    def generate(self) -> List[DocumentChunk]:
        """
        Build the resume and return it as a list of chunked DocumentChunk objects.

        Returns:
            List[DocumentChunk]: Chunked text chunks representing the resume.
        """
        if not self.chunk_order:
            return []
        
        section_map = {
            "contact_info": self._generate_contact_info,
            "work_experience": self._generate_work_experience,
            "education": self._generate_education,
            "projects": self._generate_projects,
            "skills": self._generate_skills,
            "other": self._generate_other,
        }

        assembled_parts = []
        for section in self.chunk_order:
            # Always skip "other" if None
            if section == "other" and not self.chunk_templates.other:
                continue 
            if section in section_map:
                assembled_parts.append(section_map[section]())

        full_text = "\n\n".join(assembled_parts)

        return chunk_text(text=full_text, chunk_size=self.chunk_size)
        
    def clone(
        self,
        chunk_values: Optional["ChunkValues"] = None,
        chunk_templates: Optional["ChunkTemplates"] = None,
        chunk_order: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> "MockResumeGenerator":
        """
        Create a copy of this generator, optionally overriding specific attributes.

        Args:
            chunk_values (Optional[ChunkValues]): Override chunk values.
            chunk_templates (Optional[ChunkTemplates]): Override chunk templates.
            chunk_order (Optional[List[str]]): Override section order.
            chunk_size (Optional[int]): Override chunk size.

        Returns:
            MockResumeGenerator: A new generator with the requested overrides.
        """
        new_gen = copy.deepcopy(self)
        if chunk_values is not None:
            new_gen.chunk_values = chunk_values
        if chunk_templates is not None:
            new_gen.chunk_templates = chunk_templates
        if chunk_order is not None:
            new_gen.chunk_order = chunk_order
        if chunk_size is not None:
            new_gen.chunk_size = chunk_size
        return new_gen
    

