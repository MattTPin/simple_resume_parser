"""parse_resume_cli.py
Run ResumeParserFramework from the command like.
Example: `python parse_resume_cli.py path/to/resume.pdf`
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from dataclasses import dataclass, field
from typing import List, Optional
from src.parse_classes.resume_parser_framework import ResumeParserFramework

@dataclass
class ResumeData:
    """
    Stores structured information extracted from a resume.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = field(default_factory=list)


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_resume_cli.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Initialize the parser
    resume_parser_framework = ResumeParserFramework()

    # Parse the resume
    resume_data: ResumeData = resume_parser_framework.parse_resume(file_path)

    # Print the results
    print("Resume Parsing Result:")
    print(f"Name: {resume_data.name}")
    print(f"Email: {resume_data.email}")
    print(f"Skills: {', '.join(resume_data.skills) if resume_data.skills else 'None'}")


if __name__ == "__main__":
    main()
