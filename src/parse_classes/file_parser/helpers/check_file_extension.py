"""check_file_extension.py
Checks file extension and confirms that it's supported by the resume extractor.
"""

import os

from src.exceptions import FileNotSupportedError

def check_file_extension(file_path: str, supported_extensions: list[str]) -> str:
    """
    Validate and return the lowercase file extension for a given file path.
    Supports multi-dot extensions like '.tar.gz'.
    """
    file_name = os.path.basename(file_path).lower()
    
    # Try to match the longest supported extension
    for ext in sorted(supported_extensions, key=len, reverse=True):
        if file_name.endswith(ext.lower()):
            return ext.lower()
    
    # If none matched
    ext = os.path.splitext(file_name)[1]
    raise FileNotSupportedError(
        extension=ext,
        supported_extensions=supported_extensions,
        context="Failed in check_file_extension() call."
    )
