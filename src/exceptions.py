"""exceptions.py
Defines custom exceptions for this project.
"""
from typing import Optional, List

# ------------------------ Field Parser Errors ------------------------
class FileParserError(Exception):
    """Base exception for file parser errors."""
    pass

class FileNotSupportedError(FileParserError):
    """Raised when the current file path has an unsupported extension."""
    def __init__(
        self,
        extension: str,
        supported_extensions: List[str],
        context: Optional[str] = None
    ):
        self.extension = extension
        self.supported_extensions = supported_extensions
        message = (
            f"File with extension '{extension}' is not supported. "
            f"Supported extensions: {supported_extensions}"
        )
        if context:
            message += f" Context: {context}"
        super().__init__(message)


class NoFilePathError(FileParserError):
    """Raised when no file path is provided but the FileParser attempts to access it."""
    def __init__(self):
        message = (
            "No file path was provided. This FileParser instance "
            "cannot access or parse a file without a valid 'file_path'."
        )
        super().__init__(message)

class FileTooLargeError(FileParserError):
    """Raised when a file exceeds the allowed file size."""
    def __init__(self, max_size: int, actual_size: int):
        super().__init__(
            f"File size is {actual_size} bytes, which exceeds the max allowed {max_size} bytes."
        )
        self.max_size = max_size
        self.actual_size = actual_size

class FileOpenError(FileParserError):
    """Raised when a file cannot be opened or read."""
    def __init__(self, file_path: str, original_error: str):
        super().__init__(
            f"Failed to open or read file: {file_path}. Original error: {original_error}"
        )
        self.file_path = file_path
        self.original_error = original_error
        
class FileEmptyError(FileParserError):
    """Raised when a file contains no parsable text."""
    def __init__(self, file_path: str, message: str | None = None):
        self.file_path = file_path  # store file path as attribute
        if message is None:
            message = f"File `{file_path}` contains no parsable text."
        super().__init__(message)

# ------------------------ Field Extraction Errors ------------------------

class FieldExtractionError(Exception):
    """
    Raised when a FieldExtractor fails to extract a value from a document.

    Attributes:
        field_name (str | None): The name of the field being extracted (optional).
        message (str): Human-readable description of the error.
        document_chunk_list (list[str] | None): Optional list of text chunks
            from the document, useful for debugging extraction failures.
    """

    def __init__(
        self,
        field_name: str | None = None,
        message: str = "Failed to extract field",
        document_chunk_list: list[str] | None = None,
    ):
        self.field_name = field_name
        self.message = message
        self.document_chunk_list = document_chunk_list
        super().__init__(self.full_message)

    @property
    def full_message(self) -> str:
        """Construct the complete error message including optional chunks."""
        base_message = (
            f"{self.message}: {self.field_name}" if self.field_name else self.message
        )
        if self.document_chunk_list:
            chunks_display = "\n".join(
                f"- {chunk}" for chunk in self.document_chunk_list
            )
            base_message += f"\n\nDocument Chunks:\n{chunks_display}"
        return base_message
    
class FieldExtractionConfigError(Exception):
    """
    Raised when a FieldExtractor instance is configured incorrectly.

    Attributes:
        field_name (str | None): The name of the field being extracted (optional).
        message (str): Human-readable description of the error.
    """
    def __init__(self, field_name: str | None = None, message: str = "Failed to extract field"):
        self.field_name = field_name
        self.message = message
        super().__init__(self.full_message)

    @property
    def full_message(self) -> str:
        if self.field_name:
            return f"{self.message}: {self.field_name}"
        return self.message

# ------------------------ ResumeParserFramework Errors ------------------------
class ResumeParserFrameworkError(Exception):
    """Base exception for resume parser framework."""
    pass

class ResumeParserFrameworkConfigError(Exception):
    """
    Raised when the ResumeParserFramework configuration is invalid.
    """
    def __init__(self, message: str):
        super().__init__(f"ResumeParserFrameworkConfigError: {message}")

# ------------------------ Extractor Map Errors ------------------------
class ExtractorMapConfigError(Exception):
    """
    Raised when the extractor_map configuration is invalid.
    Provides a clear message about what went wrong.
    """
    def __init__(self, message: str):
        super().__init__(f"ExtractorMapConfigError: {message}")


# ------------------------ LLM Querying Errors ------------------------
class LLMConfigError(Exception):
    """Raised when a required configuration (in .env by default) for LLMCLient to function 
    is missing or invalid."""

    def __init__(
        self,
        variable_name: str,
        message: str = None,
        extra_info: str = None
    ):
        """
        Args:
            variable_name: Name of the config variable.
            message: Optional custom message for the error.
            extra_info: Additional information to append to the error message.
        """
        if message is None:
            message = f"Missing or invalid configuration: {variable_name}. Please set it in your .env file."
        if extra_info:
            message += f" | {extra_info}"
        super().__init__(message)
        self.variable_name = variable_name
        self.extra_info = extra_info

    def __str__(self):
        return f"[CONFIG ERROR] {super().__str__()} | Variable: {self.variable_name}"

class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.provider = provider
        self.model = model
        self.original_exception = original_exception

        base_msg = message
        if provider:
            base_msg += f" | Provider: {provider}"
        if model:
            base_msg += f" | Model: {model}"
        if original_exception:
            base_msg += f" | Original Exception: {original_exception}"

        super().__init__(base_msg)


class LLMInitializationError(LLMError):
    """Raised when the LLM client fails to initialize."""
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_exception: Optional[Exception] = None,
        additional_message: Optional[str] = None
    ):
        message = "Failed to initialize LLM client"
        if additional_message:
            message += f": {additional_message}" 
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            original_exception=original_exception,
        )


class LLMQueryError(LLMError):
    """Raised when a query to the LLM fails."""
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        additional_message: Optional[str] = None,
        original_exception: Exception = None,
    ):
        message = "LLM query failed"
        if additional_message:
            message += f": {additional_message}" 
        
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            original_exception=original_exception,
        )


class LLMEmptyResponse(LLMError):
    """Raised when the LLM returns an empty response."""
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        super().__init__(
            message="LLM returned an empty response",
            provider=provider,
            model=model,
        )