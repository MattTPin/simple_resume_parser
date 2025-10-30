"""
llm_client.py

Universal LangChain-based client for multiple LLM providers.
Supports Anthropic (Claude).
Includes configuration validation, flexible prompting, and optional JSON parsing.
"""
import json
import os
import re
from typing import Optional, Any, Dict, Literal
import warnings

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import SCANNER_DEFAULTS
from src.exceptions import (
    LLMConfigError,
    LLMInitializationError,
    LLMQueryError,
    LLMEmptyResponse
)
from src.test_helpers.llm_client_test_helpers import (
    create_mock_llm_response
)

SUPPORTED_PROVIDERS = ["anthropic"]
load_dotenv()

class LLMClient:
    """
    A flexible, provider-agnostic client for interacting with large language models (LLMs)
    through the LangChain interface. Must be "initialized" using "initialize_client()" function
    before it can be used to make queries.

    Handles:
        - Pulling API keys from .env file
        - Resolving provider, model ID, and API keys
        - Validating configuration
        - Optional test mode with deterministic responses
        - Integration with LangChain clients (Anthropic)

    Initialization uses defaults from `SCANNER_DEFAULTS` if values are not provided.

    Attributes:
        provider (Optional[str]): Name of the LLM provider (e.g., "anthropic"). Defaults
            to SCANNER_DEFAULTS.LLM_PROVIDER.
        model (Optional[str]): Model identifier or name. If none provided then will automatically match
            the provided provider to its SCANNER_DEFAULTS default model name.
        api_key (Optional[str]): Provider-specific API key used for authentication pulled from .env. Will
            automatically match to selected provider.
        function_name (Optional[str]): Name of the function or feature invoking the LLM.
        fallback_message (Optional[str]): Default message returned when the model response is empty.
        test_mode (bool): If True, returns mock responses instead of making real API calls.
        test_response_type (str): Mock response type used in test mode.
        client (Any): Initialized LangChain chat model client.
    
    Raises:
        LLMConfigError: If required environment variables are missing or invalid.
        LLMInitializationError: If the model client cannot be initialized.
        LLMQueryError: If a query fails during execution.

    Example:
        >>> client = LLMClient(provider="anthropic", model="claude-haiku-4-5")
        >>> response, tokens = client.query(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_prompt="Summarize this paragraph.",
        ...     expect_json=False
        ... )
        >>> print(response)
        'Here’s a concise summary of the paragraph...'
    """

    def __init__(
        self,
        provider: Optional[str] = SCANNER_DEFAULTS.LLM_PROVIDER,
        model: Optional[str] = None,
        function_name: Optional[str] = None,
        fallback_message: Optional[str] = None,
        test_mode: Optional[bool] = False,
        test_response_type: Literal["success", "failed", "unexpected_json", "not_json"] = "success",
    ):
        """Initialize an LLMClient instance and resolve provider-specific configuration.

        Args:
            provider (Optional[str], optional): Name of the LLM provider (e.g., "anthropic").
                Defaults to `SCANNER_DEFAULTS.LLM_PROVIDER`.
            model (Optional[str], optional): Model identifier to use for the provider.
                If None, the default model from SCANNER_DEFAULTS will be used.
            function_name (Optional[str], optional): Name of the function or feature invoking the LLM.
                Used for logging or debugging purposes. Defaults to None.
            fallback_message (Optional[str], optional): Message to return if the model response is empty.
                Defaults to None.
            test_mode (Optional[bool], optional): If True, the client will return deterministic mock responses
                instead of querying the live LLM. Defaults to False.
            test_response_type (Literal["success", "failed", "unexpected_json", "not_json"], optional):
                Type of mock response to use when `test_mode` is True. Defaults to "success".
        """
        self.function_name = function_name
        self.fallback_message = fallback_message
        self.test_mode = test_mode
        self.test_response_type = test_response_type

        # --- Resolve configuration ---
        self.provider = provider
        self._resolve_provider()
        
        self._resolve_model(model)
        self._resolve_api_key()

        # Only fill client when `initialize_client()` is run
        self.client = None

    # --- Init helpers ---
    def _resolve_provider(self) -> None:
        """
        Validate that self.provider is valid and supported LLM provider in this class.

        Raises:
            LLMConfigError: If the provider is not one of the supported providers.
        """
        if self.provider not in SUPPORTED_PROVIDERS:
            raise LLMConfigError(
                variable_name="LLM_PROVIDER",
                extra_info=f"Choices are: {SUPPORTED_PROVIDERS}"
            )

    def _resolve_model(self, model: Optional[str]) -> None:
        """
        Resolve and set the model ID for the selected provider.
        - Uses the `model` parameter if provided.
        - Otherwise, falls back to the default model ID from `SCANNER_DEFAULTS`.

        Raises:
            LLMConfigError: If no model ID is provided or available for the selected provider.
        """
        default_models = {
            "anthropic": SCANNER_DEFAULTS.ANTHROPIC_MODEL_ID,
        }

        resolved_model = model or default_models.get(self.provider)
        if not resolved_model:
            raise LLMConfigError(
                variable_name=f"{self.provider}_MODEL_ID",
                message=(
                    f"You must provide a model ID for `{self.provider}` either via SCANNER_DEFAULTS "
                    "or by explicitly passing `model` when initializing LLMClient."
                )
            )

        self.model = resolved_model

    def _resolve_api_key(self) -> None:
        """
        Retrieve and validate the API key for the selected provider from environment variables.
        Does not check if the API key is valid, simply loads it.

        Raises:
            LLMConfigError: If the API key is missing or the provider is invalid.
        """
        api_key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
        }

        key_name = api_key_map.get(self.provider)
        if not key_name:
            raise LLMConfigError(
                variable_name="LLM_PROVIDER",
                message=f"No API key mapping defined for provider `{self.provider}`"
            )

        api_key = os.getenv(key_name)
        if not api_key or api_key == "<REPLACE_ME>":
            raise LLMConfigError(
                variable_name=f"{self.provider}_API_KEY",
                message=(
                    f"You must set a `{self.provider}` API key in your environment variables "
                    "to run LLM queries to their services."
                )
            )

        self.api_key = api_key
    
    def initialize_client(self) -> None:
        """
        Initialize the LangChain chat model client for the selected provider.
        - Imports the provider-specific client dynamically.
        - Initializes the client using the resolved `self.model` and `self.api_key`.
        - Assigns the initialized client to `self.client`.

        Notes:
            - No API call is made during initialization, so this method does not incur costs.
            - For actual connectivity or credential verification, use `test_connection()`.

        Raises:
            LLMConfigError: If the provider is not supported.
            LLMInitializationError: If the client cannot be initialized due to an internal error.

        Example:
            >>> client = LLMClient(provider="anthropic")
            >>> client.initialize_client()
            >>> isinstance(client.client, ChatAnthropic)
            True
        """
        try:
            if self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                self.client = ChatAnthropic(
                    model=self.model,
                    anthropic_api_key=self.api_key,
                    temperature=0.5
                )
            else:
                raise LLMConfigError(
                    variable_name="LLM_PROVIDER",
                    extra_info=f"Unsupported provider: {self.provider}"
                )
        except Exception as e:
            raise LLMInitializationError(
                provider=self.provider,
                model=self.model,
                original_exception=e
            )

    # --- QUERY EXECUTION ---
    def query(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.5,
        expect_json: bool = False,
    ) -> str | dict:
        """
        Perform a model query with flexible configuration.

        Args:
            system_prompt (Optional[str]): Instruction or behavioral setup for the model.
            user_prompt (str): Input text or main query.
            temperature (float): Model creativity level (0.0–1.0).
            expect_json (bool): Whether to parse response as JSON.

        Returns:
            str | dict: dict if `expect_json` is True otherwise str. str may be returned even
                when `expect_json` is True if the LLM does not behave as expected.l
        """
        if not self.client:
            raise LLMInitializationError(provider=self.provider, model=self.model)

        messages = [
            SystemMessage(content=system_prompt) if system_prompt else None,
            HumanMessage(content=user_prompt)
        ]
        messages = [m for m in messages if m]  # Remove None

        try:
            if self.test_mode == False:
                # Query the LLM
                response: AIMessage = self.client.invoke(
                    messages,
                    temperature=temperature
                )
                
            elif self.function_name and self.test_response_type:
                # Return a mock LLM response (for testing)
                response: AIMessage = create_mock_llm_response(
                    function_name=self.function_name,
                    response_type=self.test_response_type,
                    provider=self.provider
                )
            else:
                # Raise incorrect test config error
                raise LLMQueryError(
                    provider=self.provider,
                    model=self.model,
                    additional_message= (
                        "Test mode is enabled without valid test variables having been defined. "
                        f"self.test_mode = {self.test_mode} "
                        f"self.function_name = {self.function_name} "
                    ),
                )
            
            # Raise error if no response
            if not response or not response.content:
                raise LLMEmptyResponse(provider=self.provider, model=self.model)
            
            # Get the result text
            response_content = response.content.strip()
            
            if expect_json:
                try:
                    # Try to parse the json
                    response_content = self._clean_llm_json_response(response_text=response_content)
                except Exception as e:
                    # Warn the user if we're expecting a json response but didn't get one (LLM faliure)
                    warnings.warn(
                        (
                            f"LLM did not return valid JSON when it was expected to. "
                            f"Provider: `{self.provider}` "
                            f"Model: `{self.model}` "
                            f"Function: `{self.function_name}` \n"
                            f"Exception: `{e}` \n"
                            "This may occur if the LLM output was malformed or test mode variables "
                            "were not correctly defined."
                        ),
                        category=UserWarning,
                    )
            
            # Return fallback message if result text is empty
            if not response_content:
                response_content = self.fallback_message or "No query result"

            return response_content

        except Exception as e:
            raise LLMQueryError(provider=self.provider, model=self.model, original_exception=e)
    
    
    def _clean_llm_json_response(self, response_text: str):
        """
        Normalize and parse a JSON string returned by an LLM into a valid Python object.

        This method is designed to handle the common formatting issues that occur when
        large language models (LLMs) return JSON-like data wrapped in Markdown code fences,
        extra whitespace, or stray characters. It attempts to safely extract and load the
        actual JSON structure so it can be programmatically processed.

        The function performs the following steps:
        1. Removes leading and trailing whitespace.
        2. Strips Markdown-style code fences such as ```json ... ``` or ``` ... ```.
        3. Attempts to directly parse the cleaned string as JSON.
        4. If direct parsing fails, uses a regex search to extract the first valid JSON
            object (`{...}`) or array (`[...]`) from the text and parses that.
        5. Raises a `json.JSONDecodeError` if no valid JSON structure can be found.

        Args:
            response_text (str): The raw text response from an LLM that is expected to
                                contain valid JSON data, possibly wrapped in Markdown
                                formatting or other text artifacts.

        Returns:
            Any: The parsed Python object (typically a `dict` or `list`) resulting from
                successful JSON decoding.

        Raises:
            json.JSONDecodeError: If no valid JSON structure can be extracted or parsed
                                from the provided text.
        """
        # Strip leading/trailing whitespace
        text = response_text.strip()

        # Remove any code fences like ```json ... ```
        # Matches ```json ... ``` or ``` ... ``` anywhere in the string
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        
        # Sometimes the model returns JSON arrays or objects as strings; try to parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If it still fails, try to extract the JSON content by searching for '{}' or '[]'
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise  # rethrow if no valid JSON structure was found
    
    # --- TESTING ---
    def test_connection(self) -> bool:
        """
        Run provider-specific connection validation.
        Returns True if successful, False otherwise.
        
        Checks for..
            - Invalid or expired API key
            - Model quota exceeded
            - Rate limit errors
            - Missing client initialization
        
        Returns:
            bool: Whether connection is valid or not
        """
        if not self.client():
            self.initialize_client()
        
        if self.provider == "anthropic":
            return self._test_anthropic_config()
    
        raise LLMInitializationError(
            provider=self.provider,
            model=self.model,
            original_exception="No valid provider selected"
        )
    
    def _test_anthropic_config(self) -> bool:
        """Validate Anthropic connection."""
        return self._test_connection_generic("Anthropic")

    def _test_connection_generic(self, provider_name: str) -> bool:
        """
        Perform a generic connection test for the initialized client. Validates
        that the LLM client can communicate with the provider.

        The test works by invoking a lightweight "ping" or equivalent operation
        on the client. It checks for:
            - Successful connection
            - Invalid or expired API keys
            - Rate limit or quota issues

        Args:
            provider_name (str): Name of the provider being tested.

        Returns:
            bool: True if the client successfully connects, False otherwise.

        Raises:
            LLMInitializationError: 
                - If the client has not been initialized.
                - If the provider returns an error during the test, with additional context
                for quota or rate limits included in `additional_message`.
        """
        if not self.client:
            raise LLMInitializationError(
                provider=provider_name,
                model=self.model,
                original_exception="No client initialized"
            )
        try:
            response = self.client.invoke("ping")
            if response and hasattr(response, "content"):
                return True
        except Exception as e:
            additional_message = ""
            if "insufficient_quota" in str(e).lower():
                additional_message += f"Out of tokens for `{provider_name}`"
            if "rate limit" in str(e).lower():
                additional_message += f"Rate limit reached for `{provider_name}`"
            raise LLMInitializationError(
                provider=provider_name,
                model=self.model,
                original_exception=e,
                additional_message=additional_message
            )
        return False