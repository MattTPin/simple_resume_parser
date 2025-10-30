"""conftest.py
Add command line parsing to pytest.
"""

import pytest
from src.logging import LoggerFactory
from src.conftest_helpers import apply_mock_llm_patch

# --------------------------------------------------------------
# SETUP TEST LOGGING
# --------------------------------------------------------------

# Integrate logger with pytest
logger = LoggerFactory().get_logger(
    name="pytest_logger",
    logger_type="pytest",
    console=True
)
current_class = None

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Session start header."""
    logger.info("==== PYTEST SESSION START ====")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logstart(nodeid, location):
    """Called at the start of each test."""
    global current_class
    class_name = location[0]
    if class_name != current_class:
        current_class = class_name
        logger.info(f"\n---- TestClass: {current_class} ----")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Called at the end of each test phase (setup/call/teardown)."""
    if report.when != "call":
        return  # only care about the main call, not setup/teardown

    status = report.outcome.upper()  # PASSED / FAILED / SKIPPED
    if status == "PASSED":
        logger.info(f"PASSED: {report.nodeid}")
    elif status == "FAILED":
        logger.error(f"FAILED: {report.nodeid}\n{report.longreprtext}")
    elif status == "SKIPPED":
        logger.warning(f"SKIPPED: {report.nodeid}\n{report.longreprtext}")

@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Session finish footer."""
    logger.info(f"==== PYTEST SESSION END: exitstatus={exitstatus} ====")


# --------------------------------------------------------------
# SETUP RUN ARGUMENT PARSING
# --------------------------------------------------------------
def pytest_addoption(parser):
    """
    Register a command-line option for selecting the LLM test mode.

    Example usage:
        # Run tests with stubbed LLM responses (default)
        pytest

        # Run tests that make a small number of live LLM calls
        pytest --llm-mode=basic_only

        # Run all live LLM tests
        pytest --llm-mode=full
    """
    parser.addoption(
        "--llm-mode",
        action="store",
        default="mock_only",
        choices=["mock_only", "basic_only", "full"],
        help=(
            "Set the LLM test mode for pytest. Options:\n"
            "  'mock_only' (default): Use stubbed/mock responses.\n"
            "  'basic_only': Run a minimal subset of live LLM tests.\n"
            "  'full': Run all live LLM tests."
        ),
    )

@pytest.fixture(scope="session")
def LLM_TEST_MODE(request):
    """
    Sets LLM test mode detected in pytest_addoption to LLM_TEST_MODE
    fixture variable.

    Returns:
        str: One of 'mock_only', 'basic_only', 'full'.

    Usage in tests:
        def test_example(LLM_TEST_MODE):
            if LLM_TEST_MODE == "mock_only":
                # Use stubs
                ...
            elif LLM_TEST_MODE == "basic_only":
                # Run minimal live queries
                ...
            elif LLM_TEST_MODE == "full":
                # Run all live tests
                ...
    """
    return request.config.getoption("--llm-mode")



@pytest.fixture(autouse=False)
def FORCE_MOCK_LLM_RESPONSES(monkeypatch):
    """
    Always enable mock LLM responses for FieldExtractor subclasses.

    Scope:
      - Per test function or per test class (via `@pytest.mark.usefixtures`).

    Usage:
      - Class-level:
        ```
        @pytest.mark.usefixtures("FORCE_MOCK_LLM_RESPONSES")
        class TestResumeExtractor:
            ...
        ```
      - Function-level:
        ```
        def test_example(FORCE_MOCK_LLM_RESPONSES):
            ...
        ```

    This fixture applies the patch **unconditionally** and is ideal for
    files/tests that should never make live LLM calls.
    """
    apply_mock_llm_patch(monkeypatch)
    yield


@pytest.fixture
def USE_MOCK_LLM_RESPONSE_SETTING(monkeypatch, LLM_TEST_MODE):
    """
    Conditionally enable mock LLM responses based on `LLM_TEST_MODE`.

    Behavior:
      - Applies `apply_mock_llm_patch` only if `LLM_TEST_MODE == "full"`.
      - Otherwise, extractors retain their default behavior.

    Scope:
      - Typically used at the **function level**, but can be combined with class-level tests.

    Usage example:
        - Class-level:
            ```
            @pytest.mark.usefixtures("USE_MOCK_LLM_RESPONSE_SETTING")
            class TestResumeExtractor:
                ...
            ```
        - Function-level:
            ```
            def test_example(USE_MOCK_LLM_RESPONSE_SETTING):
                ...
            ```
    """
    # Apply patch for all settings EXCEPT LLM_TEST_MODE == "full"
    if LLM_TEST_MODE != "full":
        apply_mock_llm_patch(monkeypatch)
    yield