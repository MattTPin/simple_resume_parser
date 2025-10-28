"""logging.py
Holds configured loggers.
"""
from typing import Literal
import logging
import os
import sys
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()  # load .env

ENV = os.getenv("ENV", "development")  # e.g., development, staging, prod

LoggerType = Literal["default", "pytest", "extractor", "extractor_error"]


class LoggerFactory:
    """
    Factory to create configured loggers for different purposes.

    Logging behavior depends on environment (ENV):
      - Console logging is optional.
      - Local file logging in development (separate folders per logger type).
      - Cloud logging (optional) in staging/production using watchtower.
      - Duplicate handlers and propagation are avoided automatically.
    """

    def __init__(self, env: str = ENV, base_log_folder: str = "logs"):
        self.env = env
        self.base_log_folder = base_log_folder

    def get_logger(
        self,
        name: str,
        logger_type: LoggerType = "default",
        console: bool = True
    ) -> logging.Logger:
        """
        Create and return a configured logger based on type.
        """
        logger = logging.getLogger(name)

        # Prevent duplicate handlers
        if logger.hasHandlers():
            return logger

        # Disable propagation to root logger
        logger.propagate = False

        # Set log level
        level = logging.DEBUG if logger_type in ["default", "pytest"] else logging.INFO
        logger.setLevel(level)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        # Optional console handler
        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # File logging (always available in development)
        if self.env in ["development", "local", "test"]:
            log_folder = self._get_log_folder_for_type(logger_type)
            os.makedirs(log_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(log_folder, f"{name}_{timestamp}.log")
            fh = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # CloudWatch handler for staging/production
        elif self.env in ["staging", "production"]:
            self._add_cloudwatch_handler(logger, logger_type, formatter)

        # Safety: ensure at least one handler exists
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger

    @lru_cache(maxsize=None)
    def get_extractor_field_logger(self, field_name: str) -> logging.Logger:
        """
        Return a field-specific extractor logger that writes to a subfolder.
        Example:
            logs/extraction_failures/name/name_20251028_103022.log
            logs/extraction_failures/skills/skills_20251028_103022.log
        """
        safe_field_name = field_name or "other"

        logger = logging.getLogger(f"extractor_{safe_field_name}")

        # Prevent duplicate handlers (esp. if re-requested)
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False  # do not print to console

        # Create subfolder for the field
        log_folder = os.path.join(
            self.base_log_folder, "extraction_failures", safe_field_name
        )
        os.makedirs(log_folder, exist_ok=True)

        # Use timestamped filename to avoid overwrite
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(
            log_folder, f"{safe_field_name}_{timestamp}.log"
        )

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        fh = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _get_log_folder_for_type(self, logger_type: LoggerType) -> str:
        """Return folder path based on logger type."""
        if any("pytest" in arg for arg in sys.argv):
            return os.path.join(self.base_log_folder, "tests")

        mapping = {
            "default": self.base_log_folder,
            "pytest": os.path.join(self.base_log_folder, "tests"),
            "extractor": os.path.join(self.base_log_folder, "extraction_failures"),
            "extractor_error": os.path.join(self.base_log_folder, "extractor_errors"),
        }
        return mapping.get(logger_type, self.base_log_folder)

    def _add_cloudwatch_handler(
        self,
        logger: logging.Logger,
        logger_type: LoggerType,
        formatter: logging.Formatter,
    ):
        """Optional AWS CloudWatch logging for staging/production."""
        try:
            import watchtower

            log_group = {
                "default": "default_logs",
                "extractor": "extractor_logs",
                "extractor_error": "extractor_error_logs",
            }.get(logger_type, "default_logs")

            aws_handler = watchtower.CloudWatchLogHandler(log_group=log_group)
            aws_handler.setFormatter(formatter)
            logger.addHandler(aws_handler)

        except ImportError:
            logger.warning("watchtower not installed, skipping cloud logging.")
