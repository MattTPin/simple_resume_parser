"""logging.py
Holds configured loggers.
"""
from typing import Literal
import logging
import os
from datetime import datetime
import sys

from dotenv import load_dotenv

load_dotenv()  # load .env

ENV = os.getenv("ENV", "development")  # e.g., development, staging, prod

# --------------------------------------------------------------
# SETUP LOGGER
# --------------------------------------------------------------
LoggerType = Literal["default", "pytest", "extractor", "extractor_error"]

class LoggerFactory:
    """
    Factory to create configured loggers for different purposes.

    Logging behavior depends on environment (ENV):
      - Console logging is always enabled.
      - Local file logging in development with separate folders per logger type.
      - Cloud logging (optional) in staging/production using watchtower.
      - Timestamped log files prevent overwrites.
      - Duplicate handlers are avoided automatically.
    """

    def __init__(self, env: str = ENV, base_log_folder: str = "logs"):
        self.env = env
        self.base_log_folder = base_log_folder

    def get_logger(self, name: str, logger_type: LoggerType = "default") -> logging.Logger:
        """
        Create and return a configured logger based on type.

        Args:
            name (str): Name of the logger.
            logger_type (LoggerType): Type of logger to create.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            return logger  # prevent duplicate handlers

        # Set logging level
        level = logging.DEBUG if logger_type in ["default", "pytest"] else logging.INFO
        logger.setLevel(level)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Development file logging
        if self.env == "development":
            log_folder = self._get_log_folder_for_type(logger_type)
            os.makedirs(log_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(log_folder, f"{name}_{timestamp}.log")
            fh = logging.FileHandler(log_file_path)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Cloud logging for staging/production
        # if self.env in ["staging", "production"]:
        #     self._add_cloudwatch_handler(logger, logger_type, formatter)

        return logger

    def _get_log_folder_for_type(self, logger_type: LoggerType) -> str:
        """
        Return folder path based on logger type.
        """
        if any("pytest" in arg for arg in sys.argv):
            return os.path.join(self.base_log_folder, "tests")
        mapping = {
            "default": self.base_log_folder,
            "pytest": os.path.join(self.base_log_folder, "tests"),
            "extractor": os.path.join(self.base_log_folder, "extraction_failures"),
            "extractor_error": os.path.join(self.base_log_folder, "extractor_errors"),
        }
        return mapping.get(logger_type, self.base_log_folder)

    # def _add_cloudwatch_handler(self, logger: logging.Logger, logger_type: LoggerType, formatter: logging.Formatter):
    #     """
    #     Optional AWS CloudWatch logging for staging/production.
    #     """
    #     try:
    #         import watchtower
    #         log_group = "default_logs"
    #         if logger_type == "extractor":
    #             log_group = "extractor_logs"
    #         elif logger_type == "extractor_error":
    #             log_group = "extractor_error_logs"
    #         aws_handler = watchtower.CloudWatchLogHandler(log_group=log_group)
    #         aws_handler.setFormatter(formatter)
    #         logger.addHandler(aws_handler)
    #     except ImportError:
    #         logger.warning("watchtower not installed, skipping cloud logging.")