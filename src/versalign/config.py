"""Configuration settings for versalign."""

import logging
import os

# Global logger name for versalign
LOGGER_NAME = "versalign"
LOGGER_LEVEL = int(os.getenv("LOGGER_LEVEL", logging.INFO))


# Configure global logger
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOGGER_LEVEL)


DEFAULT_GAP_REPR = "-"
