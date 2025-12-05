"""Structured logging configuration for the application."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    name: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with structured formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name. If None, returns root logger.
        format_string: Custom format string. Uses default if None.
    
    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    This is a convenience function that returns a child logger
    of the main application logger.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Default application logger
logger = setup_logging(name="snappfood_sentiment")
