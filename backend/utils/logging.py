"""
Logging utilities for HifazatAI backend services.
Provides structured logging with different levels and formatters.
"""

import logging
import sys
from datetime import datetime
from typing import Optional
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        use_colors: Whether to use colored output for console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if use_colors:
        colored_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(colored_formatter)
    else:
        console_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get a logger instance with default configuration.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional logging level override
    
    Returns:
        Logger instance
    """
    # Get log level from environment or use INFO as default
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    
    # Get log file from environment
    log_file = os.getenv("LOG_FILE")
    
    return setup_logging(
        name=name,
        level=log_level,
        log_file=log_file,
        use_colors=True
    )


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Application-specific loggers
def get_api_logger() -> logging.Logger:
    """Get logger for API components"""
    return get_logger("hifazat.api")


def get_pipeline_logger(pipeline_name: str) -> logging.Logger:
    """Get logger for pipeline components"""
    return get_logger(f"hifazat.pipeline.{pipeline_name}")


def get_database_logger() -> logging.Logger:
    """Get logger for database operations"""
    return get_logger("hifazat.database")


# Error tracking utilities
class ErrorTracker:
    """Simple error tracking for monitoring"""
    
    def __init__(self):
        self.error_counts = {}
        self.logger = get_logger("hifazat.errors")
    
    def track_error(self, component: str, error: Exception, context: dict = None):
        """Track an error occurrence"""
        error_key = f"{component}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        self.logger.error(
            f"Error in {component}: {str(error)}",
            extra={
                "component": component,
                "error_type": type(error).__name__,
                "error_count": self.error_counts[error_key],
                "context": context or {}
            }
        )
    
    def get_error_summary(self) -> dict:
        """Get summary of tracked errors"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": self.error_counts.copy(),
            "timestamp": datetime.now().isoformat()
        }


# Global error tracker instance
error_tracker = ErrorTracker()