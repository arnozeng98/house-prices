"""
================================================================================
LOGGING CONFIGURATION MODULE
================================================================================

This module sets up a comprehensive logging system for the entire project.
Provides both file and console logging with customizable formats and levels.

Features:
    - File-based logging to logs/results.log
    - Console logging with color formatting
    - Per-module logger creation
    - Structured logging for model results and metrics
    - Automatic log directory creation

Example:
    from src.logging_config import setup_logging, get_logger
    
    # Initialize logging system
    setup_logging()
    
    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started")
================================================================================
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.config import get_config


# Color codes for console logging
class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds color codes to console log output.
    
    Color scheme:
        - DEBUG: Blue
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Red background
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[91m',   # Red
    }
    RESET = '\033[0m'            # Reset color
    
    def format(self, record):
        """
        Format log record with color codes.
        
        Args:
            record: The LogRecord to format
            
        Returns:
            Formatted log message with color codes
        """
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Global logger dictionary to keep track of created loggers
_loggers = {}


def setup_logging(config_path: str = "configs/default.yaml") -> None:
    """
    Set up the logging system for the entire project.
    
    This function:
    1. Loads logging configuration from YAML
    2. Creates log directory if needed
    3. Sets up both file and console handlers
    4. Configures log format and level
    
    Should be called once at the start of the application.
    
    Args:
        config_path (str): Path to the configuration file
        
    Example:
        >>> setup_logging()
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    try:
        config = get_config(config_path)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        config = None
    
    # Get logging configuration
    if config is not None:
        log_config = config.logging
        log_file = log_config.log_file
        log_level_str = log_config.level
        log_format_str = log_config.format
        date_format_str = log_config.date_format
        console_output = log_config.console_output
    else:
        # Fallback defaults
        log_file = "logs/results.log"
        log_level_str = "INFO"
        log_format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format_str = "%Y-%m-%d %H:%M:%S"
        console_output = True
    
    # Convert log level string to logging constant
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format_str, datefmt=date_format_str)
    
    # File handler - always enabled
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler - if enabled
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Use color formatter for console output
        color_formatter = ColorFormatter(log_format_str, datefmt=date_format_str)
        console_handler.setFormatter(color_formatter)
        root_logger.addHandler(console_handler)
    
    # Log setup completion
    root_logger.info(f"Logging initialized - Level: {log_level_str}, "
                     f"File: {log_file}, Console: {console_output}")


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a specific module.
    
    This function returns a named logger that can be used for logging
    within a specific module or component. Loggers are cached to
    avoid creating duplicates.
    
    Args:
        name (str): The name of the logger (typically __name__)
        
    Returns:
        logging.Logger: A configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        >>> logger.error("This is an error message")
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


def log_metrics(metrics: dict, stage: str = "evaluation", logger: Optional[logging.Logger] = None) -> None:
    """
    Log model metrics in a structured format.
    
    This function formats and logs model evaluation metrics such as
    RMSE, MAE, R2, and cross-validation scores.
    
    Args:
        metrics (dict): Dictionary containing metric names and values
        stage (str): Stage name (e.g., "training", "validation", "evaluation")
        logger (Optional[logging.Logger]): Logger instance. If None, uses root logger.
        
    Example:
        >>> metrics = {'rmse': 0.125, 'mae': 0.098, 'r2': 0.92}
        >>> log_metrics(metrics, stage='validation')
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"METRICS - {stage.upper()}")
    logger.info(f"{'='*60}")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"  {metric_name:.<40} {metric_value:.6f}")
        else:
            logger.info(f"  {metric_name:.<40} {metric_value}")
    
    logger.info(f"{'='*60}\n")


def log_model_comparison(results: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a comparison of multiple models' performance.
    
    This function creates a formatted table of model performance metrics
    for easy comparison and analysis.
    
    Args:
        results (dict): Dictionary with model names as keys and metric dicts as values
        logger (Optional[logging.Logger]): Logger instance. If None, uses root logger.
        
    Example:
        >>> results = {
        ...     'xgboost': {'rmse': 0.125, 'mae': 0.098},
        ...     'random_forest': {'rmse': 0.135, 'mae': 0.105}
        ... }
        >>> log_model_comparison(results)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MODEL COMPARISON")
    logger.info(f"{'='*80}")
    
    # Get all metric names
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    # Create header
    header = f"{'Model':<20}"
    for metric in all_metrics:
        header += f" {metric:>12}"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Log each model's results
    for model_name, metrics in results.items():
        row = f"{model_name:<20}"
        for metric in all_metrics:
            value = metrics.get(metric, "N/A")
            if isinstance(value, (int, float)):
                row += f" {value:>12.6f}"
            else:
                row += f" {str(value):>12}"
        logger.info(row)
    
    logger.info(f"{'='*80}\n")


def log_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a section header for better log organization.
    
    Args:
        title (str): The title of the section
        logger (Optional[logging.Logger]): Logger instance. If None, uses root logger.
        
    Example:
        >>> log_section("Data Preprocessing")
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {title.upper()}")
    logger.info(f"{'='*60}")
