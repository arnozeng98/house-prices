"""
================================================================================
UTILITY FUNCTIONS MODULE
================================================================================

This module contains general-purpose helper functions used throughout
the project. Includes progress bars, path utilities, device detection,
and other common operations.

Features:
    - Rich progress bars with TQDM
    - Path handling and validation
    - Device detection utilities
    - File I/O helpers
    - String formatting utilities
    - Memory management helpers

Example:
    from src.utils.helpers import get_progress_bar, ensure_path, format_duration
    
    # Create a progress bar
    pbar = get_progress_bar(total=100, desc="Processing")
    for i in range(100):
        # Do work
        pbar.update(1)
    
    # Format duration
    seconds = 3661
    formatted = format_duration(seconds)  # Returns "1h 1m 1s"
================================================================================
"""

import os
import time
import psutil
import torch
from pathlib import Path
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PROGRESS BAR UTILITIES
# ============================================================================

def get_progress_bar(
    total: int,
    desc: str = "",
    unit: str = "it",
    position: int = 0,
    leave: bool = True,
    disable: bool = False,
    ncols: Optional[int] = None
) -> tqdm:
    """
    Create a formatted TQDM progress bar with consistent styling.
    
    This function provides a standardized way to create progress bars
    throughout the project with consistent appearance and behavior.
    
    Args:
        total (int): Total number of iterations
        desc (str): Description of the progress bar
        unit (str): Unit name for iterations (default: "it")
        position (int): Position of the bar (for multiple bars)
        leave (bool): Whether to keep the bar after completion
        disable (bool): Disable the progress bar
        ncols (Optional[int]): Width of the progress bar
        
    Returns:
        tqdm: A progress bar object
        
    Example:
        >>> pbar = get_progress_bar(100, desc="Training")
        >>> for i in range(100):
        ...     time.sleep(0.01)
        ...     pbar.update(1)
        >>> pbar.close()
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        position=position,
        leave=leave,
        disable=disable,
        ncols=ncols,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )


def get_iterable_progress_bar(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "it"
) -> tqdm:
    """
    Create a progress bar for iterating over an iterable.
    
    Args:
        iterable: The iterable to wrap
        desc (str): Description of the progress bar
        total (Optional[int]): Total number of items (auto-detected if possible)
        unit (str): Unit name for iterations
        
    Returns:
        tqdm: A progress bar object wrapping the iterable
        
    Example:
        >>> items = range(100)
        >>> for item in get_iterable_progress_bar(items, desc="Processing"):
        ...     # Process item
        ...     pass
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )


# ============================================================================
# PATH UTILITIES
# ============================================================================

def ensure_path(path: Union[str, Path], is_dir: bool = False) -> Path:
    """
    Ensure a path exists, creating directories if necessary.
    
    This function validates and creates paths for files or directories.
    
    Args:
        path (Union[str, Path]): Path to ensure
        is_dir (bool): Whether the path is a directory
        
    Returns:
        Path: The validated Path object
        
    Raises:
        ValueError: If path validation fails
        
    Example:
        >>> path = ensure_path("data/output", is_dir=True)
        >>> file_path = ensure_path("data/output/results.csv", is_dir=False)
    """
    path_obj = Path(path)
    
    if is_dir:
        # Create directory structure
        path_obj.mkdir(parents=True, exist_ok=True)
    else:
        # Create parent directories for file
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    return path_obj


def check_file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists at the given path.
    
    Args:
        path (Union[str, Path]): Path to check
        
    Returns:
        bool: True if file exists, False otherwise
    """
    return Path(path).is_file()


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        path (Union[str, Path]): Path to the file
        
    Returns:
        int: File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return file_path.stat().st_size


def get_directory_size(path: Union[str, Path]) -> int:
    """
    Calculate total size of all files in a directory.
    
    Args:
        path (Union[str, Path]): Path to the directory
        
    Returns:
        int: Total size in bytes
        
    Raises:
        NotADirectoryError: If path is not a directory
    """
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    total_size = 0
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device_info() -> dict:
    """
    Get detailed information about the available computing device.
    
    Returns detailed information about CUDA/CPU availability and specs.
    
    Returns:
        dict: Dictionary containing device information
        
    Example:
        >>> info = get_device_info()
        >>> print(info['device_type'])  # 'cuda' or 'cpu'
        >>> print(info['device_name'])  # GPU name or 'CPU'
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
    }
    
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
        info['gpu_memory_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        info['device_type'] = 'cpu'
        info['device_name'] = 'CPU'
        info['cuda_device_count'] = 0
        info['gpu_memory_mb'] = 0
    
    return info


def get_system_memory_info() -> dict:
    """
    Get system memory usage information.
    
    Returns:
        dict: Memory information including total, used, and available
        
    Example:
        >>> mem_info = get_system_memory_info()
        >>> print(f"Memory used: {mem_info['percent']:.1f}%")
    """
    memory = psutil.virtual_memory()
    return {
        'total_mb': memory.total / (1024**2),
        'available_mb': memory.available / (1024**2),
        'used_mb': memory.used / (1024**2),
        'percent': memory.percent
    }


def check_system_resources(min_memory_mb: int = 1000) -> bool:
    """
    Check if system has sufficient resources for training.
    
    Args:
        min_memory_mb (int): Minimum required memory in MB
        
    Returns:
        bool: True if sufficient resources available
    """
    mem_info = get_system_memory_info()
    
    if mem_info['available_mb'] < min_memory_mb:
        logger.warning(
            f"Low available memory: {mem_info['available_mb']:.1f}MB "
            f"(minimum required: {min_memory_mb}MB)"
        )
        return False
    
    return True


# ============================================================================
# TIME UTILITIES
# ============================================================================

def format_duration(seconds: Union[int, float]) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds (Union[int, float]): Duration in seconds
        
    Returns:
        str: Formatted duration string (e.g., "1h 2m 3s")
        
    Example:
        >>> format_duration(3661)
        '1h 1m 1s'
        >>> format_duration(42)
        '42s'
    """
    seconds = int(seconds)
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0:
        parts.append(f"{secs}s")
    
    return " ".join(parts) if parts else "0s"


class Timer:
    """
    Context manager for timing code execution.
    
    Example:
        >>> with Timer(name="Data Loading"):
        ...     load_data()
        ...
        >>> # Logs elapsed time
    """
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            name (str): Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        logger.info(f"[{self.name}] Starting...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        formatted_duration = format_duration(duration)
        
        if exc_type is None:
            logger.info(f"[{self.name}] Completed in {formatted_duration}")
        else:
            logger.error(f"[{self.name}] Failed after {formatted_duration}")
        
        return False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


# ============================================================================
# STRING FORMATTING UTILITIES
# ============================================================================

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
        
    Example:
        >>> format_number(3.14159, decimals=2)
        '3.14'
    """
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value (float): Value between 0 and 1, or 0-100
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
        
    Example:
        >>> format_percentage(0.9234)
        '92.3%'
    """
    if value <= 1:
        value = value * 100
    return f"{value:.{decimals}f}%"


# ============================================================================
# ARRAY/LIST UTILITIES
# ============================================================================

def split_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst (List): List to split
        chunk_size (int): Size of each chunk
        
    Returns:
        List[List]: List of chunks
        
    Example:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten a nested list.
    
    Args:
        nested_list (List[List]): Nested list to flatten
        
    Returns:
        List: Flattened list
        
    Example:
        >>> flatten_list([[1, 2], [3, 4]])
        [1, 2, 3, 4]
    """
    return [item for sublist in nested_list for item in sublist]


def get_unique_elements(lst: List) -> List:
    """
    Get unique elements from a list while preserving order.
    
    Args:
        lst (List): Input list
        
    Returns:
        List: List with unique elements in original order
    """
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]
