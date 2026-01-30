"""
================================================================================
UTILS PACKAGE INITIALIZATION
================================================================================

This package contains utility functions and helpers used throughout
the project, including progress bars, path management, device utilities,
and various helper functions.
"""

from .helpers import (
    get_progress_bar,
    get_iterable_progress_bar,
    ensure_path,
    check_file_exists,
    get_file_size,
    get_device_info,
    get_system_memory_info,
    check_system_resources,
    format_duration,
    Timer,
    format_number,
    format_percentage,
    split_list,
    flatten_list,
    get_unique_elements,
)

__all__ = [
    'get_progress_bar',
    'get_iterable_progress_bar',
    'ensure_path',
    'check_file_exists',
    'get_file_size',
    'get_device_info',
    'get_system_memory_info',
    'check_system_resources',
    'format_duration',
    'Timer',
    'format_number',
    'format_percentage',
    'split_list',
    'flatten_list',
    'get_unique_elements',
]
