"""
================================================================================
DATA I/O PACKAGE INITIALIZATION
================================================================================

This package handles all data input/output operations including:
    - Loading and saving CSV files
    - Ground truth generation
    - Dataset validation
    - Feature metadata persistence
"""

from .ground_truth import (
    load_ames_dataset_sklearn,
    load_ames_dataset_seaborn,
    generate_ground_truth,
    validate_ground_truth,
    load_kaggle_submission_format,
    load_test_ids,
)

__all__ = [
    'load_ames_dataset_sklearn',
    'load_ames_dataset_seaborn',
    'generate_ground_truth',
    'validate_ground_truth',
    'load_kaggle_submission_format',
    'load_test_ids',
]
