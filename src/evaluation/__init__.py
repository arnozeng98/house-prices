"""
================================================================================
EVALUATION PACKAGE INITIALIZATION
================================================================================

This package contains evaluation and comparison modules:
    - Model evaluation metrics
    - Ground truth comparison
    - Cross-validation utilities
"""

from .comparison import GroundTruthComparison

__all__ = ['GroundTruthComparison']
