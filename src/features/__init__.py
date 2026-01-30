"""
================================================================================
FEATURES PACKAGE INITIALIZATION
================================================================================

This package contains feature engineering and selection modules including:
    - Automatic feature creation and interactions
    - Feature selection (Boruta, permutation, mutual information)
    - Feature importance tracking
"""

from .engineer import FeatureEngineer
from .selector import FeatureSelector

__all__ = [
    'FeatureEngineer',
    'FeatureSelector',
]
