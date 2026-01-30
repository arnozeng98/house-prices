"""
================================================================================
PREPROCESSING PACKAGE INITIALIZATION
================================================================================

This package contains all data preprocessing modules including:
    - Data cleaning and missing value imputation
    - Categorical feature encoding
    - Numerical feature scaling/normalization
    - Complete preprocessing pipeline
"""

from .cleaner import DataCleaner
from .encoder import FeatureEncoder
from .scaler import FeatureScaler
from .pipeline import PreprocessingPipeline

__all__ = [
    'DataCleaner',
    'FeatureEncoder',
    'FeatureScaler',
    'PreprocessingPipeline',
]
