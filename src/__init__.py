"""
================================================================================
AMES HOUSING PRICE PREDICTION - MAIN PACKAGE INITIALIZATION
================================================================================

This package contains the complete machine learning pipeline for predicting
house prices in the Ames housing dataset competition.

Package Structure:
    - config: Configuration management and YAML loader
    - io: Data input/output operations
    - preprocessing: Data cleaning and transformation
    - features: Feature engineering and selection
    - models: Machine learning models and ensemble
    - tuning: Hyperparameter optimization with Optuna
    - evaluation: Model evaluation metrics and cross-validation
    - visualization: Plots and visualizations
    - utils: Helper functions and utilities
    - train: Main training script
    - predict: Prediction and inference script

Author: Arno
Version: 1.0.0
================================================================================
"""

__version__ = "1.0.0"
__author__ = "Arno"

# Import main components for easy access
try:
    from .config import Config
except ImportError as e:
    print(f"Warning: Could not import Config from config module: {e}")
