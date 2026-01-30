"""
================================================================================
DATA PREPROCESSING PIPELINE MODULE
================================================================================

This module orchestrates the complete data preprocessing pipeline,
combining cleaning, encoding, and scaling into a single workflow.

Features:
    - End-to-end data preprocessing
    - Consistent train/test processing
    - Feature metadata tracking
    - Reproducible transformations

Example:
    from src.preprocessing.pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline(config)
    X_train_processed, X_test_processed = pipeline.process(
        X_train, X_test, y_train
    )
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from .cleaner import DataCleaner
from .encoder import FeatureEncoder
from .scaler import FeatureScaler
import logging

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline that orchestrates multiple transformations.
    
    Pipeline steps:
        1. Data cleaning (handle missing values, outliers)
        2. Feature encoding (categorical to numerical)
        3. Feature scaling (normalize numerical features)
    
    All transformations are learned on training data and applied to test data.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Configuration object with preprocessing settings
        """
        self.config = config
        
        # Initialize pipeline components
        self.cleaner = DataCleaner(config)
        self.encoder = FeatureEncoder(config)
        self.scaler = FeatureScaler(config)
        
        # Track which features remain after each step
        self.initial_features: Optional[List[str]] = None
        self.processed_features: Optional[List[str]] = None
        
        logger.info("PreprocessingPipeline initialized")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None
    ) -> 'PreprocessingPipeline':
        """
        Fit the pipeline to training data.
        
        This learns all transformations (imputation, encoding, scaling)
        from the training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (Optional[pd.Series]): Training target (for target encoding)
            
        Returns:
            PreprocessingPipeline: Self for method chaining
        """
        logger.info("="*60)
        logger.info("FITTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        self.initial_features = X_train.columns.tolist()
        
        # Step 1: Fit cleaner
        logger.info("\n[1/3] Fitting DataCleaner...")
        self.cleaner.fit(X_train)
        
        # Step 2: Fit encoder
        logger.info("[2/3] Fitting FeatureEncoder...")
        self.encoder.fit(X_train, y_train)
        
        # Step 3: Fit scaler
        logger.info("[3/3] Fitting FeatureScaler...")
        self.scaler.fit(X_train)
        
        logger.info("\nPipeline fitting completed")
        logger.info("="*60 + "\n")
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply all transformations to data.
        
        Args:
            X (pd.DataFrame): Features to transform
            y (Optional[pd.Series]): Target variable (for target encoding)
            
        Returns:
            pd.DataFrame: Fully preprocessed features
        """
        if self.initial_features is None:
            raise ValueError("Pipeline must be fit first. Call fit() before transform()")
        
        logger.info("Applying preprocessing transformations...")
        
        # Step 1: Clean data
        X_cleaned = self.cleaner.transform(X)
        logger.info(f"  ✓ Data cleaning complete - shape: {X_cleaned.shape}")
        
        # Step 2: Encode categorical features
        X_encoded = self.encoder.transform(X_cleaned, y)
        logger.info(f"  ✓ Feature encoding complete - shape: {X_encoded.shape}")
        
        # Step 3: Scale numerical features
        X_scaled = self.scaler.transform(X_encoded)
        logger.info(f"  ✓ Feature scaling complete - shape: {X_scaled.shape}")
        
        self.processed_features = X_scaled.columns.tolist()
        
        return X_scaled
    
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Fit pipeline on training data and transform both train and test data.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (Optional[pd.DataFrame]): Test features
            y_train (Optional[pd.Series]): Training target
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
                - Transformed training data
                - Transformed test data (None if not provided)
        """
        # Fit on training data
        self.fit(X_train, y_train)
        
        # Transform training data
        X_train_processed = self.transform(X_train, y_train)
        
        # Transform test data if provided
        X_test_processed = None
        if X_test is not None:
            X_test_processed = self.transform(X_test)
        
        return X_train_processed, X_test_processed
    
    def get_processed_features(self) -> List[str]:
        """
        Get list of features after preprocessing.
        
        Returns:
            List[str]: List of final feature names
        """
        return self.processed_features.copy() if self.processed_features else []
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing pipeline.
        
        Returns:
            Dict[str, Any]: Summary including feature counts and strategies
        """
        summary = {
            'initial_feature_count': len(self.initial_features) if self.initial_features else 0,
            'processed_feature_count': len(self.processed_features) if self.processed_features else 0,
            'cleaning_strategy': {
                'numerical': self.config.preprocessing.missing_value_strategy.numerical,
                'categorical': self.config.preprocessing.missing_value_strategy.categorical,
                'outlier_handling': self.config.preprocessing.outlier_handling.enabled,
            },
            'encoding_strategy': self.encoder.encoding_strategy,
            'scaling_strategy': self.scaler.scaling_strategy,
        }
        return summary
    
    def get_missing_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of missing values in original data.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Missing value summary
        """
        return self.cleaner.get_missing_summary(X)
