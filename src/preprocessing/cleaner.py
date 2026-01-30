"""
================================================================================
DATA CLEANING AND PREPROCESSING MODULE
================================================================================

This module handles missing value imputation, outlier detection,
and general data cleaning operations.

Features:
    - Intelligent missing value handling (numerical and categorical)
    - Outlier detection and treatment (IQR, Z-score)
    - Data validation and quality checks
    - Handling domain-specific missing values in Ames dataset

Example:
    from src.preprocessing.cleaner import DataCleaner
    
    cleaner = DataCleaner(config)
    X_cleaned = cleaner.fit_transform(X)
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from sklearn.impute import SimpleImputer
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning, missing value imputation, and outlier detection.
    
    For the Ames Housing dataset, many "missing" values actually represent
    the absence of a feature (e.g., no garage means GarageArea=0, GarageCars=0).
    This cleaner handles both types of missing values appropriately.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration object with preprocessing settings
        """
        self.config = config
        self.preprocessing_config = config.preprocessing
        
        # Initialize imputers (will be fit during fit_transform)
        self.numerical_imputer = None
        self.categorical_imputer = None
        
        # Store statistics for transform
        self.numerical_mean_std = {}
        self.categorical_mode = {}
        
        # Feature names
        self.numerical_features = None
        self.categorical_features = None
        
        logger.info("DataCleaner initialized")
    
    def fit(self, X: pd.DataFrame) -> 'DataCleaner':
        """
        Fit the cleaner to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            
        Returns:
            DataCleaner: Self for method chaining
        """
        logger.info("Fitting DataCleaner to training data...")
        
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        
        # Fit numerical imputer
        numerical_strategy = self.preprocessing_config.missing_value_strategy.numerical
        self.numerical_imputer = SimpleImputer(strategy=numerical_strategy)
        self.numerical_imputer.fit(X[self.numerical_features])
        
        # Fit categorical imputer
        categorical_strategy = self.preprocessing_config.missing_value_strategy.categorical
        self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        self.categorical_imputer.fit(X[self.categorical_features])
        
        logger.info(
            f"Imputers fitted - Numerical: {numerical_strategy}, "
            f"Categorical: {categorical_strategy}"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning transformations to data.
        
        Args:
            X (pd.DataFrame): Features to clean
            
        Returns:
            pd.DataFrame: Cleaned features
        """
        if self.numerical_imputer is None:
            raise ValueError("Cleaner must be fit first. Call fit() before transform()")
        
        # Create a copy to avoid modifying original
        X_clean = X.copy()
        
        # Handle Ames-specific missing values
        # Many 'NA' values represent absence of feature, not missing data
        X_clean = self._handle_ames_specific_na(X_clean)
        
        # Impute missing values
        X_clean[self.numerical_features] = self.numerical_imputer.transform(
            X_clean[self.numerical_features]
        )
        X_clean[self.categorical_features] = self.categorical_imputer.transform(
            X_clean[self.categorical_features]
        )
        
        # Handle outliers if enabled
        if self.preprocessing_config.outlier_handling.enabled:
            X_clean = self._handle_outliers(X_clean)
        
        return X_clean
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit cleaner and transform data in one step.
        
        Args:
            X (pd.DataFrame): Features to clean
            
        Returns:
            pd.DataFrame: Cleaned features
        """
        self.fit(X)
        return self.transform(X)
    
    def _handle_ames_specific_na(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle Ames-specific missing values.
        
        In the Ames dataset, many 'NA' values represent the absence of a feature
        rather than missing data. For example:
            - GarageType='NA' means no garage
            - BsmtQual='NA' means no basement
        
        Args:
            X (pd.DataFrame): Features to process
            
        Returns:
            pd.DataFrame: Processed features
        """
        X_processed = X.copy()
        
        # Mapping of features that should have 'None' value instead of NaN
        # when they're missing (represents absence of feature)
        feature_replacements = {
            # Garage features
            'GarageType': 'No Garage',
            'GarageQual': 'None',
            'GarageCond': 'None',
            'GarageFinish': 'None',
            
            # Basement features
            'BsmtQual': 'None',
            'BsmtCond': 'None',
            'BsmtFinType1': 'None',
            'BsmtFinType2': 'None',
            'BsmtExposure': 'None',
            
            # Other features
            'Alley': 'None',
            'Fence': 'None',
            'MiscFeature': 'None',
            'PoolQC': 'None',
            'FireplaceQu': 'None',
        }
        
        for feature, replacement in feature_replacements.items():
            if feature in X_processed.columns:
                # Replace NaN values with the replacement string
                X_processed[feature] = X_processed[feature].fillna(replacement)
                
                # Also handle actual 'NA' strings (which pandas might keep as strings)
                if X_processed[feature].dtype == 'object':
                    X_processed[feature] = X_processed[feature].replace('NA', replacement)
        
        return X_processed
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical features.
        
        Args:
            X (pd.DataFrame): Features to process
            
        Returns:
            pd.DataFrame: Processed features with outliers handled
        """
        X_processed = X.copy()
        
        method = self.preprocessing_config.outlier_handling.method
        threshold = self.preprocessing_config.outlier_handling.threshold
        
        for feature in self.numerical_features:
            if method == 'iqr':
                # IQR method
                Q1 = X_processed[feature].quantile(0.25)
                Q3 = X_processed[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing them
                X_processed[feature] = X_processed[feature].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(X_processed[feature].dropna()))
                outlier_mask = z_scores > threshold
                
                if outlier_mask.any():
                    # Replace extreme values with mean
                    mean_val = X_processed[feature].mean()
                    X_processed.loc[X_processed[feature].index[outlier_mask], feature] = mean_val
        
        return X_processed
    
    def get_missing_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of missing values in the dataset.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Summary with columns [Feature, Missing_Count, Missing_Percent]
        """
        missing_data = pd.DataFrame({
            'Feature': X.columns,
            'Missing_Count': X.isnull().sum().values,
            'Missing_Percent': (X.isnull().sum().values / len(X)) * 100
        })
        
        # Sort by missing percentage (descending)
        missing_data = missing_data.sort_values('Missing_Percent', ascending=False)
        
        # Filter to show only features with missing values
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        
        return missing_data
