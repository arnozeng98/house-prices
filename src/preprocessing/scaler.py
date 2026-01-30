"""
================================================================================
FEATURE SCALING AND NORMALIZATION MODULE
================================================================================

This module handles scaling and normalization of numerical features:
    - Standard scaling (z-score normalization)
    - Robust scaling (using quantiles, resistant to outliers)
    - MinMax scaling (scales to [0, 1] range)
    - No scaling (pass-through)

Features:
    - Fit on training data and apply to test data
    - Preserve scaling parameters for inference
    - Column-wise scaling

Example:
    from src.preprocessing.scaler import FeatureScaler
    
    scaler = FeatureScaler(config)
    X_scaled = scaler.fit_transform(X)
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Scales numerical features using configurable strategies.
    
    Supports:
        - Standard scaling (z-score normalization): (x - mean) / std
        - Robust scaling: resistant to outliers, uses quantiles
        - MinMax scaling: scales to [0, 1] range
        - No scaling (pass-through)
    """
    
    def __init__(self, config: Any):
        """
        Initialize the feature scaler.
        
        Args:
            config: Configuration object with preprocessing settings
        """
        self.config = config
        self.preprocessing_config = config.preprocessing
        self.scaling_strategy = self.preprocessing_config.numerical_scaler
        
        # Initialize scalers
        self.scaler: Optional[Any] = None
        
        # Feature tracking
        self.numerical_features: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None
        self.fitted_features: Optional[List[str]] = None
        
        logger.info(f"FeatureScaler initialized with strategy: {self.scaling_strategy}")
    
    def fit(self, X: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit the scaler to training data.
        
        Args:
            X (pd.DataFrame): Training features
            
        Returns:
            FeatureScaler: Self for method chaining
        """
        logger.info(f"Fitting FeatureScaler using {self.scaling_strategy} strategy...")
        
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical features to scale: {len(self.numerical_features)}")
        logger.info(f"Categorical features (not scaled): {len(self.categorical_features)}")
        
        # Skip if no numerical features
        if not self.numerical_features:
            logger.warning("No numerical features found to scale")
            return self
        
        # Initialize and fit scaler based on strategy
        if self.scaling_strategy == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numerical_features])
            
        elif self.scaling_strategy == 'robust':
            self.scaler = RobustScaler()
            self.scaler.fit(X[self.numerical_features])
            
        elif self.scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler.fit(X[self.numerical_features])
            
        elif self.scaling_strategy == 'none':
            logger.info("Scaling disabled (none)")
            self.scaler = None
            
        else:
            raise ValueError(
                f"Unknown scaling strategy: {self.scaling_strategy}. "
                f"Must be one of: 'standard', 'robust', 'minmax', 'none'"
            )
        
        self.fitted_features = self.numerical_features.copy()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling transformations to data.
        
        Args:
            X (pd.DataFrame): Features to scale
            
        Returns:
            pd.DataFrame: Scaled features
        """
        if self.fitted_features is None:
            raise ValueError("Scaler must be fit first. Call fit() before transform()")
        
        X_scaled = X.copy()
        
        # Skip if no scaling strategy
        if self.scaling_strategy == 'none' or self.scaler is None:
            return X_scaled
        
        # Apply scaling to numerical features
        X_scaled[self.numerical_features] = self.scaler.transform(
            X_scaled[self.numerical_features]
        )
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform data in one step.
        
        Args:
            X (pd.DataFrame): Features to scale
            
        Returns:
            pd.DataFrame: Scaled features
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the scaling transformation (inverse scaling).
        
        This is useful for interpreting predictions or inverting
        transformed features back to original scale.
        
        Args:
            X (pd.DataFrame): Scaled features
            
        Returns:
            pd.DataFrame: Unscaled features
            
        Raises:
            ValueError: If scaler doesn't support inverse_transform
        """
        if self.scaler is None or self.scaling_strategy == 'none':
            return X.copy()
        
        X_unscaled = X.copy()
        
        try:
            X_unscaled[self.numerical_features] = self.scaler.inverse_transform(
                X_unscaled[self.numerical_features]
            )
        except AttributeError:
            logger.warning(f"{self.scaling_strategy} scaler doesn't support inverse transform")
        
        return X_unscaled
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get scaling statistics for numerical features.
        
        Returns scaling parameters used (mean, std, min, max, etc.)
        
        Returns:
            Dict[str, Dict[str, float]]: Statistics for each feature
        """
        stats_dict = {}
        
        if self.scaler is None:
            return stats_dict
        
        if self.scaling_strategy == 'standard':
            for feature, mean, std in zip(
                self.numerical_features,
                self.scaler.mean_,
                self.scaler.scale_
            ):
                stats_dict[feature] = {
                    'mean': float(mean),
                    'std': float(std)
                }
                
        elif self.scaling_strategy == 'robust':
            for feature, center, scale in zip(
                self.numerical_features,
                self.scaler.center_,
                self.scaler.scale_
            ):
                stats_dict[feature] = {
                    'center': float(center),
                    'scale': float(scale)
                }
                
        elif self.scaling_strategy == 'minmax':
            for feature, min_val, scale in zip(
                self.numerical_features,
                self.scaler.data_min_,
                self.scaler.scale_
            ):
                stats_dict[feature] = {
                    'min': float(min_val),
                    'scale': float(scale)
                }
        
        return stats_dict
    
    def get_numerical_feature_names(self) -> List[str]:
        """
        Get list of numerical features that were scaled.
        
        Returns:
            List[str]: List of numerical feature names
        """
        return self.numerical_features.copy() if self.numerical_features else []
    
    def get_categorical_feature_names(self) -> List[str]:
        """
        Get list of categorical features (not scaled).
        
        Returns:
            List[str]: List of categorical feature names
        """
        return self.categorical_features.copy() if self.categorical_features else []
