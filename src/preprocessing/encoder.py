"""
================================================================================
FEATURE ENCODING AND CATEGORICAL TRANSFORMATION MODULE
================================================================================

This module handles encoding of categorical features using various strategies:
    - One-hot encoding
    - Label encoding
    - Target encoding
    - Frequency encoding

Features:
    - Support for multiple encoding strategies
    - Handling of high-cardinality categorical features
    - Preservation of encoding mapping for inference

Example:
    from src.preprocessing.encoder import FeatureEncoder
    
    encoder = FeatureEncoder(config)
    X_encoded = encoder.fit_transform(X, y)
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Encodes categorical features using configurable strategies.
    
    Supports:
        - One-hot encoding (creates binary columns)
        - Label encoding (numeric labels)
        - Target encoding (mean target per category)
        - Frequency encoding (encode by frequency)
    """
    
    def __init__(self, config: Any):
        """
        Initialize the feature encoder.
        
        Args:
            config: Configuration object with preprocessing settings
        """
        self.config = config
        self.preprocessing_config = config.preprocessing
        self.encoding_strategy = self.preprocessing_config.categorical_encoder
        
        # Store encoding mappings for inference
        self.encoders_dict: Dict[str, Any] = {}
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.target_encoding_map: Dict[str, Dict] = {}
        self.frequency_encoding_map: Dict[str, Dict] = {}
        
        # Feature tracking
        self.categorical_features: Optional[List[str]] = None
        self.numerical_features: Optional[List[str]] = None
        self.fitted_features: Optional[List[str]] = None
        
        logger.info(f"FeatureEncoder initialized with strategy: {self.encoding_strategy}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEncoder':
        """
        Fit the encoder to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (Optional[pd.Series]): Target variable (required for target encoding)
            
        Returns:
            FeatureEncoder: Self for method chaining
        """
        logger.info(f"Fitting FeatureEncoder using {self.encoding_strategy} strategy...")
        
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        
        # Apply encoding strategy
        if self.encoding_strategy == 'onehot':
            self._fit_onehot(X)
        elif self.encoding_strategy == 'label':
            self._fit_label(X)
        elif self.encoding_strategy == 'target':
            if y is None:
                raise ValueError("Target variable required for target encoding")
            self._fit_target(X, y)
        elif self.encoding_strategy == 'frequency':
            self._fit_frequency(X)
        else:
            raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")
        
        self.fitted_features = self.categorical_features.copy()
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply encoding transformations to data.
        
        Args:
            X (pd.DataFrame): Features to encode
            y (Optional[pd.Series]): Target variable (for target encoding)
            
        Returns:
            pd.DataFrame: Encoded features
        """
        if not self.fitted_features:
            raise ValueError("Encoder must be fit first. Call fit() before transform()")
        
        X_encoded = X.copy()
        
        # Apply encoding strategy
        if self.encoding_strategy == 'onehot':
            X_encoded = self._transform_onehot(X_encoded)
        elif self.encoding_strategy == 'label':
            X_encoded = self._transform_label(X_encoded)
        elif self.encoding_strategy == 'target':
            X_encoded = self._transform_target(X_encoded)
        elif self.encoding_strategy == 'frequency':
            X_encoded = self._transform_frequency(X_encoded)
        
        return X_encoded
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit encoder and transform data in one step.
        
        Args:
            X (pd.DataFrame): Features to encode
            y (Optional[pd.Series]): Target variable (for target encoding)
            
        Returns:
            pd.DataFrame: Encoded features
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def _fit_onehot(self, X: pd.DataFrame) -> None:
        """
        Fit one-hot encoder for categorical features.
        
        Args:
            X (pd.DataFrame): Training features
        """
        if not self.categorical_features:
            return
        
        logger.info("Fitting one-hot encoder...")
        self.one_hot_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.int8
        )
        self.one_hot_encoder.fit(X[self.categorical_features])
    
    def _transform_onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical features.
        
        Args:
            X (pd.DataFrame): Features to encode
            
        Returns:
            pd.DataFrame: Encoded features
        """
        if self.one_hot_encoder is None or not self.categorical_features:
            return X
        
        # One-hot encode
        onehot_array = self.one_hot_encoder.transform(X[self.categorical_features])
        onehot_columns = self.one_hot_encoder.get_feature_names_out(self.categorical_features)
        onehot_df = pd.DataFrame(onehot_array, columns=onehot_columns, index=X.index)
        
        # Combine with numerical features
        X_encoded = pd.concat([
            X[self.numerical_features],
            onehot_df
        ], axis=1)
        
        return X_encoded
    
    def _fit_label(self, X: pd.DataFrame) -> None:
        """
        Fit label encoders for categorical features.
        
        Args:
            X (pd.DataFrame): Training features
        """
        logger.info("Fitting label encoders...")
        
        for feature in self.categorical_features:
            encoder = LabelEncoder()
            encoder.fit(X[feature].astype(str))
            self.encoders_dict[feature] = encoder
    
    def _transform_label(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoding to categorical features.
        
        Args:
            X (pd.DataFrame): Features to encode
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        for feature in self.categorical_features:
            if feature in self.encoders_dict:
                encoder = self.encoders_dict[feature]
                # Handle unknown categories
                X_encoded[feature] = X_encoded[feature].astype(str)
                X_encoded[feature] = encoder.transform(X_encoded[feature])
        
        return X_encoded
    
    def _fit_target(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit target encoder by computing mean target per category.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable
        """
        logger.info("Fitting target encoders...")
        
        for feature in self.categorical_features:
            # Compute mean target per category
            encoding_map = X.groupby(feature)[y.name if hasattr(y, 'name') else 0].mean()
            self.target_encoding_map[feature] = encoding_map.to_dict()
    
    def _transform_target(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target encoding to categorical features.
        
        Args:
            X (pd.DataFrame): Features to encode
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        # Global mean (for unknown categories)
        global_mean = 0  # Will be overridden by training stats
        
        for feature in self.categorical_features:
            if feature in self.target_encoding_map:
                encoding_map = self.target_encoding_map[feature]
                # Map with global mean as default for unknown categories
                X_encoded[feature] = X_encoded[feature].map(
                    encoding_map
                ).fillna(global_mean)
        
        return X_encoded
    
    def _fit_frequency(self, X: pd.DataFrame) -> None:
        """
        Fit frequency encoder by counting category occurrences.
        
        Args:
            X (pd.DataFrame): Training features
        """
        logger.info("Fitting frequency encoders...")
        
        for feature in self.categorical_features:
            # Count frequency of each category
            frequency_map = X[feature].value_counts(normalize=True).to_dict()
            self.frequency_encoding_map[feature] = frequency_map
    
    def _transform_frequency(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply frequency encoding to categorical features.
        
        Args:
            X (pd.DataFrame): Features to encode
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        for feature in self.categorical_features:
            if feature in self.frequency_encoding_map:
                encoding_map = self.frequency_encoding_map[feature]
                X_encoded[feature] = X_encoded[feature].map(encoding_map)
        
        return X_encoded
    
    def get_categorical_feature_names(self) -> List[str]:
        """
        Get list of categorical feature names.
        
        Returns:
            List[str]: List of categorical feature names
        """
        return self.categorical_features.copy() if self.categorical_features else []
    
    def get_numerical_feature_names(self) -> List[str]:
        """
        Get list of numerical feature names.
        
        Returns:
            List[str]: List of numerical feature names
        """
        return self.numerical_features.copy() if self.numerical_features else []
