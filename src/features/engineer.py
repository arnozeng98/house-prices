"""
================================================================================
FEATURE ENGINEERING MODULE - AUTOMATIC FEATURE CREATION
================================================================================

This module automatically creates new features from existing ones:
    - Polynomial features (squares, interactions)
    - Domain-specific features for housing data
    - Categorical feature interactions
    - Statistical aggregations

Features:
    - Configurable feature interaction strategies
    - Automatic generation based on feature types
    - Domain knowledge integration for Ames dataset

Example:
    from src.features.engineer import FeatureEngineer
    
    engineer = FeatureEngineer(config)
    X_engineered = engineer.fit_transform(X)
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple, Set
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Automatically creates new features from existing features.
    
    Strategies:
        1. Polynomial features (x^2, xy interactions)
        2. Domain-specific features (for housing market)
        3. Categorical interactions
        4. Ratio and statistical features
    """
    
    def __init__(self, config: Any):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration object with feature engineering settings
        """
        self.config = config
        self.feature_config = config.feature_engineering
        
        # Track features
        self.numerical_features: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None
        self.engineered_features: List[str] = []
        self.created_features: Dict[str, str] = {}  # Mapping: new_feature -> rule
        
        # Store feature engineering rules
        self.fit_data: Optional[pd.DataFrame] = None
        
        logger.info("FeatureEngineer initialized")
    
    def fit(self, X: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit feature engineer to data.
        
        Args:
            X (pd.DataFrame): Training features
            
        Returns:
            FeatureEngineer: Self for method chaining
        """
        logger.info("Fitting FeatureEngineer...")
        
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.fit_data = X.copy()
        
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with engineered columns added
        """
        if not self.feature_config.enabled:
            logger.info("Feature engineering disabled")
            return X
        
        if self.numerical_features is None:
            raise ValueError("Engineer must be fit first")
        
        X_engineered = X.copy()
        
        logger.info("Creating engineered features...")
        
        # Create polynomial features
        if self.feature_config.numerical_interactions.enabled:
            X_engineered = self._create_polynomial_features(X_engineered)
        
        # Create domain-specific features
        if self.feature_config.domain_features.enabled:
            X_engineered = self._create_domain_features(X_engineered)
        
        # Create categorical interactions
        if self.feature_config.categorical_interactions.enabled:
            X_engineered = self._create_categorical_interactions(X_engineered)
        
        logger.info(f"Created {len(self.engineered_features)} new features")
        
        return X_engineered
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit engineer and create features in one step.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with engineered columns added
        """
        self.fit(X)
        return self.transform(X)
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features (squares and interactions).
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with polynomial features added
        """
        logger.info("Creating polynomial features...")
        
        max_interactions = self.feature_config.numerical_interactions.max_interactions
        degree = self.feature_config.numerical_interactions.polynomial_degree
        
        X_poly = X.copy()
        count = 0
        
        # Create square features for top features
        if degree >= 2:
            for feature in self.numerical_features[:min(10, len(self.numerical_features))]:
                if count >= max_interactions:
                    break
                new_feature = f"{feature}_squared"
                X_poly[new_feature] = X[feature] ** 2
                self.engineered_features.append(new_feature)
                self.created_features[new_feature] = "polynomial_square"
                count += 1
        
        # Create interaction features
        if degree >= 2:
            for feat1, feat2 in combinations(
                self.numerical_features[:min(8, len(self.numerical_features))],
                2
            ):
                if count >= max_interactions:
                    break
                new_feature = f"{feat1}_x_{feat2}"
                X_poly[new_feature] = X[feat1] * X[feat2]
                self.engineered_features.append(new_feature)
                self.created_features[new_feature] = "polynomial_interaction"
                count += 1
        
        logger.info(f"Created {count} polynomial features")
        return X_poly
    
    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for Ames Housing dataset.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with domain features added
        """
        logger.info("Creating domain-specific features...")
        
        X_domain = X.copy()
        domain_rules = self.feature_config.domain_features.rules
        
        try:
            # Total area features
            if "total_area" in domain_rules:
                area_cols = [col for col in X.columns if 'SF' in col or 'Area' in col]
                if area_cols:
                    X_domain['Total_Area'] = X[area_cols].sum(axis=1)
                    self.engineered_features.append('Total_Area')
                    self.created_features['Total_Area'] = 'domain_total_area'
            
            # Quality score features
            if "quality_score" in domain_rules:
                quality_cols = [col for col in X.columns if 'Qual' in col or 'Cond' in col]
                if quality_cols:
                    # Convert to numeric if needed
                    quality_df = X[quality_cols].apply(pd.to_numeric, errors='coerce')
                    X_domain['Quality_Score'] = quality_df.mean(axis=1)
                    self.engineered_features.append('Quality_Score')
                    self.created_features['Quality_Score'] = 'domain_quality_score'
            
            # Age features
            if "age_features" in domain_rules and 'YrSold' in X.columns:
                if 'YearBuilt' in X.columns:
                    X_domain['House_Age'] = X['YrSold'] - X['YearBuilt']
                    self.engineered_features.append('House_Age')
                    self.created_features['House_Age'] = 'domain_age'
        
        except Exception as e:
            logger.warning(f"Error creating domain features: {e}")
        
        return X_domain
    
    def _create_categorical_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interactions between categorical features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with categorical interactions added
        """
        logger.info("Creating categorical feature interactions...")
        
        X_cat = X.copy()
        max_depth = self.feature_config.categorical_interactions.max_depth
        count = 0
        max_total = 30
        
        try:
            # Create 2-way interactions
            if max_depth >= 2:
                for feat1, feat2 in combinations(
                    self.categorical_features[:min(5, len(self.categorical_features))],
                    2
                ):
                    if count >= max_total:
                        break
                    new_feature = f"{feat1}_{feat2}"
                    X_cat[new_feature] = (X[feat1].astype(str) + "_" + X[feat2].astype(str))
                    self.engineered_features.append(new_feature)
                    self.created_features[new_feature] = "categorical_interaction"
                    count += 1
        
        except Exception as e:
            logger.warning(f"Error creating categorical interactions: {e}")
        
        logger.info(f"Created {count} categorical interaction features")
        return X_cat
    
    def get_engineered_features(self) -> List[str]:
        """
        Get list of engineered feature names.
        
        Returns:
            List[str]: List of newly created feature names
        """
        return self.engineered_features.copy()
    
    def get_feature_creation_summary(self) -> Dict[str, int]:
        """
        Get summary of created features by type.
        
        Returns:
            Dict[str, int]: Count of features by creation rule
        """
        summary = {}
        for rule in self.created_features.values():
            summary[rule] = summary.get(rule, 0) + 1
        return summary
