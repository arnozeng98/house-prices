"""
================================================================================
RANDOM FOREST MODEL
================================================================================

Implementation of Random Forest Regressor for house price prediction.

Features:
    - Scikit-learn based implementation
    - Full compatibility with base model interface
    - Feature importance tracking
    - Parallel training support

Example:
    from src.models.random_forest import RandomForestModel
    
    model = RandomForestModel(config)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Dict, Any, Union
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Regressor model for house price prediction.
    
    Random Forest is an ensemble method that combines multiple decision trees.
    It's robust to outliers and doesn't require feature scaling.
    """
    
    def __init__(self, config: Any, hyperparams: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model.
        
        Args:
            config: Configuration object
            hyperparams (Optional[Dict]): Hyperparameter values
        """
        super().__init__('random_forest', config, hyperparams)
    
    def _build_model(self) -> None:
        """
        Build Random Forest model with configured hyperparameters.
        """
        logger.debug("Building Random Forest model...")
        
        self.model = RandomForestRegressor(**self.hyperparams)
    
    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """
        Train Random Forest model.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional arguments (unused for RF)
        """
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with Random Forest.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from trained model.
        
        Returns:
            Dict[str, float]: Feature importance mapping
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            # If feature names not available, use numeric indices
            feature_names = [f"Feature_{i}" for i in range(self.n_features_in)]
        else:
            feature_names = self.feature_names
        
        importances = self.model.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }
