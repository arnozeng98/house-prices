"""
================================================================================
CATBOOST MODEL
================================================================================

Implementation of CatBoost Regressor for house price prediction.

Features:
    - Excellent handling of categorical features (no pre-encoding needed)
    - GPU acceleration support
    - Fast training and inference
    - Feature importance and SHAP values

Example:
    from src.models.catboost_model import CatBoostModel
    
    model = CatBoostModel(config)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
================================================================================
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from typing import Optional, Dict, Any
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    """
    CatBoost (Categorical Boosting) Regressor.
    
    CatBoost is specifically designed to work with categorical features
    and often produces excellent results with minimal tuning.
    """
    
    def __init__(self, config: Any, hyperparams: Optional[Dict[str, Any]] = None):
        """
        Initialize CatBoost model.
        
        Args:
            config: Configuration object
            hyperparams (Optional[Dict]): Hyperparameter values
        """
        super().__init__('catboost', config, hyperparams)
    
    def _build_model(self) -> None:
        """
        Build CatBoost model with configured hyperparameters.
        """
        logger.debug("Building CatBoost model...")
        
        model_params = self.hyperparams.copy()
        
        # Setup GPU if available
        if self.device.type == 'cuda':
            model_params['task_type'] = 'GPU'
            model_params['gpu_device_ids'] = [self.device.gpu_index]
        else:
            model_params['task_type'] = 'CPU'
        
        self.model = CatBoostRegressor(**model_params)
    
    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """
        Train CatBoost model.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional arguments
        """
        eval_set = kwargs.get('eval_set', None)
        
        fit_kwargs = {'verbose': 0}
        
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs['eval_set'] = (X_val, y_val)
        
        self.model.fit(X, y, **fit_kwargs)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with CatBoost.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from CatBoost model.
        
        Returns:
            Dict[str, float]: Feature importance mapping
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.n_features_in)]
        else:
            feature_names = self.feature_names
        
        importances = self.model.get_feature_importance()
        
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }
