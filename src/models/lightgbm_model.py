"""
================================================================================
LIGHTGBM MODEL
================================================================================

Implementation of LightGBM Regressor for house price prediction.

Features:
    - Faster training than XGBoost
    - Lower memory usage
    - GPU acceleration support
    - Excellent for large datasets

Example:
    from src.models.lightgbm_model import LightGBMModel
    
    model = LightGBMModel(config)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
================================================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional, Dict, Any
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """
    Light GBM (Gradient Boosting Machine) Regressor.
    
    LightGBM is a fast, distributed, high-performance gradient boosting
    framework for classification, regression and ranking tasks.
    """
    
    def __init__(self, config: Any, hyperparams: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model.
        
        Args:
            config: Configuration object
            hyperparams (Optional[Dict]): Hyperparameter values
        """
        super().__init__('lightgbm', config, hyperparams)
    
    def _build_model(self) -> None:
        """
        Build LightGBM model with configured hyperparameters.
        """
        logger.debug("Building LightGBM model...")
        
        model_params = self.hyperparams.copy()
        
        # Setup GPU if available
        if self.device.type == 'cuda':
            model_params['device'] = 'gpu'
            model_params['gpu_device_id'] = self.device.gpu_index
        
        self.model = lgb.LGBMRegressor(**model_params)
    
    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """
        Train LightGBM model.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional arguments
        """
        eval_set = kwargs.get('eval_set', None)
        
        fit_kwargs = {'verbose': -1}
        
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['callbacks'] = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        
        self.model.fit(X, y, **fit_kwargs)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with LightGBM.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from LightGBM model.
        
        Returns:
            Dict[str, float]: Feature importance mapping
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.n_features_in)]
        else:
            feature_names = self.feature_names
        
        importances = self.model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }
