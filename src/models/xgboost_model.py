"""
================================================================================
XGBOOST MODEL
================================================================================

Implementation of XGBoost Regressor for house price prediction.

Features:
    - Industry-standard gradient boosting
    - Early stopping support
    - Feature importance tracking
    - GPU acceleration support

Example:
    from src.models.xgboost_model import XGBoostModel
    
    model = XGBoostModel(config)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    predictions = model.predict(X_test)
================================================================================
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, Any, Union, Tuple
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost (eXtreme Gradient Boosting) Regressor.
    
    XGBoost is a highly optimized gradient boosting framework that often
    achieves state-of-the-art performance in machine learning competitions.
    """
    
    def __init__(self, config: Any, hyperparams: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Configuration object
            hyperparams (Optional[Dict]): Hyperparameter values
        """
        super().__init__('xgboost', config, hyperparams)
        
        # XGBoost specific attributes
        self.early_stopping_rounds = 50
        self.eval_metric = 'rmse'
    
    def _build_model(self) -> None:
        """
        Build XGBoost model with configured hyperparameters.
        """
        logger.debug("Building XGBoost model...")
        
        # Setup GPU device if available
        tree_method = 'gpu_hist' if self.device.type == 'cuda' else 'hist'
        gpu_id = self.device.gpu_index if self.device.type == 'cuda' else 0
        
        model_params = self.hyperparams.copy()
        
        # Add device parameters
        model_params['tree_method'] = tree_method
        model_params['gpu_id'] = gpu_id
        
        self.model = xgb.XGBRegressor(**model_params)
    
    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """
        Train XGBoost model with optional early stopping.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional arguments (eval_set, eval_metric, etc.)
        """
        # Extract eval_set if provided
        eval_set = kwargs.get('eval_set', None)
        
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_set = [(X_val, y_val)]
            
            self.model.fit(
                X, y,
                eval_set=eval_set,
                eval_metric=self.eval_metric,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with XGBoost.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from XGBoost model.
        
        Returns:
            Dict[str, float]: Feature importance mapping
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.n_features_in)]
        else:
            feature_names = self.feature_names
        
        # Get feature importance from model
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        
        # Create complete importance mapping
        importances = {}
        for name in feature_names:
            # XGBoost uses 'f0', 'f1', etc. for feature indices
            idx = feature_names.index(name) if name in feature_names else -1
            xgb_key = f'f{idx}' if idx >= 0 else name
            importances[name] = importance_dict.get(xgb_key, 0)
        
        return importances
