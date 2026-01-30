"""
================================================================================
ENSEMBLE MODEL - MODEL FUSION AND STACKING
================================================================================

This module implements ensemble methods to combine predictions from multiple models.

Features:
    - Weighted averaging of model predictions
    - Stacking with meta-learner
    - Voting ensemble
    - Hyperparameter tuning for ensemble weights

Example:
    from src.models.ensemble import EnsembleModel
    
    models = [rf_model, xgb_model, catboost_model, lightgbm_model, tabnet_model]
    ensemble = EnsembleModel(models, config)
    predictions = ensemble.predict(X_test)
================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple base models.
    
    Strategies:
        1. Weighted averaging: Simple weighted average of predictions
        2. Stacking: Train meta-learner on base model predictions
        3. Voting: Democratic voting among models
    """
    
    def __init__(
        self,
        models: List[Any],
        config: Any,
        strategy: str = 'weighted_average'
    ):
        """
        Initialize ensemble model.
        
        Args:
            models (List): List of trained base models
            config: Configuration object
            strategy (str): Ensemble strategy ('weighted_average', 'stacking', 'voting')
        """
        self.models = models
        self.config = config
        self.strategy = strategy
        
        # Get weights from config
        ensemble_config = config.models.ensemble
        self.weights = ensemble_config.weights
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Meta-learner for stacking
        self.meta_learner = None
        
        logger.info(f"Initialized EnsembleModel with {strategy} strategy")
        logger.info(f"Weights: {self.weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if self.strategy == 'weighted_average':
            return self._predict_weighted_average(X)
        elif self.strategy == 'stacking':
            return self._predict_stacking(X)
        elif self.strategy == 'voting':
            return self._predict_voting(X)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")
    
    def _predict_weighted_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weighted average ensemble prediction.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Weighted average predictions
        """
        predictions = []
        weights_list = []
        
        for model in self.models:
            model_name = model.model_name
            
            # Get model predictions
            preds = model.predict(X)
            predictions.append(preds)
            
            # Get weight for this model
            weight = self.weights.get(model_name, 1.0 / len(self.models))
            weights_list.append(weight)
        
        # Compute weighted average
        predictions = np.array(predictions)
        weights_array = np.array(weights_list).reshape(-1, 1)
        
        ensemble_preds = np.average(predictions, axis=0, weights=weights_list)
        
        return ensemble_preds
    
    def _predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """
        Stacking ensemble prediction using meta-learner.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Stacked predictions
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call fit() first.")
        
        # Get base model predictions
        meta_features = []
        for model in self.models:
            preds = model.predict(X)
            meta_features.append(preds.reshape(-1, 1))
        
        meta_X = np.hstack(meta_features)
        
        # Predict with meta-learner
        ensemble_preds = self.meta_learner.predict(meta_X)
        
        return ensemble_preds
    
    def _predict_voting(self, X: pd.DataFrame) -> np.ndarray:
        """
        Voting ensemble prediction (simple average).
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Voting predictions
        """
        predictions = []
        
        for model in self.models:
            preds = model.predict(X)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        ensemble_preds = np.mean(predictions, axis=0)
        
        return ensemble_preds
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """
        Fit ensemble model (meta-learner for stacking).
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            EnsembleModel: Self for method chaining
        """
        if self.strategy == 'stacking':
            logger.info("Training stacking meta-learner...")
            
            # Get base model predictions on training data
            meta_features = []
            for model in self.models:
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))
            
            meta_X = np.hstack(meta_features)
            
            # Train meta-learner (Ridge regression)
            self.meta_learner = Ridge(alpha=1.0)
            self.meta_learner.fit(meta_X, y)
            
            logger.info("Stacking meta-learner trained")
        
        return self
    
    def get_model_names(self) -> List[str]:
        """
        Get list of base model names.
        
        Returns:
            List[str]: Model names
        """
        return [model.model_name for model in self.models]
    
    def __repr__(self) -> str:
        """String representation of ensemble."""
        model_names = ', '.join(self.get_model_names())
        return f"EnsembleModel(strategy='{self.strategy}', models=[{model_names}])"
