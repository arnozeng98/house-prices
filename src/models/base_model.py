"""
================================================================================
BASE MODEL CLASS - FOUNDATION FOR ALL ML MODELS
================================================================================

This module defines the base model class that all specific models inherit from.
Provides common functionality like training, prediction, evaluation, and device management.

Features:
    - Device management (GPU/CPU with fallback)
    - Common model interface
    - Cross-validation support
    - Prediction caching
    - Model persistence

Example:
    from src.models.base_model import BaseModel
    
    class MyModel(BaseModel):
        def _build_model(self):
            # Initialize model
            self.model = SomeMLModel()
        
        def _train(self, X, y):
            # Train model
            self.model.fit(X, y)
        
        def _predict(self, X):
            # Make predictions
            return self.model.predict(X)
================================================================================
"""

import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Provides common interface and functionality for:
        - Device management (CPU/GPU)
        - Training and prediction
        - Cross-validation
        - Model persistence
        - Hyperparameter management
    """
    
    def __init__(
        self,
        model_name: str,
        config: Any,
        hyperparams: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base model.
        
        Args:
            model_name (str): Name of the model (e.g., 'xgboost', 'random_forest')
            config: Configuration object
            hyperparams (Optional[Dict]): Hyperparameter values
        """
        self.model_name = model_name
        self.config = config
        self.device = config.device
        
        # Model hyperparameters
        if hyperparams is None:
            self.hyperparams = config.models[model_name].params
        else:
            self.hyperparams = hyperparams
        
        # Model object (initialized by subclass)
        self.model = None
        
        # Training history
        self.is_trained = False
        self.train_score = None
        self.val_score = None
        self.cv_scores = None
        
        # Feature tracking
        self.n_features_in = None
        self.feature_names = None
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def _build_model(self) -> None:
        """
        Build and initialize the model.
        
        This method should be implemented by subclasses to create
        the specific model instance (e.g., XGBRegressor, RandomForestRegressor).
        """
        pass
    
    @abstractmethod
    def _train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> None:
        """
        Train the model on data.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional training arguments
        """
        pass
    
    @abstractmethod
    def _predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        pass
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[Tuple] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Fit model to training data.
        
        Args:
            X: Training features
            y: Training target
            eval_set (Optional[Tuple]): Validation data (X_val, y_val)
            **kwargs: Additional training arguments
            
        Returns:
            BaseModel: Self for method chaining
        """
        logger.info(f"Training {self.model_name}...")
        
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.n_features_in = X.shape[1]
            X_array = X.values
        else:
            self.n_features_in = X.shape[1]
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Build model if not already built
        if self.model is None:
            self._build_model()
        
        # Train model
        self._train(X_array, y_array, **kwargs)
        
        # Compute training score
        train_preds = self._predict(X_array)
        self.train_score = self._compute_metrics(y_array, train_preds)
        
        # Compute validation score if provided
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            val_preds = self._predict(X_val)
            self.val_score = self._compute_metrics(y_val, val_preds)
            logger.info(f"Validation RMSE: {self.val_score['rmse']:.6f}")
        
        self.is_trained = True
        logger.info(f"Training score (RMSE): {self.train_score['rmse']:.6f}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} must be trained first. Call fit() before predict()")
        
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        predictions = self._predict(X_array)
        
        return predictions
    
    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv_folds: int = 5,
        metric: str = 'rmse'
    ) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Training features
            y: Training target
            cv_folds (int): Number of CV folds
            metric (str): Metric to compute ('rmse', 'mae', 'r2')
            
        Returns:
            Dict[str, float]: CV scores dictionary
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.device.seed)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_array)):
            X_train_fold = X_array[train_idx]
            X_val_fold = X_array[val_idx]
            y_train_fold = y_array[train_idx]
            y_val_fold = y_array[val_idx]
            
            # Train on fold
            model_fold = self.__class__(
                self.model_name,
                self.config,
                self.hyperparams
            )
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Predict on fold
            fold_preds = model_fold.predict(X_val_fold)
            
            # Compute metric
            if metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_val_fold, fold_preds))
            elif metric == 'mae':
                score = mean_absolute_error(y_val_fold, fold_preds)
            elif metric == 'r2':
                score = r2_score(y_val_fold, fold_preds)
            else:
                score = np.sqrt(mean_squared_error(y_val_fold, fold_preds))
            
            cv_scores.append(score)
            logger.info(f"Fold {fold+1}: {metric.upper()}={score:.6f}")
        
        self.cv_scores = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'metric': metric
        }
        
        return self.cv_scores
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict[str, float]: Metrics dictionary
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path (str): File path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path (str): File path to load model from
        """
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        
        Returns:
            Dict[str, Any]: Hyperparameter dictionary
        """
        return self.hyperparams.copy()
    
    def set_hyperparams(self, hyperparams: Dict[str, Any]) -> 'BaseModel':
        """
        Set model hyperparameters.
        
        Args:
            hyperparams (Dict[str, Any]): New hyperparameter values
            
        Returns:
            BaseModel: Self for method chaining
        """
        self.hyperparams = hyperparams
        self.model = None  # Reset model so it's rebuilt with new params
        self.is_trained = False
        return self
    
    def __repr__(self) -> str:
        """String representation of model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', status='{status}')"
