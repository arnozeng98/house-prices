"""
================================================================================
TABNET MODEL - DEEP LEARNING FOR TABULAR DATA WITH CUDA SUPPORT
================================================================================

Implementation of TabNet (Neural Network for Tabular Data) with CUDA support.

Features:
    - Deep learning for tabular data
    - GPU acceleration with CUDA (automatic CPU fallback)
    - Sequential attention mechanism
    - Feature importance tracking
    - Early stopping and batch training

Example:
    from src.models.tabnet_model import TabNetModel
    
    model = TabNetModel(config)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    predictions = model.predict(X_test)
================================================================================
"""

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from typing import Optional, Dict, Any, Tuple
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class TabNetModel(BaseModel):
    """
    TabNet (Attentive Interpretable Tabular Learning) Model.
    
    TabNet is a deep learning architecture for tabular data that uses
    sequential attention to choose which features to reason from at each step.
    
    Features:
        - GPU acceleration with automatic CPU fallback
        - High-quality predictions on tabular data
        - Feature importance from attention masks
    """
    
    def __init__(self, config: Any, hyperparams: Optional[Dict[str, Any]] = None):
        """
        Initialize TabNet model.
        
        Args:
            config: Configuration object
            hyperparams (Optional[Dict]): Hyperparameter values
        """
        super().__init__('tabnet', config, hyperparams)
        
        # TabNet specific settings
        self._setup_device()
    
    def _setup_device(self) -> None:
        """
        Setup device for TabNet (GPU with fallback to CPU).
        
        TabNet uses PyTorch, so we ensure CUDA availability before using GPU.
        """
        try:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                self.device_type = 'cuda'
                self.device_id = self.device.gpu_index
                logger.info(f"TabNet will use GPU {self.device_id}")
            else:
                self.device_type = 'cpu'
                self.device_id = 0
                logger.info("TabNet will use CPU")
        except Exception as e:
            logger.warning(f"Error setting up device: {e} - falling back to CPU")
            self.device_type = 'cpu'
            self.device_id = 0
    
    def _build_model(self) -> None:
        """
        Build TabNet model with configured hyperparameters.
        """
        logger.debug("Building TabNet model...")
        
        model_params = self.hyperparams.copy()
        
        # Setup device parameters
        if self.device_type == 'cuda':
            model_params['device_type'] = 'cuda'
            model_params['device_name'] = f'cuda:{self.device_id}'
        else:
            model_params['device_type'] = 'cpu'
        
        try:
            self.model = TabNetRegressor(**model_params)
        except Exception as e:
            logger.warning(f"Error creating TabNet with GPU: {e} - trying CPU fallback")
            model_params['device_type'] = 'cpu'
            self.model = TabNetRegressor(**model_params)
    
    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """
        Train TabNet model.
        
        Args:
            X: Training features (numpy array)
            y: Training target (numpy array)
            **kwargs: Additional arguments (eval_set, etc.)
        """
        eval_set = kwargs.get('eval_set', None)
        eval_set_tabnet = None
        
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_set_tabnet = (X_val, y_val)
        
        try:
            self.model.fit(
                X, y,
                eval_set=eval_set_tabnet,
                eval_metric=['rmse'],
                max_epochs=self.hyperparams.get('max_epochs', 200),
                patience=self.hyperparams.get('patience', 20),
                batch_size=self.hyperparams.get('batch_size', 256),
                virtual_batch_size=self.hyperparams.get('virtual_batch_size', 128),
                num_workers=0,
                drop_last=False,
                verbose=0
            )
        except RuntimeError as e:
            if 'cuda' in str(e).lower():
                logger.warning(f"CUDA error during training: {e}")
                logger.warning("Attempting to retrain on CPU...")
                
                # Rebuild model on CPU
                self.device_type = 'cpu'
                model_params = self.hyperparams.copy()
                model_params['device_type'] = 'cpu'
                self.model = TabNetRegressor(**model_params)
                
                # Retry training
                self.model.fit(
                    X, y,
                    eval_set=eval_set_tabnet,
                    max_epochs=self.hyperparams.get('max_epochs', 200),
                    patience=self.hyperparams.get('patience', 20),
                    batch_size=self.hyperparams.get('batch_size', 256),
                    virtual_batch_size=self.hyperparams.get('virtual_batch_size', 128),
                    num_workers=0,
                    drop_last=False,
                    verbose=0
                )
            else:
                raise
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with TabNet.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        try:
            predictions = self.model.predict(X)
        except RuntimeError as e:
            if 'cuda' in str(e).lower():
                logger.warning(f"CUDA error during prediction: {e} - using CPU")
                # Move model to CPU if error occurs
                self.model.device_type = 'cpu'
                predictions = self.model.predict(X)
            else:
                raise
        
        # Ensure return type is numpy array
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        return predictions.flatten()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from TabNet attention masks.
        
        TabNet provides feature importance based on the attention masks
        used in the sequential decision process.
        
        Returns:
            Dict[str, float]: Feature importance mapping
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.n_features_in)]
        else:
            feature_names = self.feature_names
        
        try:
            # Get feature importance from TabNet
            importances = self.model.feature_importances_
            
            return {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }
        except Exception as e:
            logger.warning(f"Error getting TabNet feature importance: {e}")
            # Return uniform importance if error occurs
            uniform_importance = 1.0 / len(feature_names)
            return {name: uniform_importance for name in feature_names}
    
    def get_attention_masks(self) -> np.ndarray:
        """
        Get the attention masks from TabNet (advanced feature).
        
        Returns:
            np.ndarray: Attention masks showing feature selection over steps
        """
        if not hasattr(self.model, 'attention_masks'):
            return None
        
        return self.model.attention_masks
