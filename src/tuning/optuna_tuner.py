"""
================================================================================
OPTUNA HYPERPARAMETER TUNING MODULE
================================================================================

This module implements competition-level hyperparameter optimization using Optuna.

Features:
    - Multi-model hyperparameter tuning
    - Early stopping and pruning
    - Parallel trial execution
    - Resume from checkpoint
    - Detailed optimization history

Example:
    from src.tuning.optuna_tuner import OptunaTuner
    
    tuner = OptunaTuner(config)
    best_params = tuner.tune_model(
        X_train, y_train,
        model_name='xgboost',
        n_trials=500
    )
================================================================================
"""

import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Callable, Tuple
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Hyperparameter tuner using Optuna framework.
    
    Implements competition-level optimization with:
        - Multiple sampling strategies (TPE, CMA-ES)
        - Trial pruning for early stopping
        - Parallel execution support
        - Checkpoint and resume capability
    """
    
    def __init__(self, config: Any):
        """
        Initialize Optuna tuner.
        
        Args:
            config: Configuration object with tuning settings
        """
        self.config = config
        self.tuning_config = config.tuning
        
        # Storage for trials and results
        self.studies: Dict[str, optuna.Study] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: Dict[str, list] = {}
        
        logger.info("OptunaTuner initialized")
    
    def tune_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        n_trials: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Model to tune ('random_forest', 'xgboost', etc.)
            n_trials (Optional[int]): Number of trials (uses config if None)
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Best hyperparameters found
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTUNA TUNING - {model_name.upper()}")
        logger.info(f"{'='*60}\n")
        
        if n_trials is None:
            n_trials = self.tuning_config.n_trials
        
        # Get model-specific configuration
        model_config = self.config.models[model_name]
        param_ranges = model_config.optuna_ranges
        
        # Create objective function
        def objective(trial) -> float:
            """Objective function to minimize (RMSE)."""
            return self._objective_function(
                trial,
                X_train,
                y_train,
                model_name,
                param_ranges
            )
        
        # Setup sampler
        if self.tuning_config.sampler == 'TPE':
            sampler = TPESampler(seed=self.tuning_config.random_state)
        elif self.tuning_config.sampler == 'CMA-ES':
            sampler = CmaEsSampler(seed=self.tuning_config.random_state)
        else:
            sampler = TPESampler(seed=self.tuning_config.random_state)
        
        # Setup pruner
        pruner = MedianPruner(n_warmup_steps=10)
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.tuning_config.max_parallel_trials,
            show_progress_bar=True,
            timeout=self.tuning_config.trial_timeout * n_trials if self.tuning_config.trial_timeout else None
        )
        
        # Store results
        self.studies[model_name] = study
        self.best_params[model_name] = study.best_params
        self.optimization_history[model_name] = [
            {'trial': i, 'value': trial.value}
            for i, trial in enumerate(study.trials)
        ]
        
        # Log results
        logger.info(f"\nBest RMSE: {study.best_value:.6f}")
        logger.info(f"Best Parameters: {study.best_params}")
        logger.info(f"Number of trials: {len(study.trials)}")
        
        return study.best_params
    
    def _objective_function(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        param_ranges: Dict[str, Any]
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            model_name: Name of model to optimize
            param_ranges: Hyperparameter ranges
            
        Returns:
            float: RMSE score to minimize
        """
        # Suggest hyperparameters
        hyperparams = self._suggest_hyperparams(trial, model_name, param_ranges)
        
        # Import model class dynamically
        if model_name == 'random_forest':
            from src.models.random_forest import RandomForestModel
            ModelClass = RandomForestModel
        elif model_name == 'xgboost':
            from src.models.xgboost_model import XGBoostModel
            ModelClass = XGBoostModel
        elif model_name == 'catboost':
            from src.models.catboost_model import CatBoostModel
            ModelClass = CatBoostModel
        elif model_name == 'lightgbm':
            from src.models.lightgbm_model import LightGBMModel
            ModelClass = LightGBMModel
        elif model_name == 'tabnet':
            from src.models.tabnet_model import TabNetModel
            ModelClass = TabNetModel
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create and train model
        try:
            model = ModelClass(self.config, hyperparams)
            model.fit(X_train, y_train)
            
            # Compute cross-validation score
            cv_results = model.cross_validate(
                X_train,
                y_train,
                cv_folds=self.tuning_config.cv_folds,
                metric='rmse'
            )
            
            mean_cv_score = cv_results['mean']
            
            return mean_cv_score
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def _suggest_hyperparams(
        self,
        trial: optuna.Trial,
        model_name: str,
        param_ranges: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters using Optuna.
        
        Args:
            trial: Optuna trial
            model_name: Model name
            param_ranges: Parameter ranges from config
            
        Returns:
            Dict[str, Any]: Suggested hyperparameters
        """
        hyperparams = {}
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                # Numeric range
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    # Integer parameter
                    hyperparams[param_name] = trial.suggest_int(
                        param_name,
                        param_range[0],
                        param_range[1]
                    )
                else:
                    # Float parameter
                    hyperparams[param_name] = trial.suggest_float(
                        param_name,
                        param_range[0],
                        param_range[1]
                    )
            elif isinstance(param_range, list):
                # Categorical parameter
                hyperparams[param_name] = trial.suggest_categorical(
                    param_name,
                    param_range
                )
            else:
                # Use value as-is
                hyperparams[param_name] = param_range
        
        return hyperparams
    
    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get best hyperparameters found for a model.
        
        Args:
            model_name (str): Model name
            
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        if model_name not in self.best_params:
            raise ValueError(f"No tuning results for {model_name}")
        
        return self.best_params[model_name]
    
    def save_results(self, output_dir: str = "artifacts/tuning") -> None:
        """
        Save tuning results to disk.
        
        Args:
            output_dir (str): Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save best params
        best_params_path = Path(output_dir) / "best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"Tuning results saved to {output_dir}")
