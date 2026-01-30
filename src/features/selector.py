"""
================================================================================
FEATURE SELECTION MODULE - AUTOMATIC FEATURE SELECTION
================================================================================

This module implements multiple feature selection strategies:
    - Boruta feature selection (wrapper method)
    - Permutation importance
    - Mutual information
    - Correlation-based selection

Features:
    - Multiple selection strategies
    - Configurable thresholds
    - Feature importance tracking
    - Reproducible selection

Example:
    from src.features.selector import FeatureSelector
    
    selector = FeatureSelector(config)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_selected_features()
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Set, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Automatically selects the most important features using multiple strategies.
    
    Strategies:
        1. Boruta (wrapper-based feature selection)
        2. Permutation importance
        3. Mutual information
        4. Combined voting
    """
    
    def __init__(self, config: Any):
        """
        Initialize feature selector.
        
        Args:
            config: Configuration object with feature selection settings
        """
        self.config = config
        self.selection_config = config.feature_selection
        
        # Track selected features
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Dict[str, float] = {}
        self.selection_reason: Dict[str, str] = {}
        
        logger.info("FeatureSelector initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit feature selector to data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            FeatureSelector: Self for method chaining
        """
        logger.info("Fitting FeatureSelector...")
        
        if not self.selection_config.enabled:
            logger.info("Feature selection disabled - using all features")
            self.selected_features = X.columns.tolist()
            return self
        
        # Initialize importance scores for voting
        feature_votes: Dict[str, int] = {col: 0 for col in X.columns}
        importance_scores: Dict[str, List[float]] = {col: [] for col in X.columns}
        
        # Boruta selection
        if self.selection_config.boruta.enabled:
            logger.info("Running Boruta feature selection...")
            try:
                boruta_features = self._boruta_selection(X, y)
                for feat in boruta_features:
                    feature_votes[feat] += 1
                    self.selection_reason[feat] = "Boruta"
            except Exception as e:
                logger.warning(f"Boruta selection failed: {e}")
        
        # Permutation importance
        if self.selection_config.permutation_importance.enabled:
            logger.info("Computing permutation importance...")
            try:
                perm_features, perm_scores = self._permutation_importance(X, y)
                for feat, score in zip(perm_features, perm_scores):
                    feature_votes[feat] += 1
                    importance_scores[feat].append(score)
                    if feat not in self.selection_reason:
                        self.selection_reason[feat] = "Permutation"
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")
        
        # Mutual information
        if self.selection_config.mutual_information.enabled:
            logger.info("Computing mutual information...")
            try:
                mi_features, mi_scores = self._mutual_information(X, y)
                for feat, score in zip(mi_features, mi_scores):
                    feature_votes[feat] += 1
                    importance_scores[feat].append(score)
                    if feat not in self.selection_reason:
                        self.selection_reason[feat] = "Mutual Information"
            except Exception as e:
                logger.warning(f"Mutual information failed: {e}")
        
        # Select features based on voting
        self._select_features_by_voting(
            feature_votes,
            importance_scores,
            X.columns.tolist()
        )
        
        logger.info(f"Selected {len(self.selected_features)} features "
                   f"(from {len(X.columns)} total)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from data.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fit first")
        
        # Keep only selected features that exist in X
        available_features = [f for f in self.selected_features if f in X.columns]
        return X[available_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit selector and select features in one step.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            pd.DataFrame: Selected features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _boruta_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Perform Boruta feature selection.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            List[str]: Selected feature names
        """
        try:
            from boruta import BorutaPy
        except ImportError:
            logger.warning("Boruta not available - skipping")
            return []
        
        # Use Random Forest as estimator
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config.device.seed,
            n_jobs=-1
        )
        
        # Run Boruta
        boruta = BorutaPy(
            rf,
            n_estimators='auto',
            max_iter=self.selection_config.boruta.max_iterations,
            random_state=self.config.device.seed,
            verbose=0
        )
        
        boruta.fit(X.values, y.values)
        
        # Get selected features
        selected_mask = boruta.support_
        selected_features = X.columns[selected_mask].tolist()
        
        logger.info(f"Boruta selected {len(selected_features)} features")
        
        return selected_features
    
    def _permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[List[str], List[float]]:
        """
        Compute permutation importance.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            Tuple[List[str], List[float]]: Feature names and importance scores
        """
        # Train a model
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config.device.seed,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Compute permutation importance
        perm_result = permutation_importance(
            rf,
            X,
            y,
            n_repeats=self.selection_config.permutation_importance.n_repeats,
            random_state=self.config.device.seed,
            n_jobs=-1
        )
        
        # Get threshold
        threshold_pctl = self.selection_config.permutation_importance.threshold_percentile
        threshold = np.percentile(perm_result.importances_mean, threshold_pctl)
        
        # Select features above threshold
        selected_mask = perm_result.importances_mean > threshold
        selected_features = X.columns[selected_mask].tolist()
        selected_scores = perm_result.importances_mean[selected_mask].tolist()
        
        logger.info(f"Permutation importance selected {len(selected_features)} features")
        
        return selected_features, selected_scores
    
    def _mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[List[str], List[float]]:
        """
        Compute mutual information scores.
        
        Args:
            X (pd.DataFrame): Training features (must be numerical)
            y (pd.Series): Training target
            
        Returns:
            Tuple[List[str], List[float]]: Feature names and MI scores
        """
        # Only work with numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[numerical_cols]
        
        # Compute MI
        mi_scores = mutual_info_regression(X_num, y, random_state=self.config.device.seed)
        
        # Get threshold
        threshold_pctl = self.selection_config.mutual_information.threshold_percentile
        threshold = np.percentile(mi_scores, threshold_pctl)
        
        # Select features above threshold
        selected_mask = mi_scores > threshold
        selected_features = X_num.columns[selected_mask].tolist()
        selected_scores = mi_scores[selected_mask].tolist()
        
        logger.info(f"Mutual information selected {len(selected_features)} features")
        
        return selected_features, selected_scores
    
    def _select_features_by_voting(
        self,
        feature_votes: Dict[str, int],
        importance_scores: Dict[str, List[float]],
        all_features: List[str]
    ) -> None:
        """
        Select features based on voting from multiple methods.
        
        Args:
            feature_votes (Dict[str, int]): Vote count for each feature
            importance_scores (Dict[str, List[float]]): Importance scores
            all_features (List[str]): All feature names
        """
        # Ensure minimum and maximum feature counts
        min_features = self.selection_config.min_features
        max_features = self.selection_config.max_features
        
        # Sort by vote count and importance
        scored_features = []
        for feature in all_features:
            votes = feature_votes.get(feature, 0)
            avg_score = np.mean(importance_scores[feature]) if importance_scores[feature] else 0
            scored_features.append((feature, votes, avg_score))
        
        # Sort by votes (descending) then by average score
        scored_features.sort(key=lambda x: (-x[1], -x[2]))
        
        # Select features
        n_select = min(
            max(min_features, len(scored_features) // 2),
            min(max_features, len(scored_features))
        )
        
        self.selected_features = [f[0] for f in scored_features[:n_select]]
        
        # Store importance scores
        for feature, votes, avg_score in scored_features:
            self.feature_importance[feature] = avg_score
    
    def get_selected_features(self) -> List[str]:
        """
        Get list of selected feature names.
        
        Returns:
            List[str]: Selected feature names
        """
        return self.selected_features.copy() if self.selected_features else []
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """
        Get ranking of features by importance.
        
        Returns:
            pd.DataFrame: Features ranked by importance
        """
        importance_df = pd.DataFrame({
            'Feature': list(self.feature_importance.keys()),
            'Importance': list(self.feature_importance.values()),
            'Reason': [self.selection_reason.get(f, 'Unknown') for f in self.feature_importance.keys()]
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        return importance_df
