"""
================================================================================
GROUND TRUTH COMPARISON MODULE
================================================================================

This module compares model predictions with ground truth data
and logs detailed results.

Features:
    - Metric computation against ground truth
    - Detailed error analysis
    - Ranking of models by performance
    - RMSE logging for all models

Example:
    from src.evaluation.comparison import GroundTruthComparison
    
    comparator = GroundTruthComparison(config)
    results = comparator.compare_all_models(
        predictions_dict,
        ground_truth_df
    )
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from src.evaluate import compute_metrics, compare_with_ground_truth
import logging

logger = logging.getLogger(__name__)


class GroundTruthComparison:
    """
    Compares model predictions with ground truth data.
    """
    
    def __init__(self, config: Any):
        """
        Initialize ground truth comparator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.comparison_results: Dict[str, Dict[str, float]] = {}
    
    def compare_models(
        self,
        predictions_dict: Dict[str, pd.DataFrame],
        ground_truth: pd.DataFrame,
        merge_on: str = 'Id'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models against ground truth.
        
        Args:
            predictions_dict (Dict[str, pd.DataFrame]): Model predictions
            ground_truth (pd.DataFrame): Ground truth data
            merge_on (str): Column to merge on
            
        Returns:
            Dict[str, Dict[str, float]]: Comparison results
        """
        logger.info("\n" + "="*70)
        logger.info("GROUND TRUTH COMPARISON - ALL MODELS")
        logger.info("="*70 + "\n")
        
        results = {}
        
        for model_name, predictions in predictions_dict.items():
            logger.info(f"Comparing {model_name}...")
            
            # Compare with ground truth
            metrics, _ = compare_with_ground_truth(
                predictions,
                ground_truth,
                merge_on=merge_on
            )
            
            results[model_name] = metrics
            self.comparison_results[model_name] = metrics
        
        # Log summary
        self._log_summary(results)
        
        return results
    
    def _log_summary(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Log a summary of all results.
        
        Args:
            results (Dict[str, Dict[str, float]]): Comparison results
        """
        logger.info("\n" + "="*70)
        logger.info("SUMMARY - RANKED BY RMSE")
        logger.info("="*70 + "\n")
        
        # Sort by RMSE
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['rmse']
        )
        
        # Log rankings
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            logger.info(
                f"{rank:2d}. {model_name:20s} | "
                f"RMSE: {metrics['rmse']:10.4f} | "
                f"MAE: {metrics['mae']:10.4f} | "
                f"R²: {metrics['r2']:10.4f}"
            )
        
        logger.info("\n" + "="*70 + "\n")
        
        # Log best model
        best_model = sorted_results[0][0]
        best_rmse = sorted_results[0][1]['rmse']
        
        logger.info(f"🏆 Best Model: {best_model.upper()}")
        logger.info(f"   RMSE: {best_rmse:.6f}\n")
    
    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Get best performing model.
        
        Returns:
            Tuple[str, Dict[str, float]]: Model name and metrics
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available")
        
        best_model = min(
            self.comparison_results.items(),
            key=lambda x: x[1]['rmse']
        )
        
        return best_model
    
    def get_ranking(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Get models ranked by RMSE.
        
        Returns:
            List[Tuple[str, Dict[str, float]]]: Ranked models
        """
        return sorted(
            self.comparison_results.items(),
            key=lambda x: x[1]['rmse']
        )
