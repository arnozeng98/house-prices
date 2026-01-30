"""
================================================================================
MODEL EVALUATION AND METRICS MODULE
================================================================================

This module computes evaluation metrics and performs model assessment.

Metrics computed:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score
    - MAPE (Mean Absolute Percentage Error)

Features:
    - Cross-validation support
    - Metric tracking and logging
    - Comparison with ground truth

Example:
    from src.evaluate import evaluate_model, compute_metrics
    
    metrics = compute_metrics(y_true, y_pred)
    print(f"RMSE: {metrics['rmse']:.4f}")
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import KFold, cross_validate
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        prefix (str): Prefix for metric names (e.g., 'val_' for validation)
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Ensure arrays are 1D
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - avoid division by zero
    mask = y_true != 0
    if mask.any():
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    else:
        mape = np.inf
    
    metrics = {
        f'{prefix}rmse': rmse,
        f'{prefix}mae': mae,
        f'{prefix}r2': r2,
        f'{prefix}mape': mape
    }
    
    return metrics


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Evaluate model on train/test data and compute cross-validation scores.
    
    Args:
        model: Trained model object
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (Optional[pd.DataFrame]): Test features
        y_test (Optional[pd.Series]): Test target
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    results = {}
    
    # Training metrics
    train_preds = model.predict(X_train)
    train_metrics = compute_metrics(y_train.values, train_preds, prefix='train_')
    results.update(train_metrics)
    
    # Test metrics
    if X_test is not None and y_test is not None:
        test_preds = model.predict(X_test)
        test_metrics = compute_metrics(y_test.values, test_preds, prefix='test_')
        results.update(test_metrics)
    
    # Cross-validation metrics
    if cv_folds > 1:
        cv_results = model.cross_validate(X_train, y_train, cv_folds=cv_folds)
        results['cv_scores'] = cv_results
    
    return results


def compare_with_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    merge_on: str = 'Id'
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Compare predictions with ground truth data.
    
    Args:
        predictions (pd.DataFrame): Predictions with columns [Id, SalePrice]
        ground_truth (pd.DataFrame): Ground truth with columns [Id, SalePrice]
        merge_on (str): Column to merge on
        
    Returns:
        Tuple[Dict[str, float], pd.DataFrame]: Metrics and comparison DataFrame
    """
    logger.info("Comparing predictions with ground truth...")
    
    # Merge predictions and ground truth
    comparison = predictions.merge(
        ground_truth,
        on=merge_on,
        suffixes=('_pred', '_true')
    )
    
    y_true = comparison['SalePrice_true'].values
    y_pred = comparison['SalePrice_pred'].values
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Compute errors
    comparison['Absolute_Error'] = np.abs(y_true - y_pred)
    comparison['Relative_Error'] = np.abs((y_true - y_pred) / y_true)
    
    logger.info(f"Ground Truth Comparison:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    
    return metrics, comparison


def generate_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Generate detailed error analysis.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, Any]: Error analysis
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    analysis = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'median_abs_error': float(np.median(abs_errors)),
        'percentile_90_error': float(np.percentile(abs_errors, 90)),
        'percentile_95_error': float(np.percentile(abs_errors, 95)),
    }
    
    return analysis


def create_evaluation_report(
    results: Dict[str, Any],
    model_name: str
) -> pd.DataFrame:
    """
    Create a formatted evaluation report.
    
    Args:
        results (Dict[str, Any]): Evaluation results
        model_name (str): Model name
        
    Returns:
        pd.DataFrame: Formatted report
    """
    report_data = []
    
    for metric_name, metric_value in results.items():
        if not isinstance(metric_value, dict):
            report_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': metric_value if isinstance(metric_value, (int, float)) else str(metric_value)
            })
    
    report_df = pd.DataFrame(report_data)
    return report_df
