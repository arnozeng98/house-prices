"""
================================================================================
VISUALIZATION MODULE - PLOTS AND CHARTS
================================================================================

This module generates visualizations for model analysis and comparison.

Plots generated:
    - Feature importance charts
    - Prediction vs actual scatter plots
    - Error distribution histograms
    - Model comparison bar charts
    - Correlation heatmaps
    - Optuna optimization history

All plots are saved to data/img/

Example:
    from src.visualization.plots import ModelVisualizer
    
    visualizer = ModelVisualizer(config)
    visualizer.plot_feature_importance(model, X_train)
    visualizer.plot_predictions(y_true, y_pred, model_name='xgboost')
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Generates visualizations for model analysis and diagnostics.
    """
    
    def __init__(self, config: Any):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.viz_config = config.visualization
        
        # Create output directory
        self.output_dir = Path(config.paths.images_data)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style(self.viz_config.style)
        plt.rcParams['figure.dpi'] = self.viz_config.dpi
        
        logger.info(f"ModelVisualizer initialized - output dir: {self.output_dir}")
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        model_name: str,
        top_n: int = 30
    ) -> None:
        """
        Plot feature importance chart.
        
        Args:
            importance_dict (Dict[str, float]): Feature importance mapping
            model_name (str): Name of the model
            top_n (int): Number of top features to display
        """
        logger.info(f"Plotting feature importance for {model_name}...")
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_features[:top_n]]
        importances = [x[1] for x in sorted_features[:top_n]]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.viz_config.figsize['feature_importance'])
        
        bars = ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"feature_importance_{model_name}.{self.viz_config.output_format}"
        plt.savefig(output_path, dpi=self.viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        phase: str = 'test'
    ) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name
            phase (str): 'train', 'val', or 'test'
        """
        logger.info(f"Plotting predictions for {model_name} ({phase} phase)...")
        
        fig, ax = plt.subplots(figsize=self.viz_config.figsize['prediction_scatter'])
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)', fontsize=12)
        ax.set_ylabel('Predicted Price ($)', fontsize=12)
        ax.set_title(f'Actual vs Predicted - {model_name} ({phase})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"predictions_{model_name}_{phase}.{self.viz_config.output_format}"
        plt.savefig(output_path, dpi=self.viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction plot saved to {output_path}")
    
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> None:
        """
        Plot distribution of prediction errors.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name
        """
        logger.info(f"Plotting error distribution for {model_name}...")
        
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=self.viz_config.figsize['error_distribution'])
        
        # Histogram of errors
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Prediction Error ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel('Prediction Error ($)', fontsize=12)
        axes[1].set_title('Error Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Prediction Error Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"error_distribution_{model_name}.{self.viz_config.output_format}"
        plt.savefig(output_path, dpi=self.viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution plot saved to {output_path}")
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'rmse'
    ) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            results (Dict[str, Dict[str, float]]): Model results
            metric (str): Metric to compare ('rmse', 'mae', 'r2')
        """
        logger.info(f"Plotting model comparison ({metric})...")
        
        # Extract metrics
        models = list(results.keys())
        values = [results[model].get(metric, 0) for model in models]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"model_comparison_{metric}.{self.viz_config.output_format}"
        plt.savefig(output_path, dpi=self.viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {output_path}")
    
    def plot_correlation_heatmap(
        self,
        X: pd.DataFrame,
        top_features: Optional[List[str]] = None
    ) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            top_features (Optional[List[str]]): Features to include (if None, uses all)
        """
        logger.info("Plotting correlation heatmap...")
        
        # Select features
        if top_features is not None:
            X_selected = X[top_features]
        else:
            # Select only numerical columns
            X_selected = X.select_dtypes(include=[np.number])
            # Limit to top 30 features for readability
            if X_selected.shape[1] > 30:
                X_selected = X_selected.iloc[:, :30]
        
        # Compute correlation
        corr_matrix = X_selected.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.viz_config.figsize['correlation_heatmap'])
        
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"correlation_heatmap.{self.viz_config.output_format}"
        plt.savefig(output_path, dpi=self.viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation heatmap saved to {output_path}")
    
    def plot_optuna_history(self, study: 'optuna.Study', model_name: str) -> None:
        """
        Plot Optuna optimization history.
        
        Args:
            study: Optuna Study object
            model_name (str): Model name
        """
        logger.info(f"Plotting Optuna history for {model_name}...")
        
        # Extract trial values
        trial_numbers = list(range(len(study.trials)))
        trial_values = [trial.value if trial.value is not None else float('inf') for trial in study.trials]
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(trial_numbers, trial_values, marker='o', linestyle='-', alpha=0.7, markersize=3)
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title(f'Optuna Optimization History - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"optuna_history_{model_name}.{self.viz_config.output_format}"
        plt.savefig(output_path, dpi=self.viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optuna history plot saved to {output_path}")
