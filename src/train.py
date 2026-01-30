#!/usr/bin/env python
"""
================================================================================
MAIN TRAINING SCRIPT - AMES HOUSING PRICE PREDICTION
================================================================================

This is the primary training script that orchestrates the complete machine
learning pipeline including:
    1. Data loading and preprocessing
    2. Feature engineering and selection
    3. Model training with hyperparameter tuning (Optuna)
    4. Model evaluation and visualization
    5. Result logging and comparison with ground truth

Usage:
    python src/train.py

Author: Arno
Version: 1.0.0
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.logging_config import setup_logging, get_logger, log_section, log_metrics, log_model_comparison
from src.utils.helpers import Timer, get_progress_bar
from src.io import generate_ground_truth, validate_ground_truth
from src.preprocessing import PreprocessingPipeline
from src.features import FeatureEngineer, FeatureSelector
from src.models import (
    RandomForestModel, XGBoostModel, CatBoostModel,
    LightGBMModel, TabNetModel, EnsembleModel
)
from src.tuning import OptunaTuner
from src.evaluate import evaluate_model
from src.evaluation import GroundTruthComparison
from src.visualization import ModelVisualizer

# Global variables
logger = None
config = None


def load_data():
    """
    Load training and test data from CSV files.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: X_train, X_test, y_train
    """
    log_section("DATA LOADING")
    
    # Load training data
    logger.info(f"Loading training data from {config.paths.train_csv}...")
    train_df = pd.read_csv(config.paths.train_csv)
    logger.info(f"Training data shape: {train_df.shape}")
    
    # Load test data
    logger.info(f"Loading test data from {config.paths.test_csv}...")
    test_df = pd.read_csv(config.paths.test_csv)
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Split features and target
    X_train = train_df.drop('SalePrice', axis=1)
    y_train = train_df['SalePrice']
    X_test = test_df.copy()
    
    # Remove ID columns for modeling
    X_train = X_train.drop('Id', axis=1)
    test_ids = X_test['Id'].values
    X_test = X_test.drop('Id', axis=1)
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, test_ids


def preprocess_data(X_train, X_test, y_train):
    """
    Preprocess data (cleaning, encoding, scaling).
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed X_train, X_test
    """
    log_section("DATA PREPROCESSING")
    
    with Timer(name="Preprocessing"):
        pipeline = PreprocessingPipeline(config)
        X_train_processed, X_test_processed = pipeline.fit_transform(X_train, X_test, y_train)
    
    logger.info(f"Processed data - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, pipeline


def engineer_features(X_train, X_test):
    """
    Create engineered features.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features with engineered columns
    """
    log_section("FEATURE ENGINEERING")
    
    with Timer(name="Feature Engineering"):
        engineer = FeatureEngineer(config)
        X_train_eng = engineer.fit_transform(X_train)
        X_test_eng = engineer.transform(X_test)
    
    logger.info(f"Engineered features - Train: {X_train_eng.shape}, Test: {X_test_eng.shape}")
    
    return X_train_eng, X_test_eng, engineer


def select_features(X_train, X_test, y_train):
    """
    Automatically select important features.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Selected features
    """
    log_section("FEATURE SELECTION")
    
    with Timer(name="Feature Selection"):
        selector = FeatureSelector(config)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
    
    logger.info(f"Selected features - Train: {X_train_sel.shape}, Test: {X_test_sel.shape}")
    logger.info(f"Features selected: {len(selector.get_selected_features())}")
    
    return X_train_sel, X_test_sel, selector


def train_models(X_train, y_train):
    """
    Train all configured models.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dict[str, model]: Trained models
    """
    log_section("MODEL TRAINING")
    
    models = {}
    tuner = OptunaTuner(config) if config.tuning.enabled else None
    
    model_configs = [
        ('random_forest', RandomForestModel),
        ('xgboost', XGBoostModel),
        ('catboost', CatBoostModel),
        ('lightgbm', LightGBMModel),
        ('tabnet', TabNetModel),
    ]
    
    for model_name, ModelClass in model_configs:
        if not config.models[model_name].enabled:
            logger.info(f"Skipping {model_name} (disabled in config)")
            continue
        
        logger.info(f"\n[{model_name.upper()}] Training started...")
        
        with Timer(name=f"Training {model_name}"):
            # Hyperparameter tuning
            if config.tuning.enabled:
                logger.info(f"Running Optuna tuning for {model_name}...")
                best_params = tuner.tune_model(X_train, y_train, model_name)
                model = ModelClass(config, best_params)
            else:
                model = ModelClass(config)
            
            # Train model
            model.fit(X_train, y_train)
            models[model_name] = model
            
            logger.info(f"✓ {model_name} training completed")
    
    return models


def generate_predictions(models, X_test, test_ids):
    """
    Generate predictions for all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        test_ids: Test set IDs
        
    Returns:
        Dict[str, pd.DataFrame]: Predictions for each model
    """
    log_section("PREDICTION")
    
    predictions_dict = {}
    pbar = get_progress_bar(len(models), desc="Generating predictions")
    
    for model_name, model in models.items():
        logger.info(f"Predicting with {model_name}...")
        
        preds = model.predict(X_test)
        
        pred_df = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': preds
        })
        
        predictions_dict[model_name] = pred_df
        pbar.update(1)
    
    pbar.close()
    
    return predictions_dict


def create_ensemble(models):
    """
    Create ensemble model from trained models.
    
    Args:
        models: Dictionary of trained models
        
    Returns:
        EnsembleModel: Ensemble model
    """
    log_section("ENSEMBLE CREATION")
    
    models_list = list(models.values())
    ensemble = EnsembleModel(models_list, config, strategy=config.models.ensemble.strategy)
    
    logger.info(f"Ensemble created with {len(models_list)} models")
    
    return ensemble


def save_predictions(predictions_dict, ensemble_predictions):
    """
    Save all predictions to CSV files.
    
    Args:
        predictions_dict: Individual model predictions
        ensemble_predictions: Ensemble predictions
    """
    log_section("SAVING PREDICTIONS")
    
    output_dir = Path(config.paths.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual model predictions
    for model_name, predictions in predictions_dict.items():
        output_path = output_dir / f"{model_name}.csv"
        predictions.to_csv(output_path, index=False)
        logger.info(f"Saved {model_name} predictions to {output_path}")
    
    # Save ensemble predictions
    ensemble_path = output_dir / "ensemble_final.csv"
    ensemble_predictions.to_csv(ensemble_path, index=False)
    logger.info(f"Saved ensemble predictions to {ensemble_path}")


def main():
    """
    Main training pipeline.
    """
    global logger, config
    
    # Setup
    setup_logging()
    logger = get_logger(__name__)
    config = get_config()
    
    logger.info("\n" + "="*70)
    logger.info("AMES HOUSING PRICE PREDICTION - FULL PIPELINE")
    logger.info("="*70 + "\n")
    
    try:
        with Timer(name="Complete Pipeline"):
            # 1. Load data
            X_train, X_test, y_train, test_ids = load_data()
            
            # 2. Generate ground truth
            log_section("GROUND TRUTH GENERATION")
            logger.info("Generating ground truth from public Ames dataset...")
            gt_df = generate_ground_truth(
                output_path=config.paths.ground_truth_csv,
                source='sklearn'
            )
            validate_ground_truth(config.paths.ground_truth_csv)
            
            # 3. Preprocess data
            X_train_processed, X_test_processed, pipeline = preprocess_data(X_train, X_test, y_train)
            
            # 4. Engineer features
            X_train_eng, X_test_eng, engineer = engineer_features(X_train_processed, X_test_processed)
            
            # 5. Select features
            X_train_sel, X_test_sel, selector = select_features(X_train_eng, X_test_eng, y_train)
            
            # 6. Train models
            models = train_models(X_train_sel, y_train)
            
            # 7. Generate predictions
            predictions_dict = generate_predictions(models, X_test_sel, test_ids)
            
            # 8. Create ensemble
            ensemble = create_ensemble(models)
            ensemble_preds = ensemble.predict(X_test_sel)
            ensemble_pred_df = pd.DataFrame({
                'Id': test_ids,
                'SalePrice': ensemble_preds
            })
            
            # 9. Save predictions
            save_predictions(predictions_dict, ensemble_pred_df)
            
            # 10. Compare with ground truth
            log_section("GROUND TRUTH COMPARISON")
            comparator = GroundTruthComparison(config)
            results = comparator.compare_models(
                {**predictions_dict, 'ensemble_final': ensemble_pred_df},
                gt_df
            )
            
            # 11. Generate visualizations
            log_section("VISUALIZATION")
            visualizer = ModelVisualizer(config)
            
            for model_name, model in models.items():
                try:
                    importance = model.get_feature_importance()
                    visualizer.plot_feature_importance(importance, model_name)
                except Exception as e:
                    logger.warning(f"Could not plot importance for {model_name}: {e}")
            
            # Compare models
            metrics_for_comparison = {
                name: {'rmse': results[name]['rmse'], 'mae': results[name]['mae']}
                for name in results
            }
            visualizer.plot_model_comparison(metrics_for_comparison, metric='rmse')
            
            logger.info("\n✅ Pipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
