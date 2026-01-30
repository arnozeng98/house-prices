#!/usr/bin/env python
"""
================================================================================
PREDICTION SCRIPT - INFERENCE ON TEST DATA
================================================================================

This script loads trained models and generates predictions on test data.

Usage:
    python src/predict.py [--model_name ensemble_final]

The script looks for trained models in the artifacts directory and
generates predictions saved to data/output/

Author: Arno
Version: 1.0.0
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.logging_config import setup_logging, get_logger
from src.utils.helpers import Timer


def main():
    """
    Load trained models and generate predictions.
    """
    # Setup
    setup_logging()
    logger = get_logger(__name__)
    config = get_config()
    
    logger.info("="*70)
    logger.info("PREDICTION - INFERENCE ON TEST DATA")
    logger.info("="*70 + "\n")
    
    # Check if output files exist
    output_dir = Path(config.paths.output_data)
    
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        logger.info("Please run src/train.py first to generate predictions")
        sys.exit(1)
    
    # List available predictions
    prediction_files = list(output_dir.glob("*.csv"))
    
    if not prediction_files:
        logger.error("No prediction files found in output directory")
        sys.exit(1)
    
    logger.info(f"Found {len(prediction_files)} prediction files:")
    for f in prediction_files:
        df = pd.read_csv(f)
        logger.info(f"  - {f.name}: {df.shape[0]} samples")
    
    # Load and display ensemble predictions (final output)
    ensemble_path = output_dir / "ensemble_final.csv"
    if ensemble_path.exists():
        ensemble_df = pd.read_csv(ensemble_path)
        logger.info(f"\n✅ Ensemble Final Predictions:")
        logger.info(f"   Path: {ensemble_path}")
        logger.info(f"   Shape: {ensemble_df.shape}")
        logger.info(f"   Price range: ${ensemble_df['SalePrice'].min():.2f} - ${ensemble_df['SalePrice'].max():.2f}")
    else:
        logger.warning("Ensemble predictions not found")
    
    logger.info("\n✅ Prediction script completed!")


if __name__ == "__main__":
    main()
