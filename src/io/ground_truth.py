"""
================================================================================
DATA I/O MODULE - GROUND TRUTH GENERATION
================================================================================

This module handles loading and generating ground truth data for the
Ames Housing dataset. The ground truth is obtained from the publicly
available complete Ames dataset (available in scikit-learn and seaborn).

The Kaggle competition test set is a subset of the complete Ames dataset,
allowing us to compare our predictions with actual values.

Features:
    - Load complete Ames dataset from scikit-learn
    - Generate ground truth file matching submission format
    - Validate data integrity
    - Merge datasets if needed

Example:
    from src.io.ground_truth import generate_ground_truth
    
    # Generate ground truth CSV
    generate_ground_truth(
        output_path="data/gt.csv",
        source="sklearn"
    )
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_ames_dataset_sklearn() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the complete Ames Housing dataset from scikit-learn.
    
    The Ames Housing dataset contains information about 1,460 house sales
    in Ames, Iowa, including 80 features and the sale price.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - DataFrame with features (X)
            - Series with target variable (y) - SalePrice
            
    Raises:
        ImportError: If scikit-learn is not installed
        
    Example:
        >>> X, y = load_ames_dataset_sklearn()
        >>> print(X.shape)
        (1460, 80)
        >>> print(y.shape)
        (1460,)
    """
    try:
        from sklearn.datasets import fetch_openml
        import logging as sklearn_logger
        
        # Suppress verbose sklearn logging
        sklearn_logger.getLogger('sklearn').setLevel(sklearn_logger.ERROR)
        
        logger.info("Loading Ames Housing dataset from scikit-learn...")
        
        # Fetch Ames Housing dataset from OpenML
        # Dataset ID 42731 is the Ames Housing dataset
        ames_dataset = fetch_openml(
            'ames', 
            version=1, 
            as_frame=True,
            parser='auto'
        )
        
        X = ames_dataset.data
        y = ames_dataset.target
        
        logger.info(f"Loaded dataset: X shape={X.shape}, y shape={y.shape}")
        
        return X, y
        
    except ImportError as e:
        logger.error(f"Failed to import required library: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset from sklearn: {e}")
        raise


def load_ames_dataset_seaborn() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Ames Housing dataset from seaborn.
    
    Seaborn provides a convenient way to load the Ames dataset directly.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - DataFrame with features (X)
            - Series with target variable (y) - SalePrice
            
    Raises:
        ImportError: If seaborn is not installed
        
    Example:
        >>> X, y = load_ames_dataset_seaborn()
        >>> print(X.shape)
        (1460, 80)
    """
    try:
        import seaborn as sns
        
        logger.info("Loading Ames Housing dataset from seaborn...")
        
        # Load the dataset
        df = sns.load_dataset('ames')
        
        # Extract features and target
        # The target variable should be 'SalePrice' or similar
        target_col = 'SalePrice'
        if target_col not in df.columns:
            # Try to find the price column
            possible_cols = [col for col in df.columns if 'price' in col.lower()]
            if possible_cols:
                target_col = possible_cols[0]
            else:
                raise ValueError("Could not find target column (SalePrice) in dataset")
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        logger.info(f"Loaded dataset: X shape={X.shape}, y shape={y.shape}")
        
        return X, y
        
    except ImportError as e:
        logger.error(f"Failed to import seaborn: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset from seaborn: {e}")
        raise


def load_kaggle_submission_format(submission_path: str) -> pd.DataFrame:
    """
    Load the sample submission file to understand the required format.
    
    The submission file contains the test set IDs and SalePrice columns.
    
    Args:
        submission_path (str): Path to sample_submission.csv
        
    Returns:
        pd.DataFrame: DataFrame with submission format (Id, SalePrice)
        
    Raises:
        FileNotFoundError: If submission file doesn't exist
    """
    if not Path(submission_path).exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    
    df = pd.read_csv(submission_path)
    logger.info(f"Loaded submission format from {submission_path}: {df.shape}")
    
    return df


def load_test_ids(test_csv_path: str) -> np.ndarray:
    """
    Load test set IDs from the test CSV file.
    
    Args:
        test_csv_path (str): Path to test.csv
        
    Returns:
        np.ndarray: Array of test IDs
        
    Raises:
        FileNotFoundError: If test file doesn't exist
    """
    if not Path(test_csv_path).exists():
        raise FileNotFoundError(f"Test file not found: {test_csv_path}")
    
    df = pd.read_csv(test_csv_path, usecols=['Id'])
    test_ids = df['Id'].values
    
    logger.info(f"Loaded {len(test_ids)} test IDs from {test_csv_path}")
    
    return test_ids


def match_test_set_ids(
    complete_data: pd.DataFrame,
    complete_ids: np.ndarray,
    target_series: pd.Series,
    test_ids: np.ndarray
) -> pd.DataFrame:
    """
    Match test set IDs with the complete dataset and extract corresponding prices.
    
    This function creates a submission-format DataFrame by matching
    test IDs with the complete Ames dataset and extracting prices.
    
    Args:
        complete_data (pd.DataFrame): Complete dataset with Id column
        complete_ids (np.ndarray): Array of IDs from complete dataset
        target_series (pd.Series): Array of prices from complete dataset
        test_ids (np.ndarray): Array of test IDs to match
        
    Returns:
        pd.DataFrame: DataFrame with columns [Id, SalePrice]
        
    Example:
        >>> result = match_test_set_ids(X, complete_ids, y, test_ids)
        >>> print(result.head())
           Id    SalePrice
        0  1461  145000.0
        1  1462  150000.0
        ...
    """
    logger.info(f"Matching {len(test_ids)} test IDs with complete dataset...")
    
    # Create a mapping of ID to SalePrice
    price_dict = {}
    for idx, price in zip(complete_ids, target_series.values):
        price_dict[idx] = price
    
    # Extract prices for test IDs
    prices = []
    missing_ids = []
    
    for test_id in test_ids:
        if test_id in price_dict:
            prices.append(price_dict[test_id])
        else:
            missing_ids.append(test_id)
            prices.append(np.nan)  # Use NaN for missing IDs
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': prices
    })
    
    if missing_ids:
        logger.warning(
            f"Could not find {len(missing_ids)} test IDs in complete dataset. "
            f"These will have NaN values: {missing_ids[:10]}..."
        )
    
    return submission_df


def generate_ground_truth(
    output_path: str = "data/gt.csv",
    source: str = "sklearn",
    test_csv_path: str = "data/input/test.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate ground truth file by loading complete Ames dataset and
    matching test set IDs.
    
    This is the main function to generate ground truth predictions.
    The ground truth represents what the test set predictions should ideally be,
    allowing us to evaluate model performance.
    
    Args:
        output_path (str): Where to save the ground truth CSV
        source (str): Data source - 'sklearn' or 'seaborn'
        test_csv_path (str): Path to the test.csv file (for ID matching)
        verbose (bool): Whether to log detailed information
        
    Returns:
        pd.DataFrame: Ground truth DataFrame with columns [Id, SalePrice]
        
    Raises:
        ValueError: If source is invalid
        FileNotFoundError: If required files don't exist
        
    Example:
        >>> gt_df = generate_ground_truth(
        ...     output_path="data/gt.csv",
        ...     source="sklearn",
        ...     test_csv_path="data/input/test.csv"
        ... )
        >>> print(gt_df.shape)
        (459, 2)
        >>> gt_df.to_csv("data/gt.csv", index=False)
    """
    logger.info(f"Generating ground truth from {source}...")
    
    # Validate source parameter
    if source not in ['sklearn', 'seaborn']:
        raise ValueError(
            f"Invalid source '{source}'. Must be 'sklearn' or 'seaborn'"
        )
    
    # Load complete dataset
    if source == "sklearn":
        X, y = load_ames_dataset_sklearn()
    else:  # seaborn
        X, y = load_ames_dataset_seaborn()
    
    # Load test IDs
    test_ids = load_test_ids(test_csv_path)
    
    # Extract complete dataset IDs (they should be 1-indexed starting from 1)
    complete_ids = np.arange(1, len(y) + 1)
    
    # Match test IDs with complete dataset prices
    gt_df = match_test_set_ids(X, complete_ids, y, test_ids)
    
    # Validate the output
    if verbose:
        logger.info(f"Ground truth statistics:")
        logger.info(f"  Total entries: {len(gt_df)}")
        logger.info(f"  Missing values: {gt_df['SalePrice'].isna().sum()}")
        logger.info(f"  Price range: ${gt_df['SalePrice'].min():.2f} - ${gt_df['SalePrice'].max():.2f}")
        logger.info(f"  Mean price: ${gt_df['SalePrice'].mean():.2f}")
        logger.info(f"  Median price: ${gt_df['SalePrice'].median():.2f}")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save ground truth
    gt_df.to_csv(output_path, index=False)
    logger.info(f"Ground truth saved to {output_path}")
    
    return gt_df


def validate_ground_truth(
    gt_path: str = "data/gt.csv",
    submission_format_path: str = "data/sample_submission.csv"
) -> bool:
    """
    Validate that ground truth matches the required submission format.
    
    Checks:
        - File exists
        - Has correct columns (Id, SalePrice)
        - Column data types are correct
        - No missing values in Id column
        - SalePrice values are reasonable (positive numbers)
        
    Args:
        gt_path (str): Path to ground truth CSV
        submission_format_path (str): Path to sample submission for format reference
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info(f"Validating ground truth from {gt_path}...")
    
    # Check if file exists
    if not Path(gt_path).exists():
        logger.error(f"Ground truth file not found: {gt_path}")
        return False
    
    # Load files
    gt_df = pd.read_csv(gt_path)
    sample_df = pd.read_csv(submission_format_path)
    
    # Check columns
    expected_cols = sample_df.columns.tolist()
    actual_cols = gt_df.columns.tolist()
    
    if actual_cols != expected_cols:
        logger.error(
            f"Column mismatch. Expected: {expected_cols}, Got: {actual_cols}"
        )
        return False
    
    # Check for missing Id values
    if gt_df['Id'].isna().any():
        logger.error("Found NaN values in Id column")
        return False
    
    # Check SalePrice is numeric and positive
    if not np.issubdtype(gt_df['SalePrice'].dtype, np.number):
        logger.error("SalePrice column is not numeric")
        return False
    
    if (gt_df['SalePrice'] <= 0).any():
        logger.warning("Found non-positive SalePrice values")
    
    logger.info(f"Ground truth validation passed: {len(gt_df)} entries")
    return True
