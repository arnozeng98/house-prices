"""
================================================================================
CONFIGURATION MANAGEMENT MODULE
================================================================================

This module handles loading, parsing, and accessing configuration from
the default.yaml file. It provides a centralized config object that can
be accessed throughout the entire project.

Features:
    - Automatic YAML parsing and type conversion
    - Safe dictionary-like access with dot notation
    - Path validation and creation
    - Device (CUDA/CPU) detection and fallback
    - Seed management for reproducibility

Example:
    from src.config import Config
    config = Config()
    device = config.device.type
    train_path = config.paths.train_csv
================================================================================
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Enhanced dictionary class that allows attribute-style access (dot notation).
    
    This enables both config.device.type and config['device']['type'] access patterns.
    Provides convenient access to nested configuration values.
    """
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to dictionary items.
        
        Args:
            name: The key to access
            
        Returns:
            The value at the key, converted to ConfigDict if it's a dict
            
        Raises:
            AttributeError: If the key doesn't exist
        """
        try:
            value = self[name]
            # Recursively convert nested dicts to ConfigDict for seamless access
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"Configuration key '{name}' not found")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Allow attribute-style assignment to dictionary items.
        
        Args:
            name: The key to set
            value: The value to set
        """
        self[name] = value
    
    def __delattr__(self, name: str) -> None:
        """
        Allow attribute-style deletion of dictionary items.
        
        Args:
            name: The key to delete
        """
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Configuration key '{name}' not found")


class Config:
    """
    Main configuration class for the Ames Housing Price Prediction project.
    
    Responsibilities:
        1. Load and parse configuration from YAML file
        2. Detect and configure device (CUDA or CPU)
        3. Create necessary directories
        4. Set random seeds for reproducibility
        5. Provide centralized access to all configuration parameters
    
    Attributes:
        config (ConfigDict): The parsed configuration dictionary
        config_path (str): Path to the configuration YAML file
        device (torch.device): The device for tensor operations
        device_available (str): String representation of available device
    
    Example:
        >>> config = Config()
        >>> config.device.type
        'cuda'
        >>> config.models.xgboost.params.n_estimators
        500
    """
    
    # Class-level config instance (singleton pattern)
    _instance = None
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize the configuration loader.
        
        This method:
        1. Loads configuration from YAML file
        2. Sets up device (CUDA/CPU with automatic fallback)
        3. Creates required directories
        4. Sets random seeds for reproducibility
        
        Args:
            config_path (str): Path to the configuration YAML file.
                Defaults to "configs/default.yaml"
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        self.config_path = config_path
        
        # Load YAML configuration
        self._load_config()
        
        # Setup device (CUDA or CPU with fallback)
        self._setup_device()
        
        # Create necessary directories
        self._create_directories()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        logger.info("Configuration loaded successfully")
    
    def _load_config(self) -> None:
        """
        Load configuration from YAML file and convert to ConfigDict.
        
        The YAML file should be located at the path specified in __init__.
        Nested dictionaries are converted to ConfigDict for convenient access.
        
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file has syntax errors
        """
        # Check if config file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at: {self.config_path}\n"
                f"Please ensure configs/default.yaml exists."
            )
        
        # Load YAML file
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to ConfigDict for attribute access
            self.config = ConfigDict(config_dict)
            logger.info(f"Configuration loaded from: {self.config_path}")
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
    
    def _setup_device(self) -> None:
        """
        Detect and configure the device (CUDA or CPU).
        
        This method:
        1. Checks if CUDA is available (unless force_cpu is True)
        2. Sets up PyTorch device accordingly
        3. Logs device information for debugging
        
        Device priority:
            1. If force_cpu is True: use CPU
            2. If type is 'cuda' or 'auto' and CUDA available: use CUDA
            3. Otherwise: use CPU (fallback)
        """
        device_config = self.config.device
        
        # Check if force CPU mode
        if device_config.force_cpu:
            self.device = torch.device("cpu")
            self.device_available = "CPU (forced)"
            logger.info("Device set to CPU (forced by user)")
            return
        
        # Auto-detect or explicit GPU request
        device_type = device_config.type
        
        if device_type in ["cuda", "auto"]:
            if torch.cuda.is_available():
                gpu_index = device_config.get("gpu_index", 0)
                self.device = torch.device(f"cuda:{gpu_index}")
                self.device_available = f"CUDA GPU {gpu_index}"
                logger.info(
                    f"CUDA available - Using GPU {gpu_index}\n"
                    f"CUDA Version: {torch.version.cuda}\n"
                    f"GPU Name: {torch.cuda.get_device_name(gpu_index)}"
                )
            else:
                self.device = torch.device("cpu")
                self.device_available = "CPU (CUDA not available - fallback)"
                logger.warning("CUDA requested but not available - falling back to CPU")
        else:
            # Explicit CPU request
            self.device = torch.device("cpu")
            self.device_available = "CPU"
            logger.info("Device set to CPU")
    
    def _create_directories(self) -> None:
        """
        Create necessary directories if they don't exist.
        
        This method ensures all required directories for data, logs, and
        artifacts are created before the pipeline runs.
        
        Directories created:
            - logs_root: For log files
            - output_data: For prediction outputs
            - images_data: For visualization outputs
            - models_dir: For saved model artifacts
            - scalers_dir: For saved preprocessing scalers
            - feature_config_dir: For feature configuration metadata
        """
        # List of directories to create
        dirs_to_create = [
            self.config.paths.logs_root,
            self.config.paths.output_data,
            self.config.paths.images_data,
            self.config.paths.models_dir,
            self.config.paths.scalers_dir,
            self.config.paths.feature_config_dir,
        ]
        
        # Create each directory if it doesn't exist
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Required directories created/verified")
    
    def _set_seeds(self) -> None:
        """
        Set random seeds for reproducibility.
        
        This method sets seeds for:
            1. Python's random module
            2. NumPy's random number generator
            3. PyTorch's CPU random number generator
            4. PyTorch's CUDA random number generator
        
        These should be called before any model training to ensure
        reproducible results across different runs.
        """
        import random
        
        # Extract seed values from config
        seed = self.config.device.seed
        numpy_seed = self.config.device.numpy_seed
        torch_seed = self.config.device.torch_seed
        
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy random seed
        np.random.seed(numpy_seed)
        
        # Set PyTorch CPU random seed
        torch.manual_seed(torch_seed)
        
        # Set PyTorch CUDA random seeds if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)
            torch.cuda.manual_seed_all(torch_seed)
            # Ensure deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(
            f"Random seeds set - Python: {seed}, NumPy: {numpy_seed}, "
            f"PyTorch: {torch_seed}"
        )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot-notation path.
        
        This allows accessing nested values using a dot-separated string
        instead of nested bracket access.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
                (e.g., "device.type", "models.xgboost.params.n_estimators")
            default (Any): Default value if key is not found
        
        Returns:
            The value at the specified path, or default if not found
        
        Example:
            >>> config.get("device.type", "cpu")
            'cuda'
            >>> config.get("models.unknown.param", "default_value")
            'default_value'
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, (dict, ConfigDict)):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow direct attribute access to top-level config sections.
        
        Args:
            name: The configuration section name (e.g., "device", "models")
        
        Returns:
            The configuration section
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self.config, name)
    
    @classmethod
    def get_instance(cls, config_path: str = "configs/default.yaml") -> 'Config':
        """
        Get or create a singleton instance of Config.
        
        This method implements the singleton pattern to ensure only one
        Config instance exists throughout the application lifetime.
        
        Args:
            config_path (str): Path to configuration YAML file
        
        Returns:
            Config: The singleton Config instance
        """
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
    def __repr__(self) -> str:
        """
        String representation of the Config object.
        
        Returns:
            A string showing the configuration file path and device info
        """
        return (
            f"Config(path='{self.config_path}', "
            f"device='{self.device_available}')"
        )


# Create a global config instance for easy import and use
def get_config(config_path: str = "configs/default.yaml") -> Config:
    """
    Convenience function to get the Config instance.
    
    Args:
        config_path (str): Path to configuration YAML file
    
    Returns:
        Config: The Config instance
    """
    return Config.get_instance(config_path)
