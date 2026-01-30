# Ames Housing Price Prediction - Advanced Regression Techniques

## 📋 Overview

This project implements a state-of-the-art machine learning pipeline for the **Ames Housing Price Prediction** competition on Kaggle. The solution combines modern deep learning techniques (TabNet), gradient boosting methods (XGBoost, CatBoost, LightGBM), ensemble learning, and advanced hyperparameter optimization to achieve competitive results.

**[中文版本](docs/README_zh.md)** | **[Technical Deep Dive](docs/tsd.md)**

## 🎯 Project Goals

1. **Generate Ground Truth**: Extract complete Ames dataset from scikit-learn to establish true labels for model evaluation
2. **Build Production-Grade Pipeline**: Implement modular, reusable ML pipeline following software engineering best practices
3. **Achieve Top-Tier Performance**: Utilize modern techniques including:
   - Automatic feature engineering and selection (Boruta, Permutation Importance, Mutual Information)
   - GPU-accelerated deep learning (TabNet with CUDA support)
   - Competition-level hyperparameter tuning (Optuna with 500+ trials)
   - Smart ensemble learning (weighted averaging + stacking)
4. **Comprehensive Documentation**: Detailed technical specifications and architecture documentation

## 🚀 Quick Start

### Installation

```bash
# Clone/download the project
cd house-prices

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
./run_full_pipeline.sh
```

### Requirements

- **Python**: 3.8+
- **RAM**: Minimum 8GB (16GB+ recommended for Optuna tuning)
- **CUDA** (Optional): For GPU acceleration with TabNet
- **Disk Space**: ~2GB for datasets and artifacts

### File Structure

```
house-prices/
├── configs/
│   └── default.yaml              # Central configuration (all parameters)
├── src/
│   ├── config.py                 # Configuration management
│   ├── logging_config.py          # Logging setup
│   ├── io/                        # Data input/output
│   │   └── ground_truth.py        # Ground truth generation
│   ├── preprocessing/             # Data cleaning & transformation
│   │   ├── cleaner.py
│   │   ├── encoder.py
│   │   ├── scaler.py
│   │   └── pipeline.py
│   ├── features/                  # Feature engineering & selection
│   │   ├── engineer.py
│   │   └── selector.py
│   ├── models/                    # ML models
│   │   ├── base_model.py          # Abstract base class
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── catboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── tabnet_model.py        # Deep learning (GPU)
│   │   └── ensemble.py
│   ├── tuning/                    # Hyperparameter optimization
│   │   └── optuna_tuner.py        # Optuna-based tuning
│   ├── evaluation/                # Model evaluation
│   │   └── comparison.py          # Ground truth comparison
│   ├── visualization/             # Plotting & analysis
│   │   └── plots.py
│   ├── utils/                     # Utility functions
│   │   └── helpers.py
│   ├── train.py                   # Main training script
│   ├── predict.py                 # Inference script
│   └── evaluate.py                # Evaluation metrics
├── data/
│   ├── input/                     # Raw data
│   │   ├── train.csv
│   │   └── test.csv
│   ├── output/                    # Model predictions
│   ├── img/                       # Visualizations
│   └── gt.csv                     # Ground truth (generated)
├── docs/
│   ├── tsd.md                     # Technical specification
│   └── README_zh.md               # Chinese documentation
├── logs/
│   └── results.log                # Execution logs & metrics
├── requirements.txt               # Python dependencies
└── run_full_pipeline.sh           # One-click execution script
```

## 📊 Pipeline Stages

### 1️⃣ Ground Truth Generation
- Loads complete Ames dataset from scikit-learn
- Matches test IDs with true sale prices
- Validates and saves to `data/gt.csv`
- Enables objective model comparison

### 2️⃣ Data Preprocessing
- **Cleaning**: Intelligent missing value handling (median for numerical, mode for categorical)
- **Ames-specific**: Handles domain-specific NAs (e.g., "no garage" not missing)
- **Outlier Detection**: IQR and Z-score methods
- **Encoding**: Target encoding, one-hot, label encoding (configurable)
- **Scaling**: Robust, standard, or MinMax scaling

### 3️⃣ Automatic Feature Engineering
- **Polynomial Features**: Squares and interaction terms
- **Domain Features**: Total area, quality scores, house age
- **Categorical Interactions**: Multi-way combinations
- **Limits**: Configurable max interactions to prevent explosion

### 4️⃣ Intelligent Feature Selection
Uses **voting ensemble** of three selection methods:
- **Boruta**: Wrapper-based feature selection with RF
- **Permutation Importance**: Model-agnostic importance
- **Mutual Information**: Information-theoretic ranking
- **Result**: Only most predictive features retained

### 5️⃣ Model Training & Tuning
Trains 5 complementary models with automatic hyperparameter optimization:

| Model | Type | GPU Support | Best Use |
|-------|------|-------------|----------|
| **Random Forest** | Ensemble | No | Baseline, interpretability |
| **XGBoost** | Gradient Boosting | Yes | Competitive baseline |
| **CatBoost** | Gradient Boosting | Yes | Categorical features |
| **LightGBM** | Gradient Boosting | Yes | Fast, memory-efficient |
| **TabNet** | Deep Learning | Yes (CUDA) | Modern SOTA method |

**Hyperparameter Tuning**:
- Framework: Optuna with TPE sampler
- Trials: 500 per model (configurable)
- Objective: Minimize CV RMSE
- Cross-validation: 5-fold
- Timeout: 5 min per trial

### 6️⃣ Ensemble & Prediction
- **Strategy**: Weighted averaging (weights from config)
- **Output**: Final predictions saved to `data/output/`
- **Format**: [Id, SalePrice] matching Kaggle submission

### 7️⃣ Ground Truth Comparison
- Compares all predictions against true values
- Computes RMSE, MAE, R², MAPE
- Ranks models by performance
- Detailed error analysis per model

### 8️⃣ Visualization & Reporting
Generates:
- Feature importance charts (all models)
- Actual vs predicted scatter plots
- Prediction error distributions
- Model performance comparison
- Correlation heatmaps
- Optuna optimization history

## 🔧 Configuration

All parameters are centralized in `configs/default.yaml`:

```yaml
# Device Configuration
device:
  type: "auto"           # 'cuda', 'cpu', or 'auto'
  force_cpu: false
  seed: 42

# Preprocessing
preprocessing:
  missing_value_strategy:
    numerical: "median"
    categorical: "mode"
  numerical_scaler: "robust"
  categorical_encoder: "target"

# Feature Engineering
feature_engineering:
  enabled: true
  numerical_interactions:
    enabled: true
    polynomial_degree: 2
    max_interactions: 50

# Feature Selection
feature_selection:
  enabled: true
  boruta:
    enabled: true
    max_iterations: 100
  min_features: 20
  max_features: 200

# Models
models:
  random_forest:
    enabled: true
    params:
      n_estimators: 200
      max_depth: 15
  xgboost:
    enabled: true
    params:
      n_estimators: 500
      max_depth: 6
  # ... (more models)

# Optuna Tuning
tuning:
  enabled: true
  n_trials: 500
  max_parallel_trials: 2
  sampler: "TPE"
  cv_folds: 5
```

To modify behavior, edit `configs/default.yaml` - **no code changes needed**.

## 📈 Performance

### Model Comparison (Ground Truth RMSE)

Models are evaluated on the complete test set against true values from scikit-learn's Ames dataset.

**Sample Results** (will vary based on feature engineering and tuning):
- XGBoost: ~0.12 RMSE
- CatBoost: ~0.11 RMSE  
- LightGBM: ~0.13 RMSE
- TabNet: ~0.14 RMSE
- Ensemble: ~0.10 RMSE (best)

Results are logged in `logs/results.log`

## 🛠️ Advanced Features

### GPU Acceleration
- **XGBoost, CatBoost, LightGBM**: Automatic GPU detection
- **TabNet**: CUDA optimization with CPU fallback
- **Automatic Fallback**: If GPU training fails, automatically retries on CPU

```python
# In configs/default.yaml
device:
  type: "auto"      # Automatically uses CUDA if available
  force_cpu: false  # Set to true to force CPU mode
```

### Custom Model Training

```python
from src.models.xgboost_model import XGBoostModel
from src.config import get_config

config = get_config()

# Create and train model
model = XGBoostModel(config)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Make predictions
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()

# Cross-validation
cv_results = model.cross_validate(X_train, y_train, cv_folds=5)
```

### Custom Preprocessing Pipeline

```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(config)
X_train_processed, X_test_processed = pipeline.fit_transform(
    X_train, X_test, y_train
)
```

## 📊 Output Files

After running the pipeline:

```
data/
├── output/
│   ├── random_forest.csv      # RF predictions
│   ├── xgboost.csv            # XGBoost predictions
│   ├── catboost.csv           # CatBoost predictions
│   ├── lightgbm.csv           # LightGBM predictions
│   ├── tabnet.csv             # TabNet predictions
│   └── ensemble_final.csv     # Final ensemble predictions
├── img/
│   ├── feature_importance_*.png
│   ├── predictions_*.png
│   ├── error_distribution_*.png
│   ├── model_comparison_*.png
│   ├── correlation_heatmap.png
│   └── optuna_history_*.png
└── gt.csv                      # Ground truth data

logs/
└── results.log                 # Training logs & metrics
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce `tuning.max_parallel_trials` in config
- Reduce `tabnet.batch_size`
- Run on CPU (`device.force_cpu: true`)

### CUDA Errors
- Automatic fallback to CPU is enabled
- Check NVIDIA driver: `nvidia-smi`
- Verify PyTorch CUDA version matches GPU driver

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Slow Training
- Disable Optuna tuning: `tuning.enabled: false`
- Reduce `n_trials` for quick testing
- Enable GPU in config

## 📚 Documentation

- **[README (中文)](docs/README_zh.md)**: Chinese version with detailed explanations
- **[Technical Specification Document](docs/tsd.md)**: In-depth architecture, algorithms, and implementation details
- **Inline Code Comments**: Extensive English comments in all modules

## 🔬 Research & References

Models and techniques used:

1. **TabNet**: [TabNet Paper](https://arxiv.org/abs/1908.07442) - Attentive interpretable tabular learning
2. **XGBoost**: Gradient boosting with second-order optimization
3. **CatBoost**: Boosting with categorical features handling
4. **Boruta**: Feature selection algorithm
5. **Optuna**: Bayesian hyperparameter optimization framework

## 📝 Version History

- **v1.0.0** (Jan 2026): Initial release with full pipeline, multi-model support, Optuna tuning, CUDA support

## 👨‍💻 Author

- **Arno** - Original implementation

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for the housing price prediction competition
- scikit-learn for the complete Ames dataset
- Contributors to XGBoost, CatBoost, LightGBM, TabNet projects

---

**Last Updated**: January 2026

**Questions?** Check `docs/tsd.md` for technical details or review the inline code comments.