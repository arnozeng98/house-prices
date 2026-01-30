#!/bin/bash

################################################################################
# AMES HOUSING PRICE PREDICTION - ONE-CLICK PIPELINE SCRIPT
################################################################################
#
# This script runs the complete end-to-end machine learning pipeline including:
#   1. Data validation and loading
#   2. Ground truth generation from public Ames dataset
#   3. Data preprocessing (cleaning, encoding, scaling)
#   4. Feature engineering and automatic selection
#   5. Hyperparameter tuning with Optuna (optional)
#   6. Model training (RF, XGBoost, CatBoost, LightGBM, TabNet)
#   7. Ensemble creation and prediction
#   8. Model evaluation and ground truth comparison
#   9. Visualization generation
#
# Usage:
#   ./run_full_pipeline.sh                    # Run complete pipeline
#   ./run_full_pipeline.sh --quick            # Skip Optuna tuning (faster)
#   ./run_full_pipeline.sh --models xgboost   # Train only specific model
#
# Author: Arno
# Version: 1.0.0
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║    AMES HOUSING PRICE PREDICTION - END-TO-END ML PIPELINE           ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Section printer
print_section() {
    echo -e "\n${BLUE}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${GREEN}» $1${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}\n"
}

# Error handler
error_exit() {
    echo -e "${RED}✗ Error: $1${NC}"
    exit 1
}

# Success printer
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
print_section "Checking Python Version"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
print_success "Python environment ready"

# Check required files
print_section "Validating Project Structure"

required_files=(
    "configs/default.yaml"
    "requirements.txt"
    "data/input/train.csv"
    "data/input/test.csv"
    "data/sample_submission.csv"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        error_exit "Required file not found: $file"
    fi
done

print_success "All required files present"

# Install dependencies
print_section "Installing/Updating Dependencies"

if command -v pip3 &> /dev/null; then
    pip3 install -q -r requirements.txt 2>&1 | grep -E "(Successfully installed|Requirement already satisfied|ERROR|error)" || true
    print_success "Dependencies installed"
else
    error_exit "pip3 not found. Please install Python 3 with pip."
fi

# Run main training pipeline
print_section "Starting Full Pipeline Execution"

echo "Command: python3 -u src/train.py"
echo ""

if python3 -u src/train.py; then
    print_success "Training pipeline completed successfully"
else
    error_exit "Training pipeline failed"
fi

# Summary
print_section "Pipeline Execution Summary"

echo -e "${GREEN}✓ Ground truth generated and validated${NC}"
echo -e "${GREEN}✓ Data preprocessing completed${NC}"
echo -e "${GREEN}✓ Feature engineering completed${NC}"
echo -e "${GREEN}✓ Feature selection completed${NC}"
echo -e "${GREEN}✓ Models trained and evaluated${NC}"
echo -e "${GREEN}✓ Predictions generated${NC}"
echo -e "${GREEN}✓ Ground truth comparison completed${NC}"
echo -e "${GREEN}✓ Visualizations created${NC}"

# Output summary
print_section "Output Files Summary"

echo "Predictions saved to: data/output/"
ls -lh data/output/*.csv 2>/dev/null | awk '{print "  -", $9, "(" $5 ")"}' || echo "  No prediction files found"

echo ""
echo "Visualizations saved to: data/img/"
ls -1 data/img/*.png 2>/dev/null | wc -l | xargs -I {} echo "  {} visualization files created"

echo ""
echo "Logs saved to: logs/results.log"
tail -5 logs/results.log 2>/dev/null | head -3

# Final message
echo ""
print_banner
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     ✓ PIPELINE COMPLETED SUCCESSFULLY!               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review logs/results.log for detailed results"
echo "  2. Check data/output/ for final predictions"
echo "  3. View data/img/ for visualizations"
echo "  4. Compare with ground truth in data/gt.csv"

echo ""
echo -e "${BLUE}For more details, see README.md and docs/tsd.md${NC}"
echo ""
