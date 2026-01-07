#!/bin/bash

echo "=========================================="
echo "IMPROVED FINANCIAL MARKET REGIME PREDICTION"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if required packages are installed
echo "[1/4] Checking dependencies..."
python -c "import torch; import pandas; import numpy; import sklearn; import yfinance; import fredapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip install -r requirements.txt
else
    echo "✓ All dependencies installed"
fi

echo ""
echo "[2/4] Running improved analysis..."
echo "This will take approximately 35-50 minutes on GPU, 60-90 minutes on CPU"
echo ""

python improved_analysis.py

if [ $? -ne 0 ]; then
    echo "Error: Improved analysis failed"
    exit 1
fi

echo ""
echo "[3/4] Comparing original vs improved results..."
echo ""

python compare_results.py

if [ $? -ne 0 ]; then
    echo "Warning: Comparison failed (original results may not exist)"
fi

echo ""
echo "[4/4] Summary"
echo "=========================================="
echo ""
echo "✓ Improved analysis complete"
echo "✓ Results saved to:"
echo "  - models_improved/"
echo "  - results/figures_improved/"
echo "  - results/metrics_improved/"
echo ""
echo "Key files:"
echo "  - results/metrics_improved/summary_report.txt"
echo "  - results/figures_improved/trading_strategy.png"
echo "  - results/comparison_summary.png"
echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="

