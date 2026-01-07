@echo off
echo ==========================================
echo IMPROVED FINANCIAL MARKET REGIME PREDICTION
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

REM Check dependencies
echo [1/4] Checking dependencies...
python -c "import torch; import pandas; import numpy; import sklearn; import yfinance; import fredapi" >nul 2>&1
if errorlevel 1 (
    echo Missing dependencies. Installing...
    pip install -r requirements.txt
) else (
    echo √ All dependencies installed
)

echo.
echo [2/4] Running improved analysis...
echo This will take approximately 35-50 minutes on GPU, 60-90 minutes on CPU
echo.

python improved_analysis.py

if errorlevel 1 (
    echo Error: Improved analysis failed
    exit /b 1
)

echo.
echo [3/4] Comparing original vs improved results...
echo.

python compare_results.py

if errorlevel 1 (
    echo Warning: Comparison failed (original results may not exist)
)

echo.
echo [4/4] Summary
echo ==========================================
echo.
echo √ Improved analysis complete
echo √ Results saved to:
echo   - models_improved/
echo   - results/figures_improved/
echo   - results/metrics_improved/
echo.
echo Key files:
echo   - results/metrics_improved/summary_report.txt
echo   - results/figures_improved/trading_strategy.png
echo   - results/comparison_summary.png
echo.
echo ==========================================
echo DONE!
echo ==========================================
pause

