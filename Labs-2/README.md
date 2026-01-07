# Labs-2: Second Attempt (Conformal Prediction Approach)

## Overview

This folder contains the successful second attempt using conformal prediction for regime-based tactical asset allocation. This approach provides statistically valid uncertainty quantification and forms the basis of the conference paper.

**Status:** Active - Publication-ready results

**Key Achievement:** 91.80% coverage with 0.95% standard deviation using Hybrid CP

---

## Lab Structure

### Core Infrastructure

- **data_loader.py** - Unified data loader for all experiments
- **quick_start.py** - Quick start script for running experiments
- **run_all_experiments.py** - Run all lab experiments sequentially
- **validate_everything.py** - Comprehensive validation script

### Lab Experiments

#### 01_regime_discovery
HMM-based regime discovery experiments
- Discovers bull, bear, neutral regimes
- 18.1% bear, 47.9% bull, 34.0% neutral distribution

#### 02_gan_architectures
GAN architecture experiments (exploratory)
- TimeGAN for synthetic data generation
- Not used in final paper

#### 03_classification_models
Classification model comparison
- Gradient Boosting: 98.16% accuracy (best)
- Random Forest, Transformer, LSTM comparisons

#### 04_loss_functions
Loss function experiments
- Focal loss, weighted cross-entropy
- Addresses class imbalance

#### 05_multi_market
Multi-market validation
- Tests across different market conditions
- Validates generalization

#### 06_baselines
Baseline model implementations
- Buy-and-hold strategy
- Point prediction portfolio
- Comparison benchmarks

#### 07_conformal_prediction
Core conformal prediction implementation
- Standard split conformal prediction
- Class-conditional CP
- Adaptive CP
- Hybrid CP (novel contribution)

#### 08_cp_portfolio
Portfolio optimization with conformal prediction
- CP Conservative strategy
- CP Average strategy
- Transaction cost analysis

#### 09_statistical_tests
Statistical validation
- Permutation tests
- Cross-validation
- Significance testing

#### 10_tft_baseline
Temporal Fusion Transformer baseline
- Deep learning comparison
- Not used in final paper

#### 11_enhanced_cp
Enhanced conformal prediction methods
- Addresses class imbalance
- Adaptive methods for non-stationarity

#### 12_multi_asset
Multi-asset conformal prediction (KEY CONTRIBUTION)
- SPY, TLT, GLD validation
- Asset-specific calibration
- Demonstrates generalization across asset classes

---

## Key Results

### Classification Performance
- Gradient Boosting: 98.16% accuracy
- HMM regime discovery: Realistic distributions

### Conformal Prediction Coverage
- Standard CP: 91.09% coverage, 4.61% std dev
- Class-Conditional: 89.67% coverage, 1.57% std dev
- Adaptive CP: 91.09% coverage, 4.29% std dev
- **Hybrid CP: 91.80% coverage, 0.95% std dev** (best)

### Portfolio Performance (SPY, 2020-2024)
- Buy & Hold: 69.88% return, -25.43% max drawdown
- Point Prediction: 61.06% return, -13.59% max drawdown
- **CP Conservative: 40.01% return, -10.21% max drawdown** (59.9% drawdown reduction)

---

## Running Experiments

### Quick Start
```bash
cd Labs-2
python quick_start.py
```

### Run All Experiments
```bash
cd Labs-2
python run_all_experiments.py
```

### Run Specific Lab
```bash
cd Labs-2/12_multi_asset
python run_multi_asset_experiments.py
```

---

## Data Dependencies

All experiments use data from the root `data/` folder:
- `data/raw/` - Raw downloaded data (Yahoo Finance, FRED)
- `data/processed/` - Processed features and labels

Data is automatically downloaded on first run via `data_loader.py`.

---

## Conference Paper

Results from these experiments form the basis of:
- **latex_conference_paper.txt** - IEEE conference paper
- 8 review rounds completed
- 62+ critical fixes applied
- Ready for submission

Key paper sections derived from Labs-2:
- Section III: Methodology (Labs 01, 03, 07)
- Section IV: Experiments (Labs 08, 11, 12)
- Section V: Results (All labs)
- Section VI: Multi-Asset Validation (Lab 12)

---

## Reproducibility

All experiments are fully reproducible:
- Fixed random seeds
- Exact package versions in requirements.txt
- Automated data download
- Documented hyperparameters

Runtime: Approximately 5-10 minutes for full pipeline on CPU.

---

## Comparison with Labs-1

| Aspect | Labs-1 (LSTM/GAN) | Labs-2 (Conformal Prediction) |
|--------|-------------------|-------------------------------|
| Accuracy | Low (failed) | 98.16% |
| Uncertainty | None | Statistical guarantees |
| Approach | Deep learning | HMM + Gradient Boosting + CP |
| Results | Unreliable | Publication-ready |
| Status | Archived | Active |

---

## Next Steps

1. Compile conference paper in Overleaf
2. Add figures from Labs-2 experiments
3. Submit to conference
4. Extend to more asset classes
5. Explore online conformal prediction

---

## Contact

For questions about these experiments, see:
- Main README.md in root folder
- Conference paper documentation
- Individual lab README files

