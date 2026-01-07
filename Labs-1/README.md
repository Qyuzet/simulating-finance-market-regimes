# Labs-1: First Attempt (LSTM/GAN Approach)

## Overview

This folder contains the first attempt at financial market regime prediction using LSTM and GAN-based approaches. These experiments were conducted before the successful conformal prediction approach (see Labs-2).

**Status:** Archived - Low accuracy results led to pivot to conformal prediction approach

**Key Issue:** Poor classification accuracy and unreliable predictions

---

## Contents

### Python Scripts

- **improved_analysis.py** - WGAN-GP + Balanced LSTM implementation
- **run_analysis.py** - Complete CGAN + LSTM hybrid model pipeline
- **check_abstract.py** - Abstract validation script
- **check_cv.py** - Cross-validation checking
- **check_split.py** - Data split verification

### Notebooks

- **main_analysis.ipynb** - Jupyter notebook for exploratory analysis

### Models

- **models/** - First generation models (CGAN, LSTM, Random Forest, Logistic Regression, ARIMA)
- **models_improved/** - Second generation models (WGAN-GP, improved LSTM)

### Results

- **results/** - Experimental results, figures, and metrics from first attempt
  - figures/ - Visualization outputs
  - figures_improved/ - Improved model visualizations
  - metrics/ - Performance metrics
  - metrics_improved/ - Improved model metrics

### Data Files

- **final_performance.csv** - Final performance metrics
- **final_results.csv** - Final experimental results

### Scripts

- **run_improved.bat** - Windows batch script to run improved analysis
- **run_improved.sh** - Unix shell script to run improved analysis

---

## Why This Approach Failed

1. **Low Accuracy:** LSTM models struggled with regime classification
2. **Mode Collapse:** GAN-based approaches had stability issues
3. **Class Imbalance:** Difficulty handling imbalanced regime distributions
4. **Overfitting:** Models didn't generalize well to test data
5. **No Uncertainty Quantification:** Point predictions without confidence measures

---

## Lessons Learned

These experiments led to important insights:

1. **Need for Uncertainty Quantification:** Point predictions insufficient for financial decisions
2. **Regime Discovery First:** HMM-based regime discovery more reliable than supervised labels
3. **Simpler Models:** Gradient Boosting outperformed complex deep learning
4. **Conformal Prediction:** Provides statistical guarantees missing from LSTM/GAN approach

---

## Transition to Labs-2

Based on these failures, the project pivoted to:

- **Conformal Prediction** for uncertainty quantification
- **HMM + Gradient Boosting** for more reliable regime classification
- **Statistical Validation** with provable coverage guarantees
- **Multi-Asset Testing** for generalization

See **Labs-2/** for the successful second attempt using conformal prediction.

---

## Historical Context

This folder is preserved for:

1. **Documentation** of the research process
2. **Comparison** with successful approach
3. **Learning** from failed experiments
4. **Reproducibility** of the complete research journey

---

## Note

These files are archived and not actively maintained. For current work, see:

- **Labs-2/** - Successful conformal prediction approach
- **final_analysis.py** (root) - Best performing HMM + Gradient Boosting method
- **latex_conference_paper.txt** - Conference paper based on Labs-2 results

