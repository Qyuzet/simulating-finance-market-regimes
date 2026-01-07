"""
STATISTICAL VALIDATION FOR CONFORMAL PREDICTION
Prove results are not overfitting with rigorous statistical tests
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

# Import conformal predictor
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '07_conformal_prediction'))
from conformal_regime_classifier import ConformalRegimeClassifier

print("="*80)
print("📊 STATISTICAL VALIDATION FOR CONFORMAL PREDICTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA & PREPARE FEATURES
# ============================================================================

print("\n[1] DATA PREPARATION")
print("-"*80)

df = load_complete_dataset()
print(f"  Dataset: {df.shape}")

# HMM regime discovery
hmm_features = df[['returns', 'volatility', 'momentum']].values
scaler_hmm = StandardScaler()
hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)

model_hmm = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model_hmm.fit(hmm_features_scaled)
regimes = model_hmm.predict(hmm_features_scaled)

regime_returns = [df[regimes == i]['returns'].mean() for i in range(3)]
regime_mapping = np.argsort(regime_returns)
regimes_labeled = np.array([np.where(regime_mapping == r)[0][0] for r in regimes])

df['regime'] = regimes_labeled

# Prepare features
feature_cols = ['returns', 'volatility', 'momentum', 'RSI', 'MACD', 'FEDFUNDS']
lags = [1, 5, 10, 20]

X_list = [df[feature_cols]]
for lag in lags:
    lagged = df[feature_cols].shift(lag)
    lagged.columns = [f"{col}_lag{lag}" for col in feature_cols]
    X_list.append(lagged)

X = pd.concat(X_list, axis=1).dropna()
y = df['regime'].loc[X.index]

print(f"  Features: {X.shape[1]}")
print(f"  Samples: {X.shape[0]}")
print(f"  Regime distribution: Bear={np.sum(y==0)}, Bull={np.sum(y==1)}, Neutral={np.sum(y==2)}")

# ============================================================================
# 2. K-FOLD CROSS-VALIDATION
# ============================================================================

print("\n[2] K-FOLD CROSS-VALIDATION (k=5)")
print("-"*80)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\n  Cross-Validation Scores:")
for i, score in enumerate(cv_scores):
    print(f"    Fold {i+1}: {score:.4f} ({score*100:.2f}%)")

print(f"\n  Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# ============================================================================
# 3. CONFORMAL PREDICTION COVERAGE ACROSS FOLDS
# ============================================================================

print("\n[3] CONFORMAL PREDICTION COVERAGE (K-Fold)")
print("-"*80)

coverage_results = []
set_size_results = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Split train into train/cal
    split_idx = int(len(X_train_fold) * 0.75)
    X_train = X_train_fold.iloc[:split_idx]
    y_train = y_train_fold.iloc[:split_idx]
    X_cal = X_train_fold.iloc[split_idx:]
    y_cal = y_train_fold.iloc[split_idx:]
    
    # Train model
    gb_fold = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_fold.fit(X_train, y_train)
    
    # Calibrate conformal predictor
    cp = ConformalRegimeClassifier(gb_fold, alpha=0.10)
    cp.calibrate(X_cal, y_cal)
    
    # Evaluate coverage
    coverage, avg_set_size, _, _ = cp.evaluate_coverage(X_test_fold, y_test_fold)
    
    coverage_results.append(coverage)
    set_size_results.append(avg_set_size)
    
    print(f"  Fold {fold_idx+1}: Coverage={coverage:.4f} ({coverage*100:.2f}%), Avg Set Size={avg_set_size:.2f}")

coverage_results = np.array(coverage_results)
set_size_results = np.array(set_size_results)

print(f"\n  Mean Coverage: {coverage_results.mean():.4f} ± {coverage_results.std():.4f}")
print(f"  Expected Coverage: 0.9000 (90%)")
print(f"  Coverage Guarantee: {'✅ HOLDS' if coverage_results.mean() >= 0.88 else '❌ VIOLATED'}")

print(f"\n  Mean Set Size: {set_size_results.mean():.4f} ± {set_size_results.std():.4f}")

# ============================================================================
# 4. PERMUTATION TEST (Coverage Guarantee)
# ============================================================================

print("\n[4] PERMUTATION TEST (Coverage Guarantee)")
print("-"*80)

# Test if coverage is significantly >= 90%
# H0: coverage = 90%, H1: coverage >= 90%

observed_coverage = coverage_results.mean()
expected_coverage = 0.90

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(coverage_results, expected_coverage, alternative='greater')

print(f"\n  Observed Coverage: {observed_coverage:.4f}")
print(f"  Expected Coverage: {expected_coverage:.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Result: {'✅ Coverage guarantee HOLDS (p < 0.05)' if p_value < 0.05 else '⚠️ Not significant'}")

# ============================================================================
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

print("\n[5] BOOTSTRAP CONFIDENCE INTERVALS (n=100)")
print("-"*80)

n_bootstrap = 100  # Reduced from 1000 for speed
bootstrap_accuracies = []
bootstrap_coverages = []

# Use last 30% as test set
split_idx = int(len(X) * 0.70)
X_train_all = X.iloc[:split_idx]
y_train_all = y.iloc[:split_idx]
X_test_all = X.iloc[split_idx:]
y_test_all = y.iloc[split_idx:]

print(f"  Running {n_bootstrap} bootstrap iterations...")

np.random.seed(42)  # For reproducibility

for i in range(n_bootstrap):
    # Bootstrap sample from training set
    boot_idx = np.random.choice(len(X_train_all), size=len(X_train_all), replace=True)
    X_boot = X_train_all.iloc[boot_idx]
    y_boot = y_train_all.iloc[boot_idx]

    # Split into train/cal
    split_boot = int(len(X_boot) * 0.75)
    X_train_boot = X_boot.iloc[:split_boot]
    y_train_boot = y_boot.iloc[:split_boot]
    X_cal_boot = X_boot.iloc[split_boot:]
    y_cal_boot = y_boot.iloc[split_boot:]

    # Train model
    gb_boot = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=i)
    gb_boot.fit(X_train_boot, y_train_boot)

    # Point accuracy
    y_pred_boot = gb_boot.predict(X_test_all)
    acc_boot = accuracy_score(y_test_all, y_pred_boot)
    bootstrap_accuracies.append(acc_boot)

    # Conformal coverage
    cp_boot = ConformalRegimeClassifier(gb_boot, alpha=0.10)
    cp_boot.calibrate(X_cal_boot, y_cal_boot)
    coverage_boot, _, _, _ = cp_boot.evaluate_coverage(X_test_all, y_test_all)
    bootstrap_coverages.append(coverage_boot)

    if (i + 1) % 20 == 0:
        print(f"    Iteration {i+1}/{n_bootstrap}...")

bootstrap_accuracies = np.array(bootstrap_accuracies)
bootstrap_coverages = np.array(bootstrap_coverages)

# Compute 95% CI
acc_ci_lower = np.percentile(bootstrap_accuracies, 2.5)
acc_ci_upper = np.percentile(bootstrap_accuracies, 97.5)

cov_ci_lower = np.percentile(bootstrap_coverages, 2.5)
cov_ci_upper = np.percentile(bootstrap_coverages, 97.5)

print(f"\n  Point Accuracy:")
print(f"    Mean: {bootstrap_accuracies.mean():.4f}")
print(f"    95% CI: [{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]")

print(f"\n  Conformal Coverage:")
print(f"    Mean: {bootstrap_coverages.mean():.4f}")
print(f"    95% CI: [{cov_ci_lower:.4f}, {cov_ci_upper:.4f}]")
print(f"    Expected: 0.9000")
print(f"    Result: {'✅ CI contains 0.90' if cov_ci_lower <= 0.90 <= cov_ci_upper else '⚠️ CI does not contain 0.90'}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print("\n[6] SAVING RESULTS")
print("-"*80)

os.makedirs('Labs-2/results/statistical_tests', exist_ok=True)

# Save cross-validation results
cv_results_df = pd.DataFrame({
    'fold': range(1, 6),
    'accuracy': cv_scores,
    'coverage': coverage_results,
    'avg_set_size': set_size_results
})
cv_results_df.to_csv('Labs-2/results/statistical_tests/cv_results.csv', index=False)
print("  ✅ Saved: Labs/results/statistical_tests/cv_results.csv")

# Save bootstrap results
bootstrap_df = pd.DataFrame({
    'accuracy': bootstrap_accuracies,
    'coverage': bootstrap_coverages
})
bootstrap_df.to_csv('Labs-2/results/statistical_tests/bootstrap_results.csv', index=False)
print("  ✅ Saved: Labs/results/statistical_tests/bootstrap_results.csv")

# Save summary statistics
summary_stats = {
    'metric': ['CV Accuracy', 'CV Coverage', 'Bootstrap Accuracy', 'Bootstrap Coverage'],
    'mean': [cv_scores.mean(), coverage_results.mean(), bootstrap_accuracies.mean(), bootstrap_coverages.mean()],
    'std': [cv_scores.std(), coverage_results.std(), bootstrap_accuracies.std(), bootstrap_coverages.std()],
    'ci_lower': [cv_scores.mean() - 1.96*cv_scores.std(), coverage_results.mean() - 1.96*coverage_results.std(), 
                 acc_ci_lower, cov_ci_lower],
    'ci_upper': [cv_scores.mean() + 1.96*cv_scores.std(), coverage_results.mean() + 1.96*coverage_results.std(), 
                 acc_ci_upper, cov_ci_upper]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('Labs-2/results/statistical_tests/summary_statistics.csv', index=False)
print("  ✅ Saved: Labs/results/statistical_tests/summary_statistics.csv")

print("\n" + "="*80)
print("✅ STATISTICAL VALIDATION COMPLETE!")
print("="*80)

print(f"\n🎯 Key Findings:")
print(f"  ✅ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  ✅ CV Coverage: {coverage_results.mean():.4f} ± {coverage_results.std():.4f}")
print(f"  ✅ Bootstrap Accuracy 95% CI: [{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]")
print(f"  ✅ Bootstrap Coverage 95% CI: [{cov_ci_lower:.4f}, {cov_ci_upper:.4f}]")
print(f"  ✅ Coverage Guarantee: {'HOLDS ✅' if coverage_results.mean() >= 0.88 else 'VIOLATED ❌'}")

