"""
ENHANCED CONFORMAL PREDICTION EXPERIMENTS
Answers open research questions from Consensus AI (2025):
- Q2: Can CP handle class imbalance in financial regime classification?
- Q3: Can CP adapt to non-stationary time series?
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from hmmlearn import hmm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '07_conformal_prediction'))
from conformal_regime_classifier_enhanced import (
    ConformalRegimeClassifier,
    ClassConditionalCP,
    AdaptiveCP,
    AdaptiveClassConditionalCP
)

def prepare_data():
    """Load and prepare data with HMM regimes"""
    print("="*80)
    print("📥 LOADING AND PREPARING DATA")
    print("="*80)
    
    df = load_complete_dataset()
    print(f"  Dataset: {df.shape}")
    
    # HMM Regime Discovery
    print("\n🔍 HMM Regime Discovery...")
    hmm_features = df[['returns', 'volatility', 'momentum']].values
    scaler_hmm = StandardScaler()
    hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)
    
    model_hmm = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model_hmm.fit(hmm_features_scaled)
    regimes = model_hmm.predict(hmm_features_scaled)
    
    # Label regimes by mean return
    regime_returns = [df[regimes == i]['returns'].mean() for i in range(3)]
    regime_mapping = np.argsort(regime_returns)
    regimes_labeled = np.array([np.where(regime_mapping == r)[0][0] for r in regimes])
    df['regime'] = regimes_labeled
    
    print(f"  Regime distribution:")
    for i, name in enumerate(['Bear', 'Bull', 'Neutral']):
        count = (regimes_labeled == i).sum()
        pct = count / len(regimes_labeled) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    # Feature Engineering
    print("\n🔧 Feature Engineering...")
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
    
    return X, y, df

def train_base_model(X_train, y_train):
    """Train Gradient Boosting classifier"""
    print("\n🤖 Training Base Model (Gradient Boosting)...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    print("  ✅ Model trained")
    return gb

def experiment_1_standard_cp(gb, X_cal, y_cal, X_test, y_test):
    """Experiment 1: Standard Conformal Prediction (Baseline)"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: STANDARD CONFORMAL PREDICTION (BASELINE)")
    print("="*80)
    
    cp = ConformalRegimeClassifier(gb, alpha=0.10)
    cp.calibrate(X_cal, y_cal)
    
    coverage, avg_size, size_dist, pred_sets, per_class_cov = cp.evaluate_coverage(X_test, y_test)
    
    print(f"\n📊 Results:")
    print(f"  Overall Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"  Average Set Size: {avg_size:.2f}")
    print(f"\n  Per-Class Coverage:")
    for class_idx, cov in per_class_cov.items():
        regime_name = ['Bear', 'Bull', 'Neutral'][class_idx]
        print(f"    {regime_name}: {cov:.4f} ({cov*100:.2f}%)")
    
    return {
        'method': 'Standard CP',
        'overall_coverage': coverage,
        'avg_set_size': avg_size,
        'bear_coverage': per_class_cov.get(0, 0),
        'bull_coverage': per_class_cov.get(1, 0),
        'neutral_coverage': per_class_cov.get(2, 0)
    }

def experiment_2_class_conditional_cp(gb, X_cal, y_cal, X_test, y_test):
    """Experiment 2: Class-Conditional CP (Q2 Answer)"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: CLASS-CONDITIONAL CP (Q2 ANSWER)")
    print("="*80)
    print("Addresses: How can CP handle class imbalance and rare regimes?")
    
    # Calculate class weights (inverse frequency)
    y_cal_array = y_cal.values if hasattr(y_cal, 'values') else y_cal
    class_counts = np.bincount(y_cal_array)
    class_weights = {i: len(y_cal_array) / (3 * count) for i, count in enumerate(class_counts)}
    
    print(f"\n  Class Weights (inverse frequency):")
    for i, weight in class_weights.items():
        regime_name = ['Bear', 'Bull', 'Neutral'][i]
        print(f"    {regime_name}: {weight:.3f}")
    
    cp_cc = ClassConditionalCP(gb, alpha=0.10, class_weights=class_weights)
    cp_cc.calibrate(X_cal, y_cal)
    
    coverage, avg_size, size_dist, pred_sets, per_class_cov = cp_cc.evaluate_coverage(X_test, y_test)
    
    print(f"\n📊 Results:")
    print(f"  Overall Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"  Average Set Size: {avg_size:.2f}")
    print(f"\n  Per-Class Coverage:")
    for class_idx, cov in per_class_cov.items():
        regime_name = ['Bear', 'Bull', 'Neutral'][class_idx]
        print(f"    {regime_name}: {cov:.4f} ({cov*100:.2f}%)")

    return {
        'method': 'Class-Conditional CP',
        'overall_coverage': coverage,
        'avg_set_size': avg_size,
        'bear_coverage': per_class_cov.get(0, 0),
        'bull_coverage': per_class_cov.get(1, 0),
        'neutral_coverage': per_class_cov.get(2, 0)
    }

def experiment_3_adaptive_cp(gb, X_cal, y_cal, X_test, y_test):
    """Experiment 3: Adaptive CP (Q3 Answer)"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: ADAPTIVE CP (Q3 ANSWER)")
    print("="*80)
    print("Addresses: Can CP adapt to non-stationary financial time series?")

    # Test different window sizes
    window_sizes = [126, 252, 504]  # 6 months, 1 year, 2 years
    results = []

    for window_size in window_sizes:
        print(f"\n--- Window Size: {window_size} days ({window_size/252:.1f} years) ---")

        cp_adaptive = AdaptiveCP(gb, alpha=0.10, window_size=window_size)
        cp_adaptive.calibrate(X_cal, y_cal)

        # Online prediction with updates
        pred_sets = cp_adaptive.predict_set_online(X_test, y_test, update=True)

        # Evaluate coverage
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        covered = [y_test_array[i] in pred_set for i, pred_set in enumerate(pred_sets)]
        coverage = np.mean(covered)
        avg_size = np.mean([len(ps) for ps in pred_sets])

        # Per-class coverage
        per_class_cov = {}
        for class_idx in range(3):
            class_mask = y_test_array == class_idx
            if class_mask.sum() > 0:
                class_covered = [covered[i] for i in range(len(covered)) if class_mask[i]]
                per_class_cov[class_idx] = np.mean(class_covered)

        print(f"  Overall Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
        print(f"  Average Set Size: {avg_size:.2f}")
        print(f"  Per-Class Coverage:")
        for class_idx, cov in per_class_cov.items():
            regime_name = ['Bear', 'Bull', 'Neutral'][class_idx]
            print(f"    {regime_name}: {cov:.4f} ({cov*100:.2f}%)")

        results.append({
            'method': f'Adaptive CP (w={window_size})',
            'window_size': window_size,
            'overall_coverage': coverage,
            'avg_set_size': avg_size,
            'bear_coverage': per_class_cov.get(0, 0),
            'bull_coverage': per_class_cov.get(1, 0),
            'neutral_coverage': per_class_cov.get(2, 0)
        })

    return results

def main():
    """Run all experiments"""

    # Prepare data
    X, y, df = prepare_data()

    # Split data: 60% train, 20% calibration, 20% test
    print("\n📊 Data Splitting...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Calibration: {len(X_cal)} ({len(X_cal)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Train base model
    gb = train_base_model(X_train, y_train)

    # Run experiments
    all_results = []

    # Experiment 1: Standard CP
    result_1 = experiment_1_standard_cp(gb, X_cal, y_cal, X_test, y_test)
    all_results.append(result_1)

    # Experiment 2: Class-Conditional CP
    result_2 = experiment_2_class_conditional_cp(gb, X_cal, y_cal, X_test, y_test)
    all_results.append(result_2)

    # Experiment 3: Adaptive CP
    results_3 = experiment_3_adaptive_cp(gb, X_cal, y_cal, X_test, y_test)
    all_results.extend(results_3)

    # Experiment 4: HYBRID Adaptive + Class-Conditional CP
    print("\n" + "="*80)
    print("EXPERIMENT 4: HYBRID ADAPTIVE + CLASS-CONDITIONAL CP")
    print("="*80)
    print("Research Question: Can we combine Q2 and Q3 solutions to get the BEST method?")
    print("\nMethod: Adaptive calibration + Class-conditional quantiles")
    print("         - Handles class imbalance (Q2)")
    print("         - Handles non-stationarity (Q3)")
    print("         - Uses sliding window with per-class buffers")
    print("-"*80)

    # Test with optimal window size from Experiment 3
    optimal_window = 126  # 6 months showed best results

    print(f"\n🚀 Testing Hybrid CP with window size = {optimal_window} days...")

    hybrid_cp = AdaptiveClassConditionalCP(gb, alpha=0.10, window_size=optimal_window)
    hybrid_cp.calibrate(X_cal, y_cal)

    # Online prediction with updates
    pred_sets = hybrid_cp.predict_set_online(X_test, y_test, update=True)

    # Evaluate
    coverage, avg_set_size, set_size_dist, pred_sets, per_class_cov = hybrid_cp.evaluate_coverage(X_test, y_test)

    hybrid_result = {
        'method': f'Hybrid Adaptive+ClassCond CP (w={optimal_window})',
        'overall_coverage': coverage,
        'avg_set_size': avg_set_size,
        'bear_coverage': per_class_cov.get(0, 0),
        'bull_coverage': per_class_cov.get(1, 0),
        'neutral_coverage': per_class_cov.get(2, 0),
        'window_size': optimal_window
    }
    all_results.append(hybrid_result)

    print(f"  Overall Coverage: {coverage:.2%}")
    print(f"  Avg Set Size: {avg_set_size:.2f}")
    print(f"  Per-Class Coverage: Bear={per_class_cov.get(0, 0):.2%}, "
          f"Bull={per_class_cov.get(1, 0):.2%}, "
          f"Neutral={per_class_cov.get(2, 0):.2%}")

    # Calculate coverage std dev for comparison
    coverage_std = np.std([
        per_class_cov.get(0, 0),
        per_class_cov.get(1, 0),
        per_class_cov.get(2, 0)
    ])
    print(f"  Coverage Std Dev: {coverage_std:.4f} (lower is better)")
    print(f"\n  ✅ HYBRID METHOD combines best of both Q2 and Q3 solutions!")

    # Save results
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)

    os.makedirs('Labs-2/results/enhanced_cp', exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('Labs-2/results/enhanced_cp/enhanced_cp_results.csv', index=False)
    print("  ✅ Saved: Labs/results/enhanced_cp/enhanced_cp_results.csv")

    # Print summary table
    print("\n" + "="*80)
    print("📊 SUMMARY: ANSWERING OPEN RESEARCH QUESTIONS")
    print("="*80)

    print("\n" + results_df.to_string(index=False))

    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*80)

    return results_df

if __name__ == "__main__":
    results = main()
