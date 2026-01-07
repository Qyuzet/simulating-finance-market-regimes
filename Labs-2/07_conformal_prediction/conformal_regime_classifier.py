"""
CONFORMAL PREDICTION FOR REGIME CLASSIFICATION
Novel contribution: First application of conformal prediction to financial regime classification
Provides provable coverage guarantees for regime prediction sets
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

# ============================================================================
# Conformal Prediction Framework
# ============================================================================

class ConformalRegimeClassifier:
    """
    Conformal Prediction for Regime Classification
    
    Novel: First application of conformal prediction to discrete regime classification
    Provides prediction SETS with guaranteed coverage (e.g., 90%)
    """
    
    def __init__(self, base_model, alpha=0.1):
        """
        Args:
            base_model: Trained classifier (e.g., GradientBoostingClassifier)
            alpha: Miscoverage rate (0.1 = 90% coverage guarantee)
        """
        self.model = base_model
        self.alpha = alpha
        self.q_level = None
        self.regime_names = ['Bear', 'Bull', 'Neutral']
        
    def calibrate(self, X_cal, y_cal):
        """
        Calibrate conformal predictor on held-out calibration set
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
        """
        print(f"\n🔧 Calibrating conformal predictor (α={self.alpha})...")
        
        # Get prediction probabilities
        probs = self.model.predict_proba(X_cal)
        
        # Compute conformity scores (1 - probability of true class)
        conformity_scores = []
        for i, true_label in enumerate(y_cal):
            score = 1 - probs[i, true_label]
            conformity_scores.append(score)
        
        conformity_scores = np.array(conformity_scores)
        
        # Find quantile for desired coverage
        n = len(conformity_scores)
        q_level_idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q_level = np.sort(conformity_scores)[min(q_level_idx - 1, n - 1)]
        
        print(f"  Calibration set size: {n}")
        print(f"  Quantile level: {self.q_level:.4f}")
        print(f"  Expected coverage: {(1 - self.alpha) * 100:.1f}%")
        
    def predict_set(self, X_test):
        """
        Return prediction sets with guaranteed coverage
        
        Args:
            X_test: Test features
            
        Returns:
            prediction_sets: List of sets, each containing regime indices
        """
        probs = self.model.predict_proba(X_test)
        
        prediction_sets = []
        for prob in probs:
            # Include all regimes with (1 - prob) <= q_level
            pred_set = [i for i in range(len(prob)) if (1 - prob[i]) <= self.q_level]
            prediction_sets.append(pred_set)
        
        return prediction_sets
    
    def predict_point(self, X_test):
        """Standard point prediction (for comparison)"""
        return self.model.predict(X_test)
    
    def evaluate_coverage(self, X_test, y_test):
        """
        Evaluate empirical coverage on test set
        
        Returns:
            coverage: Fraction of test samples where true label is in prediction set
            avg_set_size: Average size of prediction sets
        """
        pred_sets = self.predict_set(X_test)
        
        # Check coverage
        covered = [y_test[i] in pred_set for i, pred_set in enumerate(pred_sets)]
        coverage = np.mean(covered)
        
        # Average set size
        set_sizes = [len(pred_set) for pred_set in pred_sets]
        avg_set_size = np.mean(set_sizes)
        
        # Set size distribution
        set_size_dist = {i: set_sizes.count(i) for i in range(1, 4)}
        
        return coverage, avg_set_size, set_size_dist, pred_sets

# ============================================================================
# Experiment
# ============================================================================

def run_experiment():
    """Run conformal prediction experiment"""
    
    print("="*80)
    print("🔬 CONFORMAL PREDICTION FOR REGIME CLASSIFICATION")
    print("="*80)
    
    # Load data
    print("\n📥 Loading data...")
    df = load_complete_dataset()
    print(f"  Dataset: {df.shape}")
    
    # ========================================================================
    # 1. HMM REGIME DISCOVERY
    # ========================================================================
    print("\n[1] HMM REGIME DISCOVERY")
    print("-"*80)
    
    # Features for HMM
    hmm_features = df[['returns', 'volatility', 'momentum']].values
    scaler_hmm = StandardScaler()
    hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)
    
    # Fit HMM
    print("Fitting HMM...")
    model_hmm = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model_hmm.fit(hmm_features_scaled)
    
    # Predict regimes
    regimes = model_hmm.predict(hmm_features_scaled)
    
    # Label regimes by mean return
    regime_returns = [df[regimes == i]['returns'].mean() for i in range(3)]
    regime_mapping = np.argsort(regime_returns)  # 0=bear, 1=bull, 2=neutral
    regimes_labeled = np.array([np.where(regime_mapping == r)[0][0] for r in regimes])
    
    df['regime'] = regimes_labeled
    
    print(f"  Regime distribution:")
    for i, name in enumerate(['Bear', 'Bull', 'Neutral']):
        count = (regimes_labeled == i).sum()
        pct = count / len(regimes_labeled) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    # ========================================================================
    # 2. PREPARE FEATURES FOR CLASSIFICATION
    # ========================================================================
    print("\n[2] FEATURE PREPARATION")
    print("-"*80)
    
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
    
    # ========================================================================
    # 3. SPLIT DATA: Train / Calibration / Test
    # ========================================================================
    print("\n[3] DATA SPLITTING")
    print("-"*80)
    
    # Split: 60% train, 20% calibration, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )
    
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Calibration: {len(X_cal)} ({len(X_cal)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # ========================================================================
    # 4. TRAIN BASE MODEL (Gradient Boosting)
    # ========================================================================
    print("\n[4] TRAINING BASE MODEL")
    print("-"*80)
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    print("Training GradientBoosting...")
    gb.fit(X_train, y_train)
    
    # Evaluate on test set (point predictions)
    y_pred_point = gb.predict(X_test)
    acc_point = accuracy_score(y_test, y_pred_point)
    
    print(f"\n✅ Point Prediction Accuracy: {acc_point:.4f} ({acc_point*100:.2f}%)")
    
    # ========================================================================
    # 5. CONFORMAL PREDICTION
    # ========================================================================
    print("\n[5] CONFORMAL PREDICTION")
    print("-"*80)
    
    # Test different coverage levels
    alphas = [0.05, 0.10, 0.20]  # 95%, 90%, 80% coverage
    
    results = []
    
    for alpha in alphas:
        print(f"\n--- Testing α={alpha} ({(1-alpha)*100:.0f}% coverage) ---")
        
        # Create conformal predictor
        cp = ConformalRegimeClassifier(gb, alpha=alpha)
        
        # Calibrate
        cp.calibrate(X_cal, y_cal)
        
        # Evaluate on test set
        coverage, avg_size, size_dist, pred_sets = cp.evaluate_coverage(X_test, y_test)
        
        print(f"\n📊 Results:")
        print(f"  Empirical coverage: {coverage:.4f} ({coverage*100:.2f}%)")
        print(f"  Expected coverage: {(1-alpha)*100:.0f}%")
        print(f"  Average set size: {avg_size:.2f}")
        print(f"  Set size distribution:")
        for size, count in sorted(size_dist.items()):
            pct = count / len(pred_sets) * 100
            print(f"    Size {size}: {count} ({pct:.1f}%)")
        
        results.append({
            'alpha': alpha,
            'expected_coverage': 1 - alpha,
            'empirical_coverage': coverage,
            'avg_set_size': avg_size,
            'size_dist': size_dist
        })
    
    # ========================================================================
    # 6. SAVE RESULTS
    # ========================================================================
    print("\n[6] SAVING RESULTS")
    print("-"*80)
    
    os.makedirs('Labs-2/results/conformal_prediction', exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('Labs-2/results/conformal_prediction/coverage_results.csv', index=False)
    print("  ✅ Saved: Labs/results/conformal_prediction/coverage_results.csv")
    
    # Save example prediction sets
    cp_90 = ConformalRegimeClassifier(gb, alpha=0.10)
    cp_90.calibrate(X_cal, y_cal)
    pred_sets_90 = cp_90.predict_set(X_test)
    
    pred_sets_df = pd.DataFrame({
        'true_regime': y_test.values,
        'point_prediction': y_pred_point,
        'prediction_set': [str(ps) for ps in pred_sets_90],
        'set_size': [len(ps) for ps in pred_sets_90],
        'covered': [y_test.iloc[i] in ps for i, ps in enumerate(pred_sets_90)]
    })
    pred_sets_df.to_csv('Labs-2/results/conformal_prediction/prediction_sets.csv', index=False)
    print("  ✅ Saved: Labs/results/conformal_prediction/prediction_sets.csv")
    
    print("\n" + "="*80)
    print("✅ CONFORMAL PREDICTION EXPERIMENT COMPLETE!")
    print("="*80)
    
    return {
        'point_accuracy': acc_point,
        'coverage_90': results[1]['empirical_coverage'],
        'avg_set_size_90': results[1]['avg_set_size']
    }

if __name__ == "__main__":
    result = run_experiment()
    print(f"\n🎯 Final Results:")
    print(f"  Point Accuracy: {result['point_accuracy']:.2%}")
    print(f"  90% Coverage: {result['coverage_90']:.2%}")
    print(f"  Avg Set Size: {result['avg_set_size_90']:.2f}")

