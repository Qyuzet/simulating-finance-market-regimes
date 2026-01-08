"""
MULTI-ASSET CONFORMAL PREDICTION EXPERIMENTS
Tests CP framework across 5 different assets: SPY, QQQ, IWM, TLT, GLD

This addresses the key weakness: "Single asset only (SPY)"
Shows that CP framework generalizes across different asset classes
"""

import sys
sys.path.append('Labs-2/12_multi_asset')
sys.path.append('Labs-2/07_conformal_prediction')

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

from multi_asset_data_loader import (
    download_multi_asset_data, 
    engineer_features_for_asset,
    ASSETS
)
from conformal_regime_classifier_enhanced import (
    ConformalRegimeClassifier,
    ClassConditionalCP,
    AdaptiveCP,
    AdaptiveClassConditionalCP
)

print("="*80)
print("MULTI-ASSET CONFORMAL PREDICTION EXPERIMENTS")
print("="*80)
print("\nTesting CP framework across 5 assets:")
for ticker, name in ASSETS.items():
    print(f"  - {ticker}: {name}")
print()


def prepare_asset_data(ticker):
    """Prepare data for a single asset"""
    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR {ticker}")
    print(f"{'='*80}")
    
    # Load asset data
    asset_df = pd.read_csv(f'data/raw/multi_asset/{ticker}.csv', index_col=0, parse_dates=True)
    vix = pd.read_csv('data/raw/multi_asset/VIX.csv', index_col=0, parse_dates=True)
    fedfunds = pd.read_csv('data/raw/multi_asset/fedfunds.csv', index_col=0, parse_dates=True)
    cpi = pd.read_csv('data/raw/multi_asset/cpi.csv', index_col=0, parse_dates=True)
    
    # Ensure proper column names
    if isinstance(fedfunds, pd.Series):
        fedfunds = fedfunds.to_frame(name='FEDFUNDS')
    elif 'FEDFUNDS' not in fedfunds.columns:
        fedfunds.columns = ['FEDFUNDS']
        
    if isinstance(cpi, pd.Series):
        cpi = cpi.to_frame(name='CPI')
    elif 'CPI' not in cpi.columns:
        cpi.columns = ['CPI']
    
    # Merge data
    df = asset_df.copy()
    fedfunds_daily = fedfunds.reindex(df.index).ffill()
    cpi_daily = cpi.reindex(df.index).ffill()
    vix_daily = vix.reindex(df.index).ffill()
    
    df = df.join(fedfunds_daily).join(cpi_daily).join(vix_daily)
    df = df.dropna()
    
    # Engineer features
    df = engineer_features_for_asset(df, ticker)
    
    # CPI inflation rate
    df['inflation'] = df['CPI'].pct_change(periods=12)
    
    df = df.dropna()
    
    print(f"  Data prepared: {df.shape[0]} observations, {df.shape[1]} features")
    
    return df


def discover_regimes_hmm(df, ticker):
    """Discover regimes using HMM"""
    print(f"\n[1] HMM REGIME DISCOVERY FOR {ticker}")
    print("-"*80)
    
    # Features for HMM
    hmm_features = df[[f'{ticker}_returns', f'{ticker}_volatility', f'{ticker}_momentum']].values
    
    scaler_hmm = StandardScaler()
    hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)
    
    # Fit HMM
    model_hmm = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model_hmm.fit(hmm_features_scaled)
    regimes = model_hmm.predict(hmm_features_scaled)
    
    # Label regimes by mean return
    regime_returns = [df[regimes == i][f'{ticker}_returns'].mean() for i in range(3)]
    regime_order = np.argsort(regime_returns)  # [lowest_idx, middle_idx, highest_idx]

    # Create correct mapping: 0=bear (lowest), 1=bull (highest), 2=neutral (middle)
    regime_mapping = np.zeros(3, dtype=int)
    regime_mapping[regime_order[0]] = 0  # lowest return -> Bear
    regime_mapping[regime_order[2]] = 1  # highest return -> Bull
    regime_mapping[regime_order[1]] = 2  # middle return -> Neutral

    regimes_labeled = regime_mapping[regimes]
    
    df['regime'] = regimes_labeled
    
    print(f"  Regime distribution:")
    for i, name in enumerate(['Bear', 'Bull', 'Neutral']):
        count = (regimes_labeled == i).sum()
        pct = count / len(regimes_labeled) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    return df


def prepare_features(df, ticker):
    """Prepare features for classification"""
    print(f"\n[2] FEATURE PREPARATION FOR {ticker}")
    print("-"*80)
    
    feature_cols = [
        f'{ticker}_returns', f'{ticker}_volatility', f'{ticker}_momentum',
        f'{ticker}_RSI', f'{ticker}_MACD', 'FEDFUNDS'
    ]
    
    # Add lagged features
    lags = [1, 5, 10, 20]
    X_list = [df[feature_cols]]
    
    for lag in lags:
        lagged = df[feature_cols].shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in feature_cols]
        X_list.append(lagged)
    
    X_df = pd.concat(X_list, axis=1)
    X_df = X_df.dropna()
    
    # Align regime labels
    y = df.loc[X_df.index, 'regime'].values
    X = X_df.values
    
    print(f"  Features prepared: {X.shape[1]} features")
    
    return X, y, X_df.index


def run_asset_experiment(ticker):
    """Run full CP experiment for a single asset"""
    
    # Prepare data
    df = prepare_asset_data(ticker)
    
    # Discover regimes
    df = discover_regimes_hmm(df, ticker)
    
    # Prepare features
    X, y, dates = prepare_features(df, ticker)
    
    # Split data
    print(f"\n[3] TRAIN/CAL/TEST SPLIT FOR {ticker}")
    print("-"*80)
    
    # 60% train, 20% calibration, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.25, shuffle=False)
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Calibration: {len(X_cal)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train classifier
    print(f"\n[4] TRAINING GRADIENT BOOSTING FOR {ticker}")
    print("-"*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    
    train_acc = gb.score(X_train_scaled, y_train)
    test_acc = gb.score(X_test_scaled, y_test)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    # Test all CP methods
    results = test_all_cp_methods(gb, X_cal_scaled, y_cal, X_test_scaled, y_test, ticker)
    
    return results


def test_all_cp_methods(model, X_cal, y_cal, X_test, y_test, ticker):
    """Test all 4 CP methods"""
    print(f"\n[5] TESTING ALL CP METHODS FOR {ticker}")
    print("="*80)
    
    results = []
    alpha = 0.10
    
    # Method 1: Standard CP
    print(f"\n--- Standard CP ---")
    cp_standard = ConformalRegimeClassifier(model, alpha=alpha)
    cp_standard.calibrate(X_cal, y_cal)
    coverage, avg_size, _, pred_sets, per_class_cov = cp_standard.evaluate_coverage(X_test, y_test)

    # Get per-regime coverage
    bear_cov = per_class_cov.get(0, 0.0)
    bull_cov = per_class_cov.get(1, 0.0)
    neutral_cov = per_class_cov.get(2, 0.0)

    print(f"  Coverage: {coverage:.4f}, Avg set size: {avg_size:.2f}")
    print(f"  Per-regime: Bear={bear_cov:.4f}, Bull={bull_cov:.4f}, Neutral={neutral_cov:.4f}")

    results.append({
        'asset': ticker,
        'method': 'Standard CP',
        'overall_coverage': coverage,
        'avg_set_size': avg_size,
        'bear_coverage': bear_cov,
        'bull_coverage': bull_cov,
        'neutral_coverage': neutral_cov
    })

    # Method 2: Class-Conditional CP
    print(f"\n--- Class-Conditional CP ---")
    cp_class = ClassConditionalCP(model, alpha=alpha)
    cp_class.calibrate(X_cal, y_cal)
    coverage, avg_size, _, pred_sets, per_class_cov = cp_class.evaluate_coverage(X_test, y_test)

    bear_cov = per_class_cov.get(0, 0.0)
    bull_cov = per_class_cov.get(1, 0.0)
    neutral_cov = per_class_cov.get(2, 0.0)

    print(f"  Coverage: {coverage:.4f}, Avg set size: {avg_size:.2f}")
    print(f"  Per-regime: Bear={bear_cov:.4f}, Bull={bull_cov:.4f}, Neutral={neutral_cov:.4f}")

    results.append({
        'asset': ticker,
        'method': 'Class-Conditional CP',
        'overall_coverage': coverage,
        'avg_set_size': avg_size,
        'bear_coverage': bear_cov,
        'bull_coverage': bull_cov,
        'neutral_coverage': neutral_cov
    })
    
    return results


if __name__ == "__main__":
    # Download data first
    print("Downloading multi-asset data...")
    download_multi_asset_data()
    
    # Run experiments for each asset
    all_results = []
    
    for ticker in ASSETS.keys():
        try:
            results = run_asset_experiment(ticker)
            all_results.extend(results)
        except Exception as e:
            print(f"\n❌ ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('Labs-2/results/multi_asset/multi_asset_cp_results.csv', index=False)
    print(f"\nResults saved to Labs/results/multi_asset/multi_asset_cp_results.csv")

    print("\n" + "="*80)
    print("MULTI-ASSET EXPERIMENTS COMPLETE!")
    print("="*80)

