"""
CONFORMAL PREDICTION-BASED PORTFOLIO OPTIMIZATION
Novel contribution: First portfolio optimization using regime prediction SETS (not point predictions)
Robust allocation under regime uncertainty with provable coverage guarantees
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

# Import conformal predictor
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '07_conformal_prediction'))
from conformal_regime_classifier import ConformalRegimeClassifier

# ============================================================================
# Portfolio Strategies
# ============================================================================

class RegimePortfolio:
    """Base class for regime-based portfolio strategies"""
    
    def __init__(self, regime_names=['Bear', 'Bull', 'Neutral']):
        self.regime_names = regime_names
        # Regime-specific allocations (equity allocation %)
        self.regime_allocations = {
            0: 0.20,  # Bear: 20% equity, 80% cash (defensive)
            1: 1.00,  # Bull: 100% equity (aggressive)
            2: 0.60   # Neutral: 60% equity, 40% cash (balanced)
        }
    
    def get_allocation(self, regime):
        """Get equity allocation for a regime"""
        return self.regime_allocations.get(regime, 0.60)


class PointPredictionPortfolio(RegimePortfolio):
    """Traditional portfolio using point predictions (single regime)"""
    
    def __init__(self):
        super().__init__()
        self.name = "Point Prediction"
    
    def allocate(self, regime_pred):
        """Allocate based on single predicted regime"""
        return self.get_allocation(regime_pred)


class ConformalPortfolio(RegimePortfolio):
    """Novel: Portfolio using conformal prediction SETS"""
    
    def __init__(self, strategy='conservative'):
        super().__init__()
        self.strategy = strategy
        self.name = f"Conformal ({strategy})"
    
    def allocate(self, pred_set):
        """
        Allocate based on prediction SET (not single regime)
        
        Args:
            pred_set: List of possible regimes (e.g., [0, 2] = bear or neutral)
        
        Returns:
            equity_allocation: Fraction allocated to equity
        """
        if len(pred_set) == 0:
            # Empty set (very uncertain) -> stay in cash
            return 0.0
        
        elif len(pred_set) == 1:
            # Singleton set (certain) -> use regime allocation
            return self.get_allocation(pred_set[0])
        
        else:
            # Multiple regimes (uncertain) -> use strategy
            allocations = [self.get_allocation(r) for r in pred_set]
            
            if self.strategy == 'conservative':
                # Worst-case: minimum allocation (most defensive)
                return min(allocations)
            
            elif self.strategy == 'aggressive':
                # Best-case: maximum allocation (most aggressive)
                return max(allocations)
            
            elif self.strategy == 'average':
                # Average allocation
                return np.mean(allocations)
            
            else:
                return np.mean(allocations)


class BuyAndHold(RegimePortfolio):
    """Baseline: Buy and hold (100% equity always)"""
    
    def __init__(self):
        super().__init__()
        self.name = "Buy & Hold"
    
    def allocate(self, *args):
        """Always 100% equity"""
        return 1.0


# ============================================================================
# Backtesting Engine
# ============================================================================

def backtest_strategy(returns, allocations, transaction_cost=0.001):
    """
    Backtest a portfolio strategy
    
    Args:
        returns: Daily returns of the asset
        allocations: Daily equity allocations (0 to 1)
        transaction_cost: Cost per trade (0.1% default)
    
    Returns:
        portfolio_returns: Daily portfolio returns
        metrics: Performance metrics
    """
    # Portfolio returns = allocation * asset_return
    portfolio_returns = allocations * returns
    
    # Transaction costs (when allocation changes)
    allocation_changes = np.abs(np.diff(allocations, prepend=allocations[0]))
    transaction_costs = allocation_changes * transaction_cost
    portfolio_returns = portfolio_returns - transaction_costs
    
    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Metrics
    total_return = cumulative_returns.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Max drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (portfolio_returns > 0).mean()
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': (allocation_changes > 0.01).sum()
    }
    
    return portfolio_returns, cumulative_returns, metrics


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    """Run CP-based portfolio optimization experiment"""
    
    print("="*80)
    print("💼 CONFORMAL PREDICTION-BASED PORTFOLIO OPTIMIZATION")
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
    
    print(f"  Regime distribution:")
    for i, name in enumerate(['Bear', 'Bull', 'Neutral']):
        count = (regimes_labeled == i).sum()
        pct = count / len(regimes_labeled) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    # ========================================================================
    # 2. PREPARE FEATURES
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
    returns = df['returns'].loc[X.index]
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    # ========================================================================
    # 3. TRAIN/CAL/TEST SPLIT
    # ========================================================================
    print("\n[3] DATA SPLITTING")
    print("-"*80)
    
    # Use last 30% for out-of-sample testing (more realistic)
    split_idx = int(len(X) * 0.70)
    
    X_train_cal = X.iloc[:split_idx]
    y_train_cal = y.iloc[:split_idx]
    returns_train_cal = returns.iloc[:split_idx]
    
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    returns_test = returns.iloc[split_idx:]
    
    # Split train/cal from train_cal
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.25, random_state=42, stratify=y_train_cal
    )
    
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Calibration: {len(X_cal)} ({len(X_cal)/len(X)*100:.1f}%)")
    print(f"  Test (OOS): {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # ========================================================================
    # 4. TRAIN MODEL & CONFORMAL PREDICTOR
    # ========================================================================
    print("\n[4] TRAINING MODEL")
    print("-"*80)
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    
    # Create conformal predictor (90% coverage)
    cp = ConformalRegimeClassifier(gb, alpha=0.10)
    cp.calibrate(X_cal, y_cal)
    
    print("  ✅ Model trained & calibrated")
    
    # ========================================================================
    # 5. GENERATE PREDICTIONS
    # ========================================================================
    print("\n[5] GENERATING PREDICTIONS")
    print("-"*80)
    
    # Point predictions
    y_pred_point = gb.predict(X_test)
    
    # Conformal prediction sets
    pred_sets = cp.predict_set(X_test)
    
    print(f"  Point predictions: {len(y_pred_point)}")
    print(f"  Prediction sets: {len(pred_sets)}")
    
    # ========================================================================
    # 6. BACKTEST STRATEGIES
    # ========================================================================
    print("\n[6] BACKTESTING STRATEGIES")
    print("-"*80)
    
    strategies = {}
    
    # Strategy 1: Buy & Hold
    bh = BuyAndHold()
    alloc_bh = pd.Series([bh.allocate() for _ in range(len(returns_test))], index=returns_test.index)
    ret_bh, cum_bh, metrics_bh = backtest_strategy(returns_test, alloc_bh)
    strategies['Buy & Hold'] = {'returns': ret_bh, 'cumulative': cum_bh, 'metrics': metrics_bh}
    
    # Strategy 2: Point Prediction
    pp = PointPredictionPortfolio()
    alloc_pp = pd.Series([pp.allocate(pred) for pred in y_pred_point], index=returns_test.index)
    ret_pp, cum_pp, metrics_pp = backtest_strategy(returns_test, alloc_pp)
    strategies['Point Prediction'] = {'returns': ret_pp, 'cumulative': cum_pp, 'metrics': metrics_pp}
    
    # Strategy 3: Conformal (Conservative)
    cp_cons = ConformalPortfolio(strategy='conservative')
    alloc_cp_cons = pd.Series([cp_cons.allocate(ps) for ps in pred_sets], index=returns_test.index)
    ret_cp_cons, cum_cp_cons, metrics_cp_cons = backtest_strategy(returns_test, alloc_cp_cons)
    strategies['CP Conservative'] = {'returns': ret_cp_cons, 'cumulative': cum_cp_cons, 'metrics': metrics_cp_cons}
    
    # Strategy 4: Conformal (Average)
    cp_avg = ConformalPortfolio(strategy='average')
    alloc_cp_avg = pd.Series([cp_avg.allocate(ps) for ps in pred_sets], index=returns_test.index)
    ret_cp_avg, cum_cp_avg, metrics_cp_avg = backtest_strategy(returns_test, alloc_cp_avg)
    strategies['CP Average'] = {'returns': ret_cp_avg, 'cumulative': cum_cp_avg, 'metrics': metrics_cp_avg}
    
    # Print results
    print("\n📊 BACKTEST RESULTS (Out-of-Sample):")
    print("-"*80)
    results_df = pd.DataFrame({name: data['metrics'] for name, data in strategies.items()}).T
    print(results_df.to_string())
    
    # ========================================================================
    # 7. SAVE RESULTS
    # ========================================================================
    print("\n[7] SAVING RESULTS")
    print("-"*80)
    
    os.makedirs('Labs-2/results/cp_portfolio', exist_ok=True)
    
    # Save metrics
    results_df.to_csv('Labs-2/results/cp_portfolio/backtest_results.csv')
    print("  ✅ Saved: Labs/results/cp_portfolio/backtest_results.csv")
    
    # Save cumulative returns for visualization
    cum_returns_df = pd.DataFrame({
        name: data['cumulative'] for name, data in strategies.items()
    })
    cum_returns_df.to_csv('Labs-2/results/cp_portfolio/cumulative_returns.csv')
    print("  ✅ Saved: Labs/results/cp_portfolio/cumulative_returns.csv")

    # Save allocations for analysis
    allocations_df = pd.DataFrame({
        'Buy & Hold': alloc_bh,
        'Point Prediction': alloc_pp,
        'CP Conservative': alloc_cp_cons,
        'CP Average': alloc_cp_avg
    })
    allocations_df.to_csv('Labs-2/results/cp_portfolio/allocations.csv')
    print("  ✅ Saved: Labs/results/cp_portfolio/allocations.csv")

    print("\n" + "="*80)
    print("✅ CP-BASED PORTFOLIO OPTIMIZATION COMPLETE!")
    print("="*80)

    return strategies, results_df

if __name__ == "__main__":
    strategies, results = run_experiment()
    
    print(f"\n🎯 Key Findings:")
    print(f"  Buy & Hold Sharpe: {results.loc['Buy & Hold', 'sharpe_ratio']:.3f}")
    print(f"  Point Prediction Sharpe: {results.loc['Point Prediction', 'sharpe_ratio']:.3f}")
    print(f"  CP Conservative Sharpe: {results.loc['CP Conservative', 'sharpe_ratio']:.3f}")
    print(f"  CP Average Sharpe: {results.loc['CP Average', 'sharpe_ratio']:.3f}")

