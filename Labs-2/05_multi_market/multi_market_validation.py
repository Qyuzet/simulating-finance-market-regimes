"""
MULTI-MARKET VALIDATION
Test regime classification across 6 major markets
Markets: S&P 500, NASDAQ, FTSE 100, DAX, Nikkei 225, Shanghai Composite
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import os

# Market configurations
MARKETS = {
    'SP500': {
        'ticker': '^GSPC',
        'name': 'S&P 500 (USA)',
        'currency': 'USD'
    },
    'NASDAQ': {
        'ticker': '^IXIC',
        'name': 'NASDAQ Composite (USA)',
        'currency': 'USD'
    },
    'FTSE': {
        'ticker': '^FTSE',
        'name': 'FTSE 100 (UK)',
        'currency': 'GBP'
    },
    'DAX': {
        'ticker': '^GDAXI',
        'name': 'DAX (Germany)',
        'currency': 'EUR'
    },
    'NIKKEI': {
        'ticker': '^N225',
        'name': 'Nikkei 225 (Japan)',
        'currency': 'JPY'
    },
    'SHANGHAI': {
        'ticker': '000001.SS',
        'name': 'Shanghai Composite (China)',
        'currency': 'CNY'
    }
}

START_DATE = '2010-01-01'
END_DATE = '2024-12-31'

def download_market_data(ticker, name):
    """Download and prepare market data"""
    print(f"  Downloading {name}...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if len(df) == 0:
            print(f"    ⚠️  No data available")
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Calculate features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['momentum'] = df['Close'].pct_change(periods=10)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        df = df.dropna()
        
        print(f"    ✅ {len(df)} observations ({df.index.min().date()} to {df.index.max().date()})")
        return df
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def fit_hmm_regimes(df, n_states=3):
    """Fit HMM to discover regimes"""
    features = df[['returns', 'volatility', 'momentum']].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    
    model.fit(features_scaled)
    regimes = model.predict(features_scaled)
    
    # Label regimes based on mean return
    regime_stats = []
    for i in range(n_states):
        mask = regimes == i
        regime_stats.append({
            'regime': i,
            'mean_return': df.loc[mask, 'returns'].mean(),
            'count': mask.sum()
        })
    
    regime_stats = sorted(regime_stats, key=lambda x: x['mean_return'])
    
    # Map: lowest return = bear, highest = bull, middle = neutral
    regime_map = {
        regime_stats[0]['regime']: 'bear',
        regime_stats[-1]['regime']: 'bull',
        regime_stats[1]['regime']: 'neutral' if n_states == 3 else 'volatile'
    }
    
    df['regime'] = [regime_map[r] for r in regimes]
    
    return df, model, regime_map

def calculate_regime_statistics(df):
    """Calculate statistics for each regime"""
    stats = {}
    
    for regime in ['bear', 'bull', 'neutral']:
        mask = df['regime'] == regime
        if mask.sum() == 0:
            continue
            
        regime_data = df[mask]
        
        stats[regime] = {
            'count': mask.sum(),
            'percentage': (mask.sum() / len(df)) * 100,
            'mean_return': regime_data['returns'].mean(),
            'std_return': regime_data['returns'].std(),
            'mean_volatility': regime_data['volatility'].mean(),
            'sharpe': regime_data['returns'].mean() / regime_data['returns'].std() if regime_data['returns'].std() > 0 else 0
        }
    
    return stats

# ============================================================================
# Main Experiment
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🌍 MULTI-MARKET VALIDATION")
    print("="*80)
    
    os.makedirs('Labs-2/results/multi_market', exist_ok=True)
    
    all_results = []
    
    for market_code, market_info in MARKETS.items():
        print(f"\n{'='*80}")
        print(f"📊 {market_info['name']}")
        print(f"{'='*80}")
        
        # Download data
        df = download_market_data(market_info['ticker'], market_info['name'])
        
        if df is None:
            continue
        
        # Fit HMM
        print(f"  Fitting HMM...")
        df, model, regime_map = fit_hmm_regimes(df)
        
        # Calculate statistics
        stats = calculate_regime_statistics(df)
        
        # Print results
        print(f"\n  📊 Regime Distribution:")
        for regime, regime_stats in stats.items():
            print(f"    {regime.upper()}: {regime_stats['count']} ({regime_stats['percentage']:.1f}%)")
            print(f"      Mean Return: {regime_stats['mean_return']:.4f}")
            print(f"      Volatility: {regime_stats['mean_volatility']:.4f}")
            print(f"      Sharpe: {regime_stats['sharpe']:.2f}")
        
        # Store results
        for regime, regime_stats in stats.items():
            all_results.append({
                'market': market_code,
                'market_name': market_info['name'],
                'regime': regime,
                **regime_stats
            })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('Labs-2/results/multi_market/results.csv', index=False)
    
    print(f"\n{'='*80}")
    print("✅ MULTI-MARKET VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: Labs/results/multi_market/results.csv")
    
    # Summary
    print(f"\n📊 SUMMARY ACROSS ALL MARKETS:")
    summary = results_df.groupby('regime').agg({
        'percentage': 'mean',
        'mean_return': 'mean',
        'mean_volatility': 'mean',
        'sharpe': 'mean'
    })
    print(summary)

