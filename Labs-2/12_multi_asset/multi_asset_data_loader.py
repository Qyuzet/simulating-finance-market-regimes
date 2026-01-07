"""
MULTI-ASSET DATA LOADER
Downloads and prepares data for 5 assets: SPY, QQQ, IWM, TLT, GLD
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
FRED_API_KEY = '6471be419152257e21225e7de5e915c5'

# Asset tickers
ASSETS = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF',
    'IWM': 'Russell 2000 ETF',
    'TLT': 'Long-Term Treasury ETF',
    'GLD': 'Gold ETF'
}

def download_multi_asset_data(force_redownload=False):
    """Download data for all 5 assets"""
    
    # Create directories
    Path('data/raw/multi_asset').mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DOWNLOADING MULTI-ASSET DATA")
    print("="*80)
    
    # Download each asset
    asset_data = {}
    for ticker, name in ASSETS.items():
        csv_path = f'data/raw/multi_asset/{ticker}.csv'
        
        if Path(csv_path).exists() and not force_redownload:
            print(f"  Loading {ticker} ({name}) from cache...")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            print(f"  Downloading {ticker} ({name})...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': ticker})
            df.index.name = 'Date'
            df.to_csv(csv_path)
            print(f"    {ticker}: {len(df)} observations")
        
        asset_data[ticker] = df
    
    # Download VIX
    print(f"  Downloading VIX...")
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})
    vix.to_csv('data/raw/multi_asset/VIX.csv')
    print(f"    VIX: {len(vix)} observations")

    # Download FRED data
    print(f"  Downloading FRED data...")
    fred = Fred(api_key=FRED_API_KEY)
    fedfunds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)

    fedfunds.index.name = 'Date'
    cpi.index.name = 'Date'
    fedfunds.to_csv('data/raw/multi_asset/fedfunds.csv', header=True)
    cpi.to_csv('data/raw/multi_asset/cpi.csv', header=True)
    print(f"    Fed Funds Rate: {len(fedfunds)} observations")
    print(f"    CPI: {len(cpi)} observations")
    
    return asset_data, vix, fedfunds, cpi


def calculate_rsi(data, window=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data):
    """Calculate MACD"""
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    return exp1 - exp2


def engineer_features_for_asset(df, asset_col):
    """Engineer features for a single asset"""
    
    # Basic features
    df[f'{asset_col}_returns'] = df[asset_col].pct_change()
    df[f'{asset_col}_log_returns'] = np.log(df[asset_col] / df[asset_col].shift(1))
    
    # Volatility features
    df[f'{asset_col}_volatility'] = df[f'{asset_col}_returns'].rolling(window=20).std()
    df[f'{asset_col}_volatility_30'] = df[f'{asset_col}_returns'].rolling(window=30).std()
    
    # Momentum features
    df[f'{asset_col}_momentum'] = df[asset_col].pct_change(periods=10)
    df[f'{asset_col}_momentum_20'] = df[asset_col].pct_change(periods=20)
    
    # Moving averages
    df[f'{asset_col}_MA10'] = df[asset_col].rolling(window=10).mean()
    df[f'{asset_col}_MA20'] = df[asset_col].rolling(window=20).mean()
    df[f'{asset_col}_MA50'] = df[asset_col].rolling(window=50).mean()
    df[f'{asset_col}_MA200'] = df[asset_col].rolling(window=200).mean()
    
    # Price ratios
    df[f'{asset_col}_price_to_MA10'] = df[asset_col] / df[f'{asset_col}_MA10']
    df[f'{asset_col}_price_to_MA20'] = df[asset_col] / df[f'{asset_col}_MA20']
    df[f'{asset_col}_price_to_MA50'] = df[asset_col] / df[f'{asset_col}_MA50']
    df[f'{asset_col}_price_to_MA200'] = df[asset_col] / df[f'{asset_col}_MA200']
    
    # RSI
    df[f'{asset_col}_RSI'] = calculate_rsi(df[asset_col])
    
    # MACD
    df[f'{asset_col}_MACD'] = calculate_macd(df[asset_col])
    
    return df


if __name__ == "__main__":
    print("Testing multi-asset data loader...")
    asset_data, vix, fedfunds, cpi = download_multi_asset_data()
    print("\nMulti-asset data download complete!")

