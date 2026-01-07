"""
UNIFIED DATA LOADER
All experiments use the SAME data as your main analysis
Combines: yfinance (S&P 500, VIX) + FRED API (Fed Funds, CPI)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from pathlib import Path
import os

# Configuration
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
FRED_API_KEY = "6471be419152257e21225e7de5e915c5"

def download_raw_data(force_redownload=False):
    """Download raw data from yfinance and FRED API"""
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    
    # Check if data already exists
    if not force_redownload:
        if (Path('data/raw/sp500.csv').exists() and 
            Path('data/raw/fedfunds.csv').exists() and 
            Path('data/raw/cpi.csv').exists()):
            print("✅ Raw data already exists. Loading from cache...")
            return load_raw_data()
    
    print("📥 Downloading raw data...")
    
    # 1. S&P 500 from yfinance
    print(f"  Fetching S&P 500 ({START_DATE} to {END_DATE})...")
    sp500 = yf.download('^GSPC', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    sp500 = sp500[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'SP500'})
    sp500.index.name = 'Date'
    sp500.to_csv('data/raw/sp500.csv', index=True)
    print(f"  ✅ S&P 500: {len(sp500)} observations")
    
    # 2. VIX from yfinance
    print("  Fetching VIX...")
    try:
        vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix[['Close']].rename(columns={'Close': 'VIX'})
        vix.index.name = 'Date'
        vix.to_csv('data/raw/vix.csv', index=True)
        print(f"  ✅ VIX: {len(vix)} observations")
    except Exception as e:
        print(f"  ⚠️  VIX failed: {e}")
    
    # 3. FRED data (Fed Funds Rate, CPI)
    print("  Fetching FRED data...")
    try:
        fred = Fred(api_key=FRED_API_KEY)
        fedfunds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
        cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
        
        fedfunds.index.name = 'Date'
        cpi.index.name = 'Date'
        fedfunds.to_csv('data/raw/fedfunds.csv', header=True)
        cpi.to_csv('data/raw/cpi.csv', header=True)
        print(f"  ✅ Fed Funds Rate: {len(fedfunds)} observations")
        print(f"  ✅ CPI: {len(cpi)} observations")
    except Exception as e:
        print(f"  ❌ FRED data failed: {e}")
        raise
    
    return load_raw_data()

def load_raw_data():
    """Load raw data from CSV files"""
    sp500 = pd.read_csv('data/raw/sp500.csv', index_col=0, parse_dates=True)
    fedfunds = pd.read_csv('data/raw/fedfunds.csv', index_col=0, parse_dates=True)
    cpi = pd.read_csv('data/raw/cpi.csv', index_col=0, parse_dates=True)
    
    # Ensure proper column names
    if isinstance(fedfunds, pd.Series):
        fedfunds = fedfunds.to_frame(name='FEDFUNDS')
    elif 'FEDFUNDS' not in fedfunds.columns:
        fedfunds.columns = ['FEDFUNDS']
        
    if isinstance(cpi, pd.Series):
        cpi = cpi.to_frame(name='CPI')
    elif 'CPI' not in cpi.columns:
        cpi.columns = ['CPI']
    
    # Load VIX if available
    vix = None
    if Path('data/raw/vix.csv').exists():
        vix = pd.read_csv('data/raw/vix.csv', index_col=0, parse_dates=True)
        if isinstance(vix, pd.Series):
            vix = vix.to_frame(name='VIX')
        elif 'VIX' not in vix.columns:
            vix.columns = ['VIX']
    
    return sp500, fedfunds, cpi, vix

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

def engineer_features(df):
    """Engineer all features (same as improved_analysis.py)"""
    
    # Basic features
    df['returns'] = df['SP500'].pct_change()
    df['log_returns'] = np.log(df['SP500'] / df['SP500'].shift(1))
    
    # Volatility features
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    df['volatility_60'] = df['returns'].rolling(window=60).std()
    
    # Momentum features
    df['momentum'] = df['SP500'].pct_change(periods=10)
    df['momentum_20'] = df['SP500'].pct_change(periods=20)
    df['momentum_60'] = df['SP500'].pct_change(periods=60)
    
    # Moving averages
    df['MA10'] = df['SP500'].rolling(window=10).mean()
    df['MA20'] = df['SP500'].rolling(window=20).mean()
    df['MA50'] = df['SP500'].rolling(window=50).mean()
    df['MA200'] = df['SP500'].rolling(window=200).mean()
    
    # Price ratios
    df['price_to_MA10'] = df['SP500'] / df['MA10']
    df['price_to_MA20'] = df['SP500'] / df['MA20']
    df['price_to_MA50'] = df['SP500'] / df['MA50']
    df['price_to_MA200'] = df['SP500'] / df['MA200']
    
    # RSI
    df['RSI'] = calculate_rsi(df['SP500'])
    
    # MACD
    df['MACD'] = calculate_macd(df['SP500'])
    
    # Volume features (if available)
    if 'Volume' in df.columns:
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    
    # CPI inflation rate
    if 'CPI' in df.columns:
        df['inflation'] = df['CPI'].pct_change(periods=12)  # YoY inflation
    
    return df

def load_complete_dataset(force_redownload=False):
    """
    Load complete dataset with all features
    This is the SAME data used in improved_analysis.py
    """
    
    # Download/load raw data
    sp500, fedfunds, cpi, vix = download_raw_data(force_redownload)
    
    # Merge datasets
    df = sp500.copy()
    fedfunds_daily = fedfunds.reindex(df.index).ffill()
    cpi_daily = cpi.reindex(df.index).ffill()
    
    if vix is not None:
        vix_daily = vix.reindex(df.index).ffill()
        df = df.join(vix_daily)
    
    df = df.join(fedfunds_daily).join(cpi_daily)
    df = df.dropna()
    
    # Engineer features
    df = engineer_features(df)
    df = df.dropna()
    
    print(f"\n✅ Complete dataset loaded:")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"   Features: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Test data loader
    print("="*80)
    print("🔬 TESTING DATA LOADER")
    print("="*80)
    
    df = load_complete_dataset()
    
    print(f"\n📊 Dataset Info:")
    print(df.info())
    print(f"\n📊 First few rows:")
    print(df.head())
    print(f"\n📊 Last few rows:")
    print(df.tail())

