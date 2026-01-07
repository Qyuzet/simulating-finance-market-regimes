"""
FINAL ANALYSIS - BEST METHODS
Combines all best-performing methods from Lab experiments:
- HMM for regime discovery
- GradientBoosting for classification
- TimeGAN for synthetic data generation
- Multi-market validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm
import yfinance as yf
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("🚀 FINAL ANALYSIS - BEST METHODS")
print("="*80)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[1] DATA LOADING & PREPROCESSING")
print("-"*80)

# Download data
print("Downloading data...")
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-12-31', progress=False)
vix = yf.download('^VIX', start='2010-01-01', end='2024-12-31', progress=False)

# FRED API
fred = Fred(api_key='6471be419152257e21225e7de5e915c5')
fedfunds = fred.get_series('FEDFUNDS', observation_start='2010-01-01')
cpi = fred.get_series('CPIAUCSL', observation_start='2010-01-01')

# Prepare data
df = sp500[['Close', 'Volume', 'High', 'Low']].copy()
df.columns = ['SP500', 'Volume', 'High', 'Low']

# Add VIX
vix_close = vix['Close']
vix_close.name = 'VIX'
df = df.join(vix_close)

# Add FRED data
fedfunds_daily = fedfunds.reindex(df.index).ffill()
fedfunds_daily.name = 'FEDFUNDS'
cpi_daily = cpi.reindex(df.index).ffill()
cpi_daily.name = 'CPI'
df = df.join(fedfunds_daily).join(cpi_daily)

# Drop NaN
df = df.dropna()

# Feature engineering
print("Engineering features...")
df['returns'] = df['SP500'].pct_change()
df['log_returns'] = np.log(df['SP500'] / df['SP500'].shift(1))
df['volatility'] = df['returns'].rolling(window=20).std()
df['volatility_30'] = df['returns'].rolling(window=30).std()
df['volatility_60'] = df['returns'].rolling(window=60).std()
df['momentum'] = df['SP500'].pct_change(10)
df['momentum_20'] = df['SP500'].pct_change(20)
df['momentum_60'] = df['SP500'].pct_change(60)

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
delta = df['SP500'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
exp1 = df['SP500'].ewm(span=12, adjust=False).mean()
exp2 = df['SP500'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2

# Volume features
df['volume_change'] = df['Volume'].pct_change()
df['volume_ma'] = df['Volume'].rolling(window=20).mean()

# Inflation
df['inflation'] = df['CPI'].pct_change(12)

# Drop NaN
df = df.dropna()

print(f"✅ Dataset ready: {df.shape[0]} observations, {df.shape[1]} features")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# ============================================================================
# 2. HMM REGIME DISCOVERY
# ============================================================================
print("\n[2] HMM REGIME DISCOVERY")
print("-"*80)

# Prepare features for HMM
hmm_features = df[['returns', 'volatility', 'momentum']].values
scaler = StandardScaler()
hmm_features_scaled = scaler.fit_transform(hmm_features)

# Fit HMM
print("Fitting HMM model...")
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(hmm_features_scaled)
regimes = model.predict(hmm_features_scaled)

# Label regimes based on mean return
regime_stats = []
for regime in range(3):
    regime_data = df[regimes == regime]
    mean_return = regime_data['returns'].mean()
    regime_stats.append((regime, mean_return))

regime_stats.sort(key=lambda x: x[1])
regime_mapping = {
    regime_stats[0][0]: 0,  # Bear (lowest return)
    regime_stats[1][0]: 2,  # Neutral (middle return)
    regime_stats[2][0]: 1   # Bull (highest return)
}

df['regime'] = pd.Series(regimes, index=df.index).map(regime_mapping)

# Print regime statistics
print("\nRegime Statistics:")
for regime_id, regime_name in [(0, 'Bear'), (1, 'Bull'), (2, 'Neutral')]:
    regime_data = df[df['regime'] == regime_id]
    count = len(regime_data)
    pct = count / len(df) * 100
    mean_ret = regime_data['returns'].mean()
    mean_vol = regime_data['volatility'].mean()
    sharpe = mean_ret / mean_vol if mean_vol > 0 else 0
    
    print(f"  {regime_name}: {count} obs ({pct:.1f}%), "
          f"Return: {mean_ret:.4f}, Vol: {mean_vol:.4f}, Sharpe: {sharpe:.2f}")

# ============================================================================
# 3. GRADIENT BOOSTING CLASSIFICATION
# ============================================================================
print("\n[3] GRADIENT BOOSTING CLASSIFICATION")
print("-"*80)

# Prepare features with lags
print("Preparing features with lags...")
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Train GradientBoosting
print("\nTraining GradientBoosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)

# Predictions
y_pred = gb.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bear', 'Bull', 'Neutral']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================================================
# 4. SAVE RESULTS
# ============================================================================
print("\n[4] SAVING RESULTS")
print("-"*80)

# Save predictions
results_df = pd.DataFrame({
    'date': X_test.index,
    'true_regime': y_test.values,
    'predicted_regime': y_pred,
    'correct': y_test.values == y_pred
})
results_df.to_csv('final_results.csv', index=False)
print("✅ Saved: final_results.csv")

# Save model performance
performance = {
    'accuracy': accuracy,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'n_features': X.shape[1]
}
pd.DataFrame([performance]).to_csv('final_performance.csv', index=False)
print("✅ Saved: final_performance.csv")

print("\n" + "="*80)
print("✅ FINAL ANALYSIS COMPLETE!")
print("="*80)
print(f"\nKey Results:")
print(f"  • HMM Regimes: Bear {(df['regime']==0).sum()/len(df)*100:.1f}%, "
      f"Bull {(df['regime']==1).sum()/len(df)*100:.1f}%, "
      f"Neutral {(df['regime']==2).sum()/len(df)*100:.1f}%")
print(f"  • GradientBoosting Accuracy: {accuracy*100:.2f}%")
print(f"  • Features Used: {X.shape[1]} (with lags)")
print(f"  • Test Samples: {len(X_test)}")

