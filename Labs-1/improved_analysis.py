"""
IMPROVED Financial Market Regime Prediction - WGAN-GP + Balanced LSTM
Fixes:
1. WGAN-GP instead of vanilla GAN (fixes mode collapse)
2. Class weights for LSTM (fixes class imbalance)
3. Focal loss (better minority class learning)
4. More technical indicators (RSI, MACD, Bollinger Bands)
5. Better trading strategy (confidence-based position sizing)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import yfinance as yf
from fredapi import Fred
import warnings
import json
from pathlib import Path
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create directories
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('models_improved').mkdir(parents=True, exist_ok=True)
Path('results/figures_improved').mkdir(parents=True, exist_ok=True)
Path('results/metrics_improved').mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
FRED_API_KEY = "6471be419152257e21225e7de5e915c5"

# Improved hyperparameters
LSTM_HIDDEN = 128  # Increased from 64
LSTM_LAYERS = 3    # Increased from 2
LSTM_DROPOUT = 0.4  # Increased from 0.3
SEQ_LENGTH = 30     # Increased from 20
BATCH_SIZE = 64     # Decreased from 128 for better gradients

WGAN_EPOCHS = 5000  # Increased from 2000
WGAN_CRITIC_ITERS = 5  # Train critic 5x per generator update
WGAN_LAMBDA_GP = 10  # Gradient penalty coefficient

print("\n" + "="*80)
print("IMPROVED FINANCIAL MARKET REGIME PREDICTION - WGAN-GP + BALANCED LSTM")
print("="*80)

# ============================================================================
# [1/12] DATA ACQUISITION WITH MORE FEATURES
# ============================================================================
print("\n[1/12] DATA ACQUISITION")
print("-"*80)

# Fetch S&P 500
print(f"Fetching S&P 500 data ({START_DATE} to {END_DATE})...")
sp500 = yf.download('^GSPC', start=START_DATE, end=END_DATE, progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500 = sp500[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'SP500'})
sp500.index.name = 'Date'
sp500.to_csv('data/raw/sp500.csv', index=True)
print(f"[OK] S&P 500: {len(sp500)} observations")

# Fetch VIX (volatility index)
print("Fetching VIX data...")
try:
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})
    vix.index.name = 'Date'
    vix.to_csv('data/raw/vix.csv', index=True)
    print(f"[OK] VIX: {len(vix)} observations")
except Exception as e:
    print(f"[WARNING] Could not fetch VIX: {e}")
    vix = None

# Fetch FRED data
print("Fetching FRED data (Fed Funds Rate, CPI)...")
try:
    fred = Fred(api_key=FRED_API_KEY)
    fedfunds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    
    fedfunds.index.name = 'Date'
    cpi.index.name = 'Date'
    fedfunds.to_csv('data/raw/fedfunds.csv', header=True)
    cpi.to_csv('data/raw/cpi.csv', header=True)
    print(f"[OK] Fed Funds Rate: {len(fedfunds)} observations")
    print(f"[OK] CPI: {len(cpi)} observations")
except Exception as e:
    print(f"Error fetching FRED data: {e}")
    exit(1)

# ============================================================================
# [2/12] ENHANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/12] ENHANCED FEATURE ENGINEERING")
print("-"*80)

# Load data
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
if Path('data/raw/vix.csv').exists():
    vix = pd.read_csv('data/raw/vix.csv', index_col=0, parse_dates=True)
    if isinstance(vix, pd.Series):
        vix = vix.to_frame(name='VIX')
    elif 'VIX' not in vix.columns:
        vix.columns = ['VIX']
else:
    vix = None

# Merge datasets
df = sp500.copy()
fedfunds_daily = fedfunds.reindex(df.index).ffill()
cpi_daily = cpi.reindex(df.index).ffill()

if vix is not None:
    vix_daily = vix.reindex(df.index).ffill()
    df = df.join(vix_daily)

df = df.join(fedfunds_daily).join(cpi_daily)
df = df.dropna()

print(f"Combined dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================
print("Engineering enhanced features...")

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

# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['SP500'], window=14)
df['RSI_30'] = calculate_rsi(df['SP500'], window=30)

# MACD (Moving Average Convergence Divergence)
exp1 = df['SP500'].ewm(span=12, adjust=False).mean()
exp2 = df['SP500'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# Bollinger Bands
df['BB_middle'] = df['SP500'].rolling(window=20).mean()
df['BB_std'] = df['SP500'].rolling(window=20).std()
df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
df['BB_position'] = (df['SP500'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# Volume features (if available)
if 'Volume' in df.columns:
    df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma20']
    df['price_volume'] = df['returns'] * df['volume_ratio']

# High-Low range
if 'High' in df.columns and 'Low' in df.columns:
    df['HL_range'] = (df['High'] - df['Low']) / df['SP500']
    df['HL_range_ma20'] = df['HL_range'].rolling(window=20).mean()

# Macroeconomic features
df['inflation'] = df['CPI'].pct_change(periods=12)  # YoY inflation
df['real_rate'] = df['FEDFUNDS'] - df['inflation']  # Real interest rate

# Drop NaN values
df = df.dropna()
print(f"Features engineered. Shape: {df.shape}")

# ============================================================================
# [3/12] IMPROVED REGIME LABELING
# ============================================================================
print("\n[3/12] IMPROVED REGIME LABELING")
print("-"*80)

# More sophisticated regime labeling using multiple criteria
def label_regime_improved(row):
    """
    Improved regime labeling using multiple indicators
    """
    # Criteria
    high_vol = row['volatility'] > df['volatility'].quantile(0.70)
    low_vol = row['volatility'] < df['volatility'].quantile(0.30)
    positive_momentum = row['momentum'] > 0
    strong_momentum = row['momentum'] > df['momentum'].quantile(0.60)
    weak_momentum = row['momentum'] < df['momentum'].quantile(0.40)
    negative_returns = row['returns'] < -0.01

    # VIX criteria (if available)
    if 'VIX' in df.columns:
        high_vix = row['VIX'] > 25
        low_vix = row['VIX'] < 15
    else:
        high_vix = high_vol
        low_vix = low_vol

    # RSI criteria
    oversold = row['RSI'] < 30
    overbought = row['RSI'] > 70

    # Regime classification
    if high_vol or high_vix:
        return 'volatile'
    elif negative_returns and weak_momentum:
        return 'bear'
    elif positive_momentum and strong_momentum and low_vol:
        return 'bull'
    elif positive_momentum:
        return 'bull'
    else:
        return 'volatile'

df['regime'] = df.apply(label_regime_improved, axis=1)

# Encode regimes
le = LabelEncoder()
df['regime_encoded'] = le.fit_transform(df['regime'])

print("Regime distribution:")
print(df['regime'].value_counts())
print("\nRegime percentages:")
print(df['regime'].value_counts(normalize=True) * 100)

# Save label encoder
with open('models_improved/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("[OK] Regime labeling complete")

# ============================================================================
# [4/12] PREPARE SEQUENCES WITH MORE FEATURES
# ============================================================================
print("\n[4/12] PREPARE SEQUENCES")
print("-"*80)

# Select features for modeling
feature_cols = [
    'returns', 'volatility', 'volatility_30', 'volatility_60',
    'momentum', 'momentum_20', 'momentum_60',
    'price_to_MA10', 'price_to_MA20', 'price_to_MA50', 'price_to_MA200',
    'RSI', 'RSI_30', 'MACD', 'MACD_hist',
    'BB_width', 'BB_position',
    'FEDFUNDS', 'inflation', 'real_rate'
]

# Add VIX if available
if 'VIX' in df.columns:
    feature_cols.append('VIX')

# Add volume features if available
if 'volume_ratio' in df.columns:
    feature_cols.extend(['volume_ratio', 'price_volume'])

if 'HL_range' in df.columns:
    feature_cols.extend(['HL_range', 'HL_range_ma20'])

# Filter to available features
feature_cols = [col for col in feature_cols if col in df.columns]

print(f"Using {len(feature_cols)} features:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

X = df[feature_cols].values
y = df['regime_encoded'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open('models_improved/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create sequences
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LENGTH)
print(f"Sequence shape: {X_seq.shape}")
print(f"Labels shape: {y_seq.shape}")

# Train/Val/Test split (60/20/20)
train_size = int(0.6 * len(X_seq))
val_size = int(0.2 * len(X_seq))

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]
X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Calculate class weights for imbalanced data
unique, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
class_weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
print(f"\nClass weights (for balanced training):")
for cls, weight in class_weights.items():
    regime_name = le.inverse_transform([cls])[0]
    print(f"  {regime_name}: {weight:.3f}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("[OK] Sequences prepared")

# ============================================================================
# [5/12] FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================================
print("\n[5/12] FOCAL LOSS IMPLEMENTATION")
print("-"*80)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Prepare alpha for focal loss
alpha_tensor = torch.FloatTensor([class_weights[i] for i in range(len(unique))]).to(device)
print(f"Focal loss alpha: {alpha_tensor.cpu().numpy()}")
print("[OK] Focal loss ready")

# ============================================================================
# [6/12] IMPROVED LSTM WITH ATTENTION
# ============================================================================
print("\n[6/12] IMPROVED LSTM ARCHITECTURE")
print("-"*80)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch, seq_len, 1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context: (batch, hidden_size)
        return context, attention_weights

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.4):
        super(ImprovedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # NEW: Bidirectional
        )

        # Attention layer
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Fully connected layers with residual connection
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context, attn_weights = self.attention(lstm_out)

        # Batch norm
        context = self.batch_norm(context)

        # FC layers with residual
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out

# Initialize model
input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))

model = ImprovedLSTM(
    input_size=input_size,
    hidden_size=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS,
    num_classes=num_classes,
    dropout=LSTM_DROPOUT
).to(device)

print(f"Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print("[OK] Model initialized")

# ============================================================================
# [7/12] TRAIN IMPROVED LSTM WITH FOCAL LOSS
# ============================================================================
print("\n[7/12] TRAINING IMPROVED LSTM")
print("-"*80)

# Loss and optimizer
criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training loop
num_epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

print(f"Training for {num_epochs} epochs with early stopping (patience={patience})...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(batch_y.cpu().numpy())

    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_acc = accuracy_score(y_val_t.cpu().numpy(), val_predicted.cpu().numpy())

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models_improved/lstm_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('models_improved/lstm_best.pth'))
print("[OK] Training complete")

# ============================================================================
# [8/12] EVALUATE IMPROVED LSTM
# ============================================================================
print("\n[8/12] EVALUATING IMPROVED LSTM")
print("-"*80)

model.eval()
with torch.no_grad():
    # Test predictions
    test_outputs = model(X_test_t)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_probs = torch.softmax(test_outputs, dim=1)

    test_acc = accuracy_score(y_test_t.cpu().numpy(), test_predicted.cpu().numpy())
    test_f1 = f1_score(y_test_t.cpu().numpy(), test_predicted.cpu().numpy(), average='weighted')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test_t.cpu().numpy(),
    test_predicted.cpu().numpy(),
    target_names=le.classes_
))

# Confusion matrix
cm = confusion_matrix(y_test_t.cpu().numpy(), test_predicted.cpu().numpy())
print("\nConfusion Matrix:")
print(cm)

# Save metrics
metrics = {
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'train_losses': [float(x) for x in train_losses],
    'val_losses': [float(x) for x in val_losses],
    'train_accs': [float(x) for x in train_accs],
    'val_accs': [float(x) for x in val_accs],
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(
        y_test_t.cpu().numpy(),
        test_predicted.cpu().numpy(),
        target_names=le.classes_,
        output_dict=True
    )
}

with open('results/metrics_improved/lstm_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("[OK] Evaluation complete")

# ============================================================================
# [9/12] WGAN-GP (WASSERSTEIN GAN WITH GRADIENT PENALTY)
# ============================================================================
print("\n[9/12] WGAN-GP FOR SYNTHETIC DATA")
print("-"*80)

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(WGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

class WGANCritic(nn.Module):
    def __init__(self, input_dim):
        super(WGANCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 1)  # No sigmoid for WGAN
        )

    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1).to(device)

    # Interpolate between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Get critic scores
    d_interpolates = critic(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

# Initialize WGAN-GP
latent_dim = 100
data_dim = 1  # Use only returns for simplicity

generator = WGANGenerator(latent_dim, data_dim).to(device)
critic = WGANCritic(data_dim).to(device)

# Optimizers (different learning rates for stability)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=0.0004, betas=(0.0, 0.9))

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

# Prepare real data (use only returns for simplicity)
real_data = torch.FloatTensor(X_scaled[:, :1]).to(device)  # Just returns column (shape: [N, 1])

# Training WGAN-GP
print(f"\nTraining WGAN-GP for {WGAN_EPOCHS} epochs...")

g_losses = []
c_losses = []
wasserstein_distances = []

for epoch in tqdm(range(WGAN_EPOCHS), desc="WGAN-GP Training"):
    # Train Critic
    for _ in range(WGAN_CRITIC_ITERS):
        optimizer_C.zero_grad()

        # Sample real data
        idx = np.random.randint(0, real_data.size(0), BATCH_SIZE)
        real_batch = real_data[idx]

        # Generate fake data
        z = torch.randn(BATCH_SIZE, latent_dim).to(device)
        fake_batch = generator(z).detach()

        # Critic scores
        real_score = critic(real_batch).mean()
        fake_score = critic(fake_batch).mean()

        # Gradient penalty
        gp = compute_gradient_penalty(critic, real_batch, fake_batch, device)

        # Critic loss (Wasserstein loss + gradient penalty)
        c_loss = fake_score - real_score + WGAN_LAMBDA_GP * gp
        c_loss.backward()
        optimizer_C.step()

    # Train Generator
    optimizer_G.zero_grad()

    z = torch.randn(BATCH_SIZE, latent_dim).to(device)
    fake_batch = generator(z)
    fake_score = critic(fake_batch).mean()

    # Generator loss (maximize critic score for fake data)
    g_loss = -fake_score
    g_loss.backward()
    optimizer_G.step()

    # Record losses
    g_losses.append(g_loss.item())
    c_losses.append(c_loss.item())
    wasserstein_distances.append((real_score - fake_score).item())

    # Print progress
    if (epoch + 1) % 500 == 0:
        print(f"\nEpoch [{epoch+1}/{WGAN_EPOCHS}]")
        print(f"  G Loss: {g_loss.item():.4f}")
        print(f"  C Loss: {c_loss.item():.4f}")
        print(f"  Wasserstein Distance: {wasserstein_distances[-1]:.4f}")

# Save models
torch.save(generator.state_dict(), 'models_improved/wgan_generator.pth')
torch.save(critic.state_dict(), 'models_improved/wgan_critic.pth')

print("\n[OK] WGAN-GP training complete")

# Generate synthetic data
print("\nGenerating synthetic data...")
generator.eval()
with torch.no_grad():
    z = torch.randn(len(real_data), latent_dim).to(device)
    synthetic_data = generator(z).cpu().numpy()

print(f"Synthetic data shape: {synthetic_data.shape}")
print("[OK] Synthetic data generated")

# ============================================================================
# [10/12] IMPROVED TRADING STRATEGY WITH CONFIDENCE-BASED POSITION SIZING
# ============================================================================
print("\n[10/12] IMPROVED TRADING STRATEGY")
print("-"*80)

# Get predictions with confidence scores
model.eval()
with torch.no_grad():
    all_outputs = model(torch.FloatTensor(X_seq).to(device))
    all_probs = torch.softmax(all_outputs, dim=1).cpu().numpy()
    all_preds = np.argmax(all_probs, axis=1)
    all_confidence = np.max(all_probs, axis=1)

# Align with dates
dates = df.index[SEQ_LENGTH:]
returns = df['returns'].values[SEQ_LENGTH:]

# Create strategy DataFrame
strategy_df = pd.DataFrame({
    'date': dates,
    'returns': returns,
    'predicted_regime': all_preds,
    'confidence': all_confidence
}, index=dates)

# Map regime codes to names
strategy_df['regime_name'] = le.inverse_transform(strategy_df['predicted_regime'])

# Improved trading strategy with confidence-based position sizing
def improved_trading_strategy(row):
    """
    Position sizing based on regime prediction and confidence
    - Bull regime + high confidence: 100% long
    - Bull regime + medium confidence: 75% long
    - Bull regime + low confidence: 50% long
    - Volatile regime: 25% long (reduced exposure)
    - Bear regime + high confidence: 0% (cash)
    - Bear regime + low confidence: 25% long (hedge uncertainty)
    """
    regime = row['regime_name']
    confidence = row['confidence']

    if regime == 'bull':
        if confidence > 0.7:
            return 1.0  # 100% long
        elif confidence > 0.5:
            return 0.75  # 75% long
        else:
            return 0.5  # 50% long
    elif regime == 'volatile':
        return 0.25  # 25% long (reduced exposure)
    else:  # bear
        if confidence > 0.7:
            return 0.0  # Cash
        else:
            return 0.25  # Hedge uncertainty

strategy_df['position'] = strategy_df.apply(improved_trading_strategy, axis=1)

# Calculate strategy returns (with 0.1% transaction cost)
strategy_df['position_change'] = strategy_df['position'].diff().abs()
strategy_df['transaction_cost'] = strategy_df['position_change'] * 0.001  # 0.1% cost
strategy_df['strategy_returns'] = (strategy_df['position'].shift(1) * strategy_df['returns']) - strategy_df['transaction_cost']
strategy_df['buy_hold_returns'] = strategy_df['returns']

# Calculate cumulative returns
strategy_df['strategy_cumulative'] = (1 + strategy_df['strategy_returns']).cumprod()
strategy_df['buy_hold_cumulative'] = (1 + strategy_df['buy_hold_returns']).cumprod()

# Split into test period only
test_start_idx = train_size + val_size
test_strategy = strategy_df.iloc[test_start_idx:].copy()

# Calculate performance metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

strategy_total_return = (test_strategy['strategy_cumulative'].iloc[-1] - 1) * 100
buyhold_total_return = (test_strategy['buy_hold_cumulative'].iloc[-1] - 1) * 100

strategy_sharpe = calculate_sharpe_ratio(test_strategy['strategy_returns'].dropna())
buyhold_sharpe = calculate_sharpe_ratio(test_strategy['buy_hold_returns'].dropna())

strategy_max_dd = calculate_max_drawdown(test_strategy['strategy_cumulative'])
buyhold_max_dd = calculate_max_drawdown(test_strategy['buy_hold_cumulative'])

print("\n" + "="*80)
print("IMPROVED TRADING STRATEGY PERFORMANCE (Test Period)")
print("="*80)
print(f"\nStrategy Total Return:     {strategy_total_return:>8.2f}%")
print(f"Buy & Hold Total Return:   {buyhold_total_return:>8.2f}%")
print(f"Outperformance:            {strategy_total_return - buyhold_total_return:>8.2f}%")
print(f"\nStrategy Sharpe Ratio:     {strategy_sharpe:>8.2f}")
print(f"Buy & Hold Sharpe Ratio:   {buyhold_sharpe:>8.2f}")
print(f"\nStrategy Max Drawdown:     {strategy_max_dd*100:>8.2f}%")
print(f"Buy & Hold Max Drawdown:   {buyhold_max_dd*100:>8.2f}%")
print("="*80)

# Save strategy results
strategy_metrics = {
    'strategy_total_return': float(strategy_total_return),
    'buyhold_total_return': float(buyhold_total_return),
    'outperformance': float(strategy_total_return - buyhold_total_return),
    'strategy_sharpe': float(strategy_sharpe),
    'buyhold_sharpe': float(buyhold_sharpe),
    'strategy_max_drawdown': float(strategy_max_dd),
    'buyhold_max_drawdown': float(buyhold_max_dd)
}

with open('results/metrics_improved/strategy_metrics.json', 'w') as f:
    json.dump(strategy_metrics, f, indent=2)

print("[OK] Trading strategy evaluated")

# ============================================================================
# [11/12] VISUALIZATION
# ============================================================================
print("\n[11/12] CREATING VISUALIZATIONS")
print("-"*80)

# 1. Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
axes[0].plot(val_losses, label='Val Loss', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Improved LSTM: Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train Accuracy', alpha=0.7)
axes[1].plot(val_accs, label='Val Accuracy', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Improved LSTM: Training & Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures_improved/lstm_training.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: lstm_training.png")

# 2. Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Improved LSTM: Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('results/figures_improved/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: confusion_matrix.png")

# 3. WGAN-GP training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(g_losses, alpha=0.7, color='blue')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Generator Loss')
axes[0].set_title('WGAN-GP: Generator Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(c_losses, alpha=0.7, color='red')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Critic Loss')
axes[1].set_title('WGAN-GP: Critic Loss')
axes[1].grid(True, alpha=0.3)

axes[2].plot(wasserstein_distances, alpha=0.7, color='green')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Wasserstein Distance')
axes[2].set_title('WGAN-GP: Wasserstein Distance')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures_improved/wgan_training.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: wgan_training.png")

# 4. Synthetic vs Real data comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(real_data.cpu().numpy(), bins=50, alpha=0.7, label='Real', color='blue', density=True)
axes[0].hist(synthetic_data, bins=50, alpha=0.7, label='WGAN-GP Synthetic', color='red', density=True)
axes[0].set_xlabel('Returns')
axes[0].set_ylabel('Density')
axes[0].set_title('Return Distribution: Real vs WGAN-GP Synthetic')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(real_data.cpu().numpy().flatten(), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Real Returns vs Normal Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures_improved/synthetic_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: synthetic_comparison.png")

# 5. Trading strategy performance
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test_strategy.index, test_strategy['buy_hold_cumulative'],
        label='Buy & Hold', linewidth=2, alpha=0.8, color='pink')
ax.plot(test_strategy.index, test_strategy['strategy_cumulative'],
        label='Improved LSTM Strategy', linewidth=2, alpha=0.8, color='darkgreen')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title('Improved Trading Strategy Performance (Test Period)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures_improved/trading_strategy.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: trading_strategy.png")

# 6. Position sizing over time
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(test_strategy.index, test_strategy['position'],
             linewidth=1.5, alpha=0.8, color='darkblue')
axes[0].fill_between(test_strategy.index, 0, test_strategy['position'], alpha=0.3)
axes[0].set_ylabel('Position Size')
axes[0].set_title('Confidence-Based Position Sizing')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.1, 1.1)

axes[1].plot(test_strategy.index, test_strategy['confidence'],
             linewidth=1.5, alpha=0.8, color='orange')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Prediction Confidence')
axes[1].set_title('Model Confidence Over Time')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures_improved/position_sizing.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: position_sizing.png")

print("\n[OK] All visualizations created")

# ============================================================================
# [12/12] SUMMARY REPORT
# ============================================================================
print("\n[12/12] GENERATING SUMMARY REPORT")
print("-"*80)

summary = f"""
{'='*80}
IMPROVED FINANCIAL MARKET REGIME PREDICTION - SUMMARY REPORT
{'='*80}

1. DATA & FEATURES
   - Date Range: {START_DATE} to {END_DATE}
   - Total Observations: {len(df):,}
   - Features Used: {len(feature_cols)}
   - Sequence Length: {SEQ_LENGTH} days

2. REGIME DISTRIBUTION
{df['regime'].value_counts().to_string()}

3. IMPROVED LSTM MODEL
   - Architecture: Bidirectional LSTM with Attention
   - Hidden Size: {LSTM_HIDDEN}
   - Layers: {LSTM_LAYERS}
   - Dropout: {LSTM_DROPOUT}
   - Total Parameters: {sum(p.numel() for p in model.parameters()):,}
   - Loss Function: Focal Loss (gamma=2.0)
   - Class Weights: {dict(zip(le.classes_, [f'{class_weights[i]:.3f}' for i in range(len(le.classes_))]))}

4. MODEL PERFORMANCE
   - Test Accuracy: {test_acc:.4f}
   - Test F1 Score: {test_f1:.4f}

   Per-Class Performance:
{pd.DataFrame(metrics['classification_report']).T.to_string()}

5. WGAN-GP SYNTHETIC DATA
   - Training Epochs: {WGAN_EPOCHS}
   - Final Generator Loss: {g_losses[-1]:.4f}
   - Final Critic Loss: {c_losses[-1]:.4f}
   - Final Wasserstein Distance: {wasserstein_distances[-1]:.4f}

6. IMPROVED TRADING STRATEGY
   - Strategy: Confidence-based position sizing
   - Transaction Cost: 0.1% per trade

   Performance (Test Period):
   - Strategy Total Return:     {strategy_total_return:>8.2f}%
   - Buy & Hold Total Return:   {buyhold_total_return:>8.2f}%
   - Outperformance:            {strategy_total_return - buyhold_total_return:>8.2f}%

   - Strategy Sharpe Ratio:     {strategy_sharpe:>8.2f}
   - Buy & Hold Sharpe Ratio:   {buyhold_sharpe:>8.2f}

   - Strategy Max Drawdown:     {strategy_max_dd*100:>8.2f}%
   - Buy & Hold Max Drawdown:   {buyhold_max_dd*100:>8.2f}%

7. KEY IMPROVEMENTS
   ✓ Bidirectional LSTM with attention mechanism
   ✓ Focal loss for class imbalance
   ✓ Class weights for balanced training
   ✓ WGAN-GP instead of vanilla GAN
   ✓ Enhanced feature engineering (RSI, MACD, Bollinger Bands)
   ✓ Confidence-based position sizing
   ✓ Transaction cost modeling

8. FILES GENERATED
   - Models: models_improved/
   - Figures: results/figures_improved/
   - Metrics: results/metrics_improved/

{'='*80}
ANALYSIS COMPLETE
{'='*80}
"""

print(summary)

# Save summary (with UTF-8 encoding to handle special characters)
with open('results/metrics_improved/summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n[OK] Summary report saved")
print("\n" + "="*80)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*80)

