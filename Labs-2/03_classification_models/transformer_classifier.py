"""
EXPERIMENT: Transformer for Regime Classification
Compare Transformer vs LSTM for sequence classification
Uses SAME data as improved_analysis.py (yfinance + FRED API)
"""

import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    """Transformer for regime classification"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.classifier(x)

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def download_and_prepare_data():
    """Load complete dataset (same as improved_analysis.py)"""
    print("📥 Loading complete dataset...")
    df = load_complete_dataset()

    # Heuristic labels (we'll replace with HMM later)
    def label_regime(row):
        if row['volatility'] > 0.02 and row['momentum'] < -0.001:
            return 0  # bear
        elif row['volatility'] < 0.015 and row['momentum'] > 0.001:
            return 1  # bull
        else:
            return 2  # volatile

    df['regime'] = df.apply(label_regime, axis=1)

    return df

def create_sequences(df, seq_len=30):
    """Create sequences for training"""
    # Use same features as improved_analysis.py
    features = ['returns', 'volatility', 'momentum', 'RSI', 'MACD', 'FEDFUNDS']

    # Check which features are available
    available_features = [f for f in features if f in df.columns]
    print(f"   Using features: {available_features}")

    sequences = []
    labels = []

    for i in range(len(df) - seq_len):
        seq = df[available_features].iloc[i:i+seq_len].values
        label = df['regime'].iloc[i+seq_len]
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)

def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """Train Transformer model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    return model

def run_experiment():
    """Run Transformer experiment"""
    print("="*80)
    print("🔬 EXPERIMENT: TRANSFORMER CLASSIFIER")
    print("="*80)
    
    # Prepare data
    df = download_and_prepare_data()
    sequences, labels = create_sequences(df, seq_len=30)
    
    print(f"\n✅ Created {len(sequences)} sequences")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    model = TransformerClassifier(input_dim=6, d_model=64, nhead=4, num_layers=2, num_classes=3)
    
    # Train
    print("\n🔬 Training Transformer...")
    model = train_model(model, train_loader, test_loader, epochs=50, device=device)
    
    # Evaluate
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())
    
    # Metrics
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=['bear', 'bull', 'volatile'], output_dict=True)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Bear Recall: {report['bear']['recall']:.4f}")
    print(f"   Bull Recall: {report['bull']['recall']:.4f}")
    print(f"   Volatile Recall: {report['volatile']['recall']:.4f}")
    
    return {
        'method': 'Transformer',
        'accuracy': accuracy,
        'bear_recall': report['bear']['recall'],
        'bull_recall': report['bull']['recall'],
        'volatile_recall': report['volatile']['recall'],
        'f1_macro': report['macro avg']['f1-score']
    }

if __name__ == "__main__":
    result = run_experiment()
    print(f"\n✅ Experiment complete: {result}")

