"""
SIMPLE TimeGAN IMPLEMENTATION (PyTorch)
Lightweight implementation without heavy dependencies
Based on: "Time-series Generative Adversarial Networks" (NeurIPS 2019)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# TimeGAN Components
# ============================================================================

class Embedder(nn.Module):
    """Embeds real sequences into latent space"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))

class Recovery(nn.Module):
    """Recovers data from latent space"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        x, _ = self.rnn(h)
        return self.fc(x)

class Generator(nn.Module):
    """Generates synthetic sequences in latent space"""
    def __init__(self, noise_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, z):
        h, _ = self.rnn(z)
        return torch.sigmoid(self.fc(h))

class Discriminator(nn.Module):
    """Discriminates between real and fake sequences"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        _, h_n = self.rnn(h)
        return torch.sigmoid(self.fc(h_n.squeeze(0)))

# ============================================================================
# Training Functions
# ============================================================================

def train_timegan(real_sequences, seq_len=30, hidden_dim=24, noise_dim=24, 
                  epochs=1000, batch_size=128, lr=0.001):
    """Train TimeGAN"""
    
    n_samples, seq_len, n_features = real_sequences.shape
    
    print(f"Training TimeGAN:")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Features: {n_features}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Device: {device}")
    
    # Initialize models
    embedder = Embedder(n_features, hidden_dim).to(device)
    recovery = Recovery(hidden_dim, n_features).to(device)
    generator = Generator(noise_dim, hidden_dim).to(device)
    discriminator = Discriminator(hidden_dim).to(device)
    
    # Optimizers
    opt_autoencoder = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr)
    opt_generator = optim.Adam(generator.parameters(), lr=lr)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Convert to tensor
    real_data = torch.FloatTensor(real_sequences).to(device)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        for batch_idx, (real_batch,) in enumerate(dataloader):
            batch_size_actual = real_batch.size(0)
            
            # ================================================================
            # Phase 1: Train Autoencoder (Embedder + Recovery)
            # ================================================================
            opt_autoencoder.zero_grad()
            
            h = embedder(real_batch)
            x_recon = recovery(h)
            
            loss_recon = mse_loss(x_recon, real_batch)
            loss_recon.backward()
            opt_autoencoder.step()
            
            # ================================================================
            # Phase 2: Train Discriminator
            # ================================================================
            opt_discriminator.zero_grad()
            
            # Real sequences
            h_real = embedder(real_batch).detach()
            d_real = discriminator(h_real)
            loss_d_real = bce_loss(d_real, torch.ones_like(d_real))
            
            # Fake sequences
            z = torch.randn(batch_size_actual, seq_len, noise_dim).to(device)
            h_fake = generator(z).detach()
            d_fake = discriminator(h_fake)
            loss_d_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_discriminator.step()
            
            # ================================================================
            # Phase 3: Train Generator
            # ================================================================
            opt_generator.zero_grad()
            
            z = torch.randn(batch_size_actual, seq_len, noise_dim).to(device)
            h_fake = generator(z)
            d_fake = discriminator(h_fake)
            
            loss_g = bce_loss(d_fake, torch.ones_like(d_fake))
            loss_g.backward()
            opt_generator.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Recon: {loss_recon.item():.4f}, D: {loss_d.item():.4f}, G: {loss_g.item():.4f}")
    
    return embedder, recovery, generator, discriminator

def generate_synthetic_data(generator, recovery, n_samples, seq_len, noise_dim):
    """Generate synthetic sequences"""
    generator.eval()
    recovery.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, seq_len, noise_dim).to(device)
        h_fake = generator(z)
        x_fake = recovery(h_fake)
    
    return x_fake.cpu().numpy()

# ============================================================================
# Main Experiment
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🔬 EXPERIMENT: SIMPLE TimeGAN")
    print("="*80)
    
    # Load data
    print("\n📥 Loading complete dataset...")
    df = load_complete_dataset()
    
    # Prepare sequences
    print("\n🔧 Preparing sequences...")
    features = ['returns', 'volatility', 'momentum', 'RSI', 'MACD']
    available_features = [f for f in features if f in df.columns]
    print(f"   Using features: {available_features}")
    
    data = df[available_features].values
    
    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    seq_len = 30
    sequences = []
    for i in range(len(data_scaled) - seq_len):
        sequences.append(data_scaled[i:i+seq_len])
    sequences = np.array(sequences)
    
    print(f"   Created {len(sequences)} sequences of length {seq_len}")
    
    # Train TimeGAN
    print("\n🔬 Training TimeGAN...")
    embedder, recovery, generator, discriminator = train_timegan(
        sequences, 
        seq_len=seq_len,
        hidden_dim=24,
        noise_dim=24,
        epochs=500,
        batch_size=128,
        lr=0.001
    )
    
    # Generate synthetic data
    print("\n🎨 Generating synthetic data...")
    synthetic_sequences = generate_synthetic_data(generator, recovery, 1000, seq_len, 24)
    
    print(f"   Generated {len(synthetic_sequences)} synthetic sequences")
    
    # Save results
    print("\n💾 Saving results...")
    os.makedirs('Labs-2/results', exist_ok=True)
    np.save('Labs-2/results/timegan_synthetic.npy', synthetic_sequences)
    print("   ✅ Saved: Labs/results/timegan_synthetic.npy")
    
    print("\n✅ TimeGAN experiment complete!")

