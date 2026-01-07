"""
EXPERIMENT: TimeGAN for Financial Time Series
State-of-the-art GAN specifically designed for time series
Uses SAME data as improved_analysis.py (yfinance + FRED API)
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

def download_data():
    """Load complete dataset (same as improved_analysis.py)"""
    print("📥 Loading complete dataset...")
    df = load_complete_dataset()
    return df

def prepare_sequences(df, seq_len=30):
    """Prepare sequences for TimeGAN"""
    # Use same features as improved_analysis.py
    features = ['returns', 'volatility', 'momentum', 'RSI', 'MACD']

    # Check which features are available
    available_features = [f for f in features if f in df.columns]
    print(f"   Using features: {available_features}")

    data = df[available_features].values
    
    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    sequences = []
    for i in range(len(data_scaled) - seq_len):
        sequences.append(data_scaled[i:i+seq_len])
    
    return np.array(sequences), scaler

def try_timegan_library():
    """Try using ydata-synthetic TimeGAN library"""
    try:
        from ydata_synthetic.synthesizers.timeseries import TimeGAN
        from ydata_synthetic.synthesizers import ModelParameters
        
        print("✅ ydata-synthetic library available")
        return True
    except ImportError:
        print("⚠️  ydata-synthetic not installed")
        print("   Install with: pip install ydata-synthetic")
        return False

def run_timegan_experiment(sequences, seq_len=30):
    """Run TimeGAN experiment"""
    print("\n🔬 Running TimeGAN experiment...")
    
    if not try_timegan_library():
        print("❌ Cannot run TimeGAN without ydata-synthetic")
        return None
    
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    from ydata_synthetic.synthesizers import ModelParameters
    
    # TimeGAN parameters
    gan_args = ModelParameters(
        batch_size=128,
        lr=5e-4,
        noise_dim=32,
        layers_dim=128,
        latent_dim=24,
        gamma=1
    )
    
    print("📊 Training TimeGAN...")
    print(f"   Sequences: {len(sequences)}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Features: {sequences.shape[2]}")
    
    # Initialize and train
    synth = TimeGAN(
        model_parameters=gan_args,
        hidden_dim=24,
        seq_len=seq_len,
        n_seq=sequences.shape[2],
        gamma=1
    )
    
    # Train
    synth.fit(sequences, train_steps=10000)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic data...")
    synthetic_sequences = synth.sample(n_samples=1000)
    
    return synthetic_sequences, synth

def evaluate_quality(real_sequences, synthetic_sequences):
    """Evaluate synthetic data quality"""
    print("\n📊 EVALUATING SYNTHETIC DATA QUALITY")
    print("="*80)
    
    # Flatten sequences for comparison
    real_flat = real_sequences.reshape(-1, real_sequences.shape[-1])
    synth_flat = synthetic_sequences.reshape(-1, synthetic_sequences.shape[-1])
    
    feature_names = ['returns', 'volatility', 'momentum', 'rsi', 'volume_change']
    
    results = {}
    
    for i, feature in enumerate(feature_names):
        # KS test
        ks_stat, ks_pval = ks_2samp(real_flat[:, i], synth_flat[:, i])
        
        # Jensen-Shannon divergence
        # Create histograms
        real_hist, bins = np.histogram(real_flat[:, i], bins=50, density=True)
        synth_hist, _ = np.histogram(synth_flat[:, i], bins=bins, density=True)
        
        # Normalize
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()
        
        jsd = jensenshannon(real_hist, synth_hist)
        
        results[feature] = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'jsd': jsd,
            'real_mean': real_flat[:, i].mean(),
            'synth_mean': synth_flat[:, i].mean(),
            'real_std': real_flat[:, i].std(),
            'synth_std': synth_flat[:, i].std()
        }
        
        print(f"\n{feature}:")
        print(f"  KS statistic: {ks_stat:.4f} (p={ks_pval:.4f})")
        print(f"  JSD: {jsd:.4f}")
        print(f"  Real mean: {real_flat[:, i].mean():.4f}, Synth mean: {synth_flat[:, i].mean():.4f}")
        print(f"  Real std: {real_flat[:, i].std():.4f}, Synth std: {synth_flat[:, i].std():.4f}")
    
    # Overall quality score
    avg_jsd = np.mean([r['jsd'] for r in results.values()])
    avg_ks_pval = np.mean([r['ks_pvalue'] for r in results.values()])
    
    print(f"\n📊 OVERALL QUALITY:")
    print(f"   Average JSD: {avg_jsd:.4f} (lower is better, <0.1 is good)")
    print(f"   Average KS p-value: {avg_ks_pval:.4f} (higher is better, >0.05 is good)")
    
    return results, avg_jsd, avg_ks_pval

def run_experiment():
    """Run complete TimeGAN experiment"""
    print("="*80)
    print("🔬 EXPERIMENT: TimeGAN FOR FINANCIAL TIME SERIES")
    print("="*80)
    
    # Check if library is available
    if not try_timegan_library():
        return {
            'method': 'TimeGAN',
            'status': 'library_not_installed',
            'avg_jsd': None,
            'avg_ks_pval': None
        }
    
    # Download data
    df = download_data()
    
    # Prepare sequences
    sequences, scaler = prepare_sequences(df, seq_len=30)
    print(f"\n✅ Prepared {len(sequences)} sequences")
    
    # Run TimeGAN
    synthetic_sequences, model = run_timegan_experiment(sequences, seq_len=30)
    
    if synthetic_sequences is None:
        return {
            'method': 'TimeGAN',
            'status': 'failed',
            'avg_jsd': None,
            'avg_ks_pval': None
        }
    
    # Evaluate quality
    results, avg_jsd, avg_ks_pval = evaluate_quality(sequences, synthetic_sequences)
    
    return {
        'method': 'TimeGAN',
        'status': 'success',
        'avg_jsd': avg_jsd,
        'avg_ks_pval': avg_ks_pval,
        'quality_score': 'good' if avg_jsd < 0.1 and avg_ks_pval > 0.05 else 'poor'
    }

if __name__ == "__main__":
    result = run_experiment()
    print(f"\n✅ Experiment complete: {result}")

