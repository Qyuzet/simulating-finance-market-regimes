"""
EXPERIMENT: HMM-based Regime Discovery
Replace heuristic labels with data-driven HMM approach
Uses SAME data as improved_analysis.py (yfinance + FRED API)
"""

import numpy as np
import pandas as pd
import sys
import os
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_complete_dataset

def download_data():
    """Load complete dataset (same as improved_analysis.py)"""
    print("📥 Loading complete dataset...")
    df = load_complete_dataset()
    return df

def fit_hmm_model(df, n_states=3):
    """Fit HMM model to discover regimes"""
    print(f"\n🔬 Fitting HMM with {n_states} states...")
    
    # Features for HMM
    features = df[['returns', 'volatility', 'momentum']].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    
    model.fit(features_scaled)
    
    # Predict regimes
    regimes = model.predict(features_scaled)
    
    return model, regimes, scaler

def analyze_regimes(df, regimes):
    """Analyze discovered regimes"""
    df_analysis = df.copy()
    df_analysis['regime'] = regimes
    
    print("\n📊 REGIME STATISTICS:")
    print("="*80)
    
    regime_stats = []
    for regime in sorted(df_analysis['regime'].unique()):
        regime_data = df_analysis[df_analysis['regime'] == regime]
        
        stats = {
            'regime': regime,
            'count': len(regime_data),
            'percentage': len(regime_data) / len(df_analysis) * 100,
            'mean_return': regime_data['returns'].mean(),
            'mean_volatility': regime_data['volatility'].mean(),
            'mean_momentum': regime_data['momentum'].mean(),
            'sharpe': regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252)
        }
        regime_stats.append(stats)
        
        print(f"\nRegime {regime}:")
        print(f"  Count: {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"  Mean Return: {stats['mean_return']:.4f}")
        print(f"  Mean Volatility: {stats['mean_volatility']:.4f}")
        print(f"  Mean Momentum: {stats['mean_momentum']:.4f}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}")
    
    return pd.DataFrame(regime_stats)

def label_regimes(regime_stats):
    """Label regimes based on characteristics"""
    # Sort by mean return
    regime_stats = regime_stats.sort_values('mean_return')
    
    labels = {}
    labels[regime_stats.iloc[0]['regime']] = 'bear'  # Lowest return
    labels[regime_stats.iloc[-1]['regime']] = 'bull'  # Highest return
    
    # Middle regime - check volatility
    middle_regime = regime_stats.iloc[1]['regime']
    if regime_stats.iloc[1]['mean_volatility'] > regime_stats['mean_volatility'].mean():
        labels[middle_regime] = 'volatile'
    else:
        labels[middle_regime] = 'neutral'
    
    return labels

def compare_with_heuristic(df):
    """Compare HMM labels with heuristic labels"""
    # Heuristic labels (current method)
    def heuristic_label(row):
        if row['volatility'] > 0.02 and row['momentum'] < -0.001:
            return 'bear'
        elif row['volatility'] < 0.015 and row['momentum'] > 0.001:
            return 'bull'
        else:
            return 'volatile'
    
    df['heuristic_regime'] = df.apply(heuristic_label, axis=1)
    
    return df

def visualize_regime_distribution(df, regime_stats, save_path=None):
    """Generate regime distribution and characteristics visualizations"""
    print("\n📊 Generating visualizations...")

    # Determine save path relative to script location
    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, '..', 'results', 'figures')

    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    print(f"  Saving to: {os.path.abspath(save_path)}")

    # Map numeric regimes to labels for visualization
    regime_stats_sorted = regime_stats.sort_values('mean_return')
    regime_names = ['Bear', 'Bull', 'Neutral']

    # Get actual counts from df
    regime_counts = []
    regime_percentages = []
    for i in range(3):
        count = (df['regime'] == i).sum()
        pct = count / len(df) * 100
        regime_counts.append(count)
        regime_percentages.append(pct)

    # Figure 1: HMM Regime Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#E63946', '#06A77D', '#457B9D']  # Bear=red, Bull=green, Neutral=blue
    bars = ax.bar(regime_names, regime_percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('HMM Regime Distribution (2010-2024)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 60)

    # Add value labels on bars
    for bar, pct, count in zip(bars, regime_percentages, regime_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.2f}%\n({count} days)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_path}/hmm_regime_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}/hmm_regime_distribution.png")

    # Figure 2: Regime Characteristics
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Mean Return by Regime
    mean_returns = [regime_stats[regime_stats['regime'] == i]['mean_return'].values[0] * 100 for i in range(3)]
    bars1 = axes[0].bar(regime_names, mean_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Mean Return (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Daily Return', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    for bar, val in zip(bars1, mean_returns):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.002 if height > 0 else height - 0.002,
                    f'{val:.3f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

    # Mean Volatility by Regime
    mean_vols = [regime_stats[regime_stats['regime'] == i]['mean_volatility'].values[0] * 100 for i in range(3)]
    bars2 = axes[1].bar(regime_names, mean_vols, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Volatility (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Mean Volatility', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, mean_vols):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Sharpe Ratio by Regime
    sharpe_ratios = [regime_stats[regime_stats['regime'] == i]['sharpe'].values[0] for i in range(3)]
    bars3 = axes[2].bar(regime_names, sharpe_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    axes[2].set_title('Annualized Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    for bar, val in zip(bars3, sharpe_ratios):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.1 if height > 0 else height - 0.1,
                    f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_path}/regime_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}/regime_characteristics.png")

def run_experiment():
    """Run HMM regime discovery experiment"""
    print("="*80)
    print("🔬 EXPERIMENT: HMM-BASED REGIME DISCOVERY")
    print("="*80)

    # Download data
    df = download_data()

    # Fit HMM
    model, regimes, scaler = fit_hmm_model(df, n_states=3)

    # Analyze regimes
    regime_stats = analyze_regimes(df, regimes)

    # Label regimes
    regime_labels = label_regimes(regime_stats)
    print(f"\n🏷️  REGIME LABELS: {regime_labels}")

    # Map numeric regimes to labels
    df['hmm_regime'] = [regime_labels[r] for r in regimes]
    df['regime'] = regimes  # Keep numeric for visualization

    # Compare with heuristic
    df = compare_with_heuristic(df)

    # Calculate agreement
    agreement = (df['hmm_regime'] == df['heuristic_regime']).mean()
    print(f"\n📊 Agreement with heuristic labels: {agreement*100:.1f}%")

    # Distribution comparison
    print("\n📊 LABEL DISTRIBUTION COMPARISON:")
    print("\nHMM Labels:")
    print(df['hmm_regime'].value_counts(normalize=True) * 100)
    print("\nHeuristic Labels:")
    print(df['heuristic_regime'].value_counts(normalize=True) * 100)

    # Generate visualizations
    visualize_regime_distribution(df, regime_stats)

    # Return metrics
    return {
        'method': 'HMM',
        'n_states': 3,
        'agreement_with_heuristic': agreement,
        'bull_pct': (df['hmm_regime'] == 'bull').mean(),
        'bear_pct': (df['hmm_regime'] == 'bear').mean(),
        'volatile_pct': (df['hmm_regime'] == 'volatile').mean(),
        'log_likelihood': model.score(scaler.transform(df[['returns', 'volatility', 'momentum']].values))
    }

if __name__ == "__main__":
    result = run_experiment()
    print(f"\n✅ Experiment complete: {result}")

