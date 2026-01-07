"""
GENERATE MULTI-ASSET FIGURES
Creates publication-quality figures for multi-asset CP results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Create output directory
Path('Labs-2/results/multi_asset').mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING MULTI-ASSET FIGURES")
print("="*80)

# Load results
df = pd.read_csv('Labs-2/results/multi_asset/multi_asset_cp_results.csv')

print(f"\nLoaded results: {len(df)} rows")
print(df.head())

# ============================================================================
# FIGURE 1: Cross-Asset Coverage Comparison
# ============================================================================

print("\n[1] Generating cross-asset coverage comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Overall Coverage
ax = axes[0]
standard_cp = df[df['method'] == 'Standard CP']
class_cp = df[df['method'] == 'Class-Conditional CP']

x = np.arange(len(standard_cp))
width = 0.35

bars1 = ax.bar(x - width/2, standard_cp['overall_coverage'], width, 
               label='Standard CP', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, class_cp['overall_coverage'], width,
               label='Class-Conditional CP', alpha=0.8, color='coral')

ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
ax.set_xlabel('Asset', fontweight='bold', fontsize=12)
ax.set_ylabel('Overall Coverage', fontweight='bold', fontsize=12)
ax.set_title('(A) Overall Coverage Across Assets', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(standard_cp['asset'])
ax.legend(loc='lower right')
ax.set_ylim([0.5, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Panel B: Average Set Size
ax = axes[1]
bars1 = ax.bar(x - width/2, standard_cp['avg_set_size'], width,
               label='Standard CP', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, class_cp['avg_set_size'], width,
               label='Class-Conditional CP', alpha=0.8, color='coral')

ax.set_xlabel('Asset', fontweight='bold', fontsize=12)
ax.set_ylabel('Average Set Size', fontweight='bold', fontsize=12)
ax.set_title('(B) Average Set Size Across Assets', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(standard_cp['asset'])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Labs-2/results/multi_asset/cross_asset_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: Labs/results/multi_asset/cross_asset_comparison.png")

# ============================================================================
# FIGURE 2: Per-Regime Coverage Heatmap
# ============================================================================

print("\n[2] Generating per-regime coverage heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Standard CP
ax = axes[0]
standard_cp = df[df['method'] == 'Standard CP']
coverage_matrix = standard_cp[['bear_coverage', 'bull_coverage', 'neutral_coverage']].values
assets = standard_cp['asset'].values

im = ax.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(['Bear', 'Bull', 'Neutral'])
ax.set_yticklabels(assets)
ax.set_title('(A) Standard CP: Per-Regime Coverage', fontweight='bold', fontsize=13)

# Add text annotations
for i in range(len(assets)):
    for j in range(3):
        text = ax.text(j, i, f'{coverage_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax, label='Coverage')

# Panel B: Class-Conditional CP
ax = axes[1]
class_cp = df[df['method'] == 'Class-Conditional CP']
coverage_matrix = class_cp[['bear_coverage', 'bull_coverage', 'neutral_coverage']].values

im = ax.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(['Bear', 'Bull', 'Neutral'])
ax.set_yticklabels(assets)
ax.set_title('(B) Class-Conditional CP: Per-Regime Coverage', fontweight='bold', fontsize=13)

# Add text annotations
for i in range(len(assets)):
    for j in range(3):
        text = ax.text(j, i, f'{coverage_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax, label='Coverage')

plt.tight_layout()
plt.savefig('Labs-2/results/multi_asset/per_regime_coverage_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: Labs/results/multi_asset/per_regime_coverage_heatmap.png")

# ============================================================================
# FIGURE 3: Coverage Standard Deviation (Balance Metric)
# ============================================================================

print("\n[3] Generating coverage balance comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

# Calculate std dev for each asset and method
results = []
for _, row in df.iterrows():
    coverages = [row['bear_coverage'], row['bull_coverage'], row['neutral_coverage']]
    std_dev = np.std(coverages, ddof=1) * 100  # Convert to percentage
    results.append({
        'asset': row['asset'],
        'method': row['method'],
        'std_dev': std_dev
    })

results_df = pd.DataFrame(results)

# Plot
standard_cp = results_df[results_df['method'] == 'Standard CP']
class_cp = results_df[results_df['method'] == 'Class-Conditional CP']

x = np.arange(len(standard_cp))
width = 0.35

bars1 = ax.bar(x - width/2, standard_cp['std_dev'], width,
               label='Standard CP', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, class_cp['std_dev'], width,
               label='Class-Conditional CP', alpha=0.8, color='coral')

ax.set_xlabel('Asset', fontweight='bold', fontsize=12)
ax.set_ylabel('Coverage Std Dev (%)', fontweight='bold', fontsize=12)
ax.set_title('Coverage Balance Across Regimes (Lower = More Balanced)', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(standard_cp['asset'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Labs-2/results/multi_asset/coverage_balance.png', dpi=300, bbox_inches='tight')
print("  Saved: Labs/results/multi_asset/coverage_balance.png")

print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE!")
print("="*80)
print("\nGenerated 3 figures:")
print("  1. cross_asset_comparison.png")
print("  2. per_regime_coverage_heatmap.png")
print("  3. coverage_balance.png")

