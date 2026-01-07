"""
VISUALIZE CP-BASED PORTFOLIO RESULTS
Publication-quality figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

print("="*80)
print("📊 VISUALIZING CP-BASED PORTFOLIO RESULTS")
print("="*80)

# Load results
results_df = pd.read_csv('Labs-2/results/cp_portfolio/backtest_results.csv', index_col=0)
cum_returns_df = pd.read_csv('Labs-2/results/cp_portfolio/cumulative_returns.csv', index_col=0, parse_dates=True)
allocations_df = pd.read_csv('Labs-2/results/cp_portfolio/allocations.csv', index_col=0, parse_dates=True)

print(f"\n✅ Loaded results:")
print(f"  Strategies: {len(results_df)}")
print(f"  Time periods: {len(cum_returns_df)}")

# ============================================================================
# Figure 1: Performance Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Cumulative Returns
ax1 = axes[0, 0]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
for i, col in enumerate(cum_returns_df.columns):
    ax1.plot(cum_returns_df.index, cum_returns_df[col], label=col, linewidth=2.5, color=colors[i])

ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax1.set_title('Cumulative Returns (Out-of-Sample)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Sharpe Ratio Comparison
ax2 = axes[0, 1]
sharpe_ratios = results_df['sharpe_ratio'].sort_values(ascending=False)
bars = ax2.barh(range(len(sharpe_ratios)), sharpe_ratios.values, color=colors[:len(sharpe_ratios)], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(sharpe_ratios)))
ax2.set_yticklabels(sharpe_ratios.index)
ax2.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sharpe_ratios.values)):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=11, fontweight='bold')

# Plot 3: Max Drawdown Comparison
ax3 = axes[1, 0]
drawdowns = results_df['max_drawdown'].sort_values() * 100  # Convert to %
bars = ax3.barh(range(len(drawdowns)), drawdowns.values, color=colors[:len(drawdowns)], alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(drawdowns)))
ax3.set_yticklabels(drawdowns.index)
ax3.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
ax3.set_title('Maximum Drawdown (Lower is Better)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, drawdowns.values)):
    ax3.text(val - 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
            va='center', ha='right', fontsize=11, fontweight='bold', color='white')

# Plot 4: Annual Return vs Volatility
ax4 = axes[1, 1]
annual_ret = results_df['annual_return'] * 100
annual_vol = results_df['annual_volatility'] * 100

for i, strategy in enumerate(results_df.index):
    ax4.scatter(annual_vol.loc[strategy], annual_ret.loc[strategy], 
               s=300, color=colors[i], alpha=0.8, edgecolor='black', linewidth=2, label=strategy)
    ax4.annotate(strategy, (annual_vol.loc[strategy], annual_ret.loc[strategy]),
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

ax4.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
ax4.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('Labs-2/results/cp_portfolio/portfolio_performance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: Labs/results/cp_portfolio/portfolio_performance.png")

# ============================================================================
# Figure 2: Allocation Dynamics
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Allocation over time
ax1 = axes[0]
for i, col in enumerate(['Point Prediction', 'CP Conservative']):
    ax1.plot(allocations_df.index, allocations_df[col] * 100, label=col, linewidth=2, color=colors[i+1], alpha=0.8)

ax1.axhline(y=100, color='#2E86AB', linestyle='--', linewidth=2, label='Buy & Hold (100%)', alpha=0.7)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Equity Allocation (%)', fontsize=12, fontweight='bold')
ax1.set_title('Dynamic Allocation Strategies', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 110)

# Plot 2: Allocation distribution
ax2 = axes[1]
allocation_bins = [0, 0.25, 0.65, 1.05]
allocation_labels = ['Defensive\n(0-25%)', 'Balanced\n(25-65%)', 'Aggressive\n(65-100%)']

for i, col in enumerate(['Point Prediction', 'CP Conservative']):
    alloc_dist = pd.cut(allocations_df[col], bins=allocation_bins, labels=allocation_labels).value_counts()
    alloc_pct = alloc_dist / len(allocations_df) * 100
    
    x_pos = np.arange(len(allocation_labels)) + i * 0.35
    ax2.bar(x_pos, alloc_pct.reindex(allocation_labels, fill_value=0), 
           width=0.35, label=col, color=colors[i+1], alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xticks(np.arange(len(allocation_labels)) + 0.175)
ax2.set_xticklabels(allocation_labels)
ax2.set_ylabel('Percentage of Days (%)', fontsize=12, fontweight='bold')
ax2.set_title('Allocation Distribution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Labs-2/results/cp_portfolio/allocation_dynamics.png', dpi=300, bbox_inches='tight')
print("✅ Saved: Labs/results/cp_portfolio/allocation_dynamics.png")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

print("\n1. Performance Metrics:")
print(results_df.to_string())

print("\n2. Sharpe Ratio Improvement:")
bh_sharpe = results_df.loc['Buy & Hold', 'sharpe_ratio']
for strategy in results_df.index:
    if strategy != 'Buy & Hold':
        improvement = (results_df.loc[strategy, 'sharpe_ratio'] / bh_sharpe - 1) * 100
        print(f"  {strategy}: {improvement:+.1f}%")

print("\n3. Risk Reduction (Max Drawdown):")
bh_dd = results_df.loc['Buy & Hold', 'max_drawdown']
for strategy in results_df.index:
    if strategy != 'Buy & Hold':
        reduction = (1 - results_df.loc[strategy, 'max_drawdown'] / bh_dd) * 100
        print(f"  {strategy}: {reduction:.1f}% lower")

print("\n4. Key Findings:")
print(f"  ✅ Point Prediction Sharpe: {results_df.loc['Point Prediction', 'sharpe_ratio']:.3f}")
print(f"  ✅ CP Conservative Sharpe: {results_df.loc['CP Conservative', 'sharpe_ratio']:.3f}")
print(f"  ✅ Buy & Hold Sharpe: {bh_sharpe:.3f}")
print(f"  ✅ Point vs BH: {(results_df.loc['Point Prediction', 'sharpe_ratio']/bh_sharpe - 1)*100:+.1f}%")
print(f"  ✅ CP vs BH: {(results_df.loc['CP Conservative', 'sharpe_ratio']/bh_sharpe - 1)*100:+.1f}%")

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)

