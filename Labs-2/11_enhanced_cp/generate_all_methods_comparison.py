"""
Generate comprehensive comparison figures for all CP methods

Figure 11: All 6 CP methods comparison (coverage + set size)
Figure 12: Transaction cost sensitivity analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# FIGURE 11: All 6 CP Methods Comparison
# ============================================================================

print("="*80)
print("GENERATING FIGURE 11: ALL CP METHODS COMPARISON")
print("="*80)

# Load results
results_df = pd.read_csv('Labs-2/results/enhanced_cp/enhanced_cp_results.csv')

# Create figure with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): Coverage by Regime
methods = results_df['method'].values
bear_cov = results_df['bear_coverage'].values * 100
bull_cov = results_df['bull_coverage'].values * 100
neutral_cov = results_df['neutral_coverage'].values * 100

x = np.arange(len(methods))
width = 0.25

bars1 = ax1.bar(x - width, bear_cov, width, label='Bear', color='#d62728', alpha=0.8)
bars2 = ax1.bar(x, bull_cov, width, label='Bull', color='#2ca02c', alpha=0.8)
bars3 = ax1.bar(x + width, neutral_cov, width, label='Neutral', color='#1f77b4', alpha=0.8)

# Add target line
ax1.axhline(y=90, color='black', linestyle='--', linewidth=1.5, label='Target (90%)', alpha=0.7)

ax1.set_xlabel('Method', fontsize=11, fontweight='bold')
ax1.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Per-Regime Coverage Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' CP', '').replace('Adaptive+ClassCond', 'Hybrid') for m in methods], 
                     rotation=45, ha='right', fontsize=9)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim([84, 96])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=7)

# Panel (b): Overall Coverage vs Coverage Std Dev
overall_cov = results_df['overall_coverage'].values * 100
coverage_std = []
for _, row in results_df.iterrows():
    std = np.std([row['bear_coverage'], row['bull_coverage'], row['neutral_coverage']]) * 100
    coverage_std.append(std)

# Color by method type
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#2ca02c', '#2ca02c', '#d62728']
method_labels = [m.replace(' CP', '').replace('Adaptive+ClassCond', 'Hybrid').replace(' (w=126)', '').replace(' (w=252)', '').replace(' (w=504)', '') 
                 for m in methods]

scatter = ax2.scatter(coverage_std, overall_cov, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add labels
for i, label in enumerate(method_labels):
    ax2.annotate(label, (coverage_std[i], overall_cov[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

# Add target line
ax2.axhline(y=90, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (90%)')

# Add ideal region (high coverage, low std)
ax2.axvspan(0, 1, ymin=0.5, ymax=1, alpha=0.1, color='green', label='Ideal Region')

ax2.set_xlabel('Coverage Std Dev (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Overall Coverage (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Coverage vs Balance Trade-off', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.set_xlim([-0.2, 5])
ax2.set_ylim([88, 93])
ax2.grid(alpha=0.3)

plt.tight_layout()

# Save
os.makedirs('Labs-2/results/enhanced_cp', exist_ok=True)
plt.savefig('Labs-2/results/enhanced_cp/all_methods_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: Labs/results/enhanced_cp/all_methods_comparison.png")
plt.close()

# ============================================================================
# FIGURE 12: Transaction Cost Sensitivity Analysis
# ============================================================================

print("\n" + "="*80)
print("GENERATING FIGURE 12: TRANSACTION COST SENSITIVITY")
print("="*80)

# Load transaction cost results
tc_df = pd.read_csv('Labs-2/results/cp_portfolio/transaction_cost_analysis.csv')

# Create figure with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): Sharpe Ratio vs Transaction Cost
pivot_sharpe = tc_df.pivot(index='strategy', columns='transaction_cost', values='sharpe_ratio')

strategies = pivot_sharpe.index.tolist()
tc_levels = ['0.00%', '0.05%', '0.10%', '0.20%']
colors_map = {'Buy & Hold': '#1f77b4', 'Point Prediction': '#ff7f0e', 
              'CP Conservative': '#d62728', 'CP Average': '#2ca02c'}

for strategy in strategies:
    values = [pivot_sharpe.loc[strategy, tc] for tc in tc_levels]
    ax1.plot(tc_levels, values, marker='o', linewidth=2.5, markersize=8, 
            label=strategy, color=colors_map[strategy], alpha=0.8)

ax1.set_xlabel('Transaction Cost', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax1.set_title('(a) Sharpe Ratio Degradation', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_ylim([0.5, 1.5])

# Panel (b): Return vs Transaction Cost
pivot_return = tc_df.pivot(index='strategy', columns='transaction_cost', values='total_return')

for strategy in strategies:
    values = [pivot_return.loc[strategy, tc] * 100 for tc in tc_levels]
    ax2.plot(tc_levels, values, marker='s', linewidth=2.5, markersize=8, 
            label=strategy, color=colors_map[strategy], alpha=0.8)

ax2.set_xlabel('Transaction Cost', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Return (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Total Return Degradation', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()

# Save
plt.savefig('Labs-2/results/cp_portfolio/transaction_cost_sensitivity.png', dpi=300, bbox_inches='tight')
print("✅ Saved: Labs/results/cp_portfolio/transaction_cost_sensitivity.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Figures:")
print("  1. Labs/results/enhanced_cp/all_methods_comparison.png (Figure 11)")
print("  2. Labs/results/cp_portfolio/transaction_cost_sensitivity.png (Figure 12)")
print("\nReady to add to conference paper!")
print("="*80)

