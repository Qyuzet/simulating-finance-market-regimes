"""
Generate figures for Enhanced Conformal Prediction results
Answers to Open Research Questions Q2 and Q3
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load results
results_df = pd.read_csv('Labs-2/results/enhanced_cp/enhanced_cp_results.csv')

print("Enhanced CP Results:")
print(results_df)

# ============================================================================
# FIGURE 1: Class-Conditional CP - Coverage Comparison (Q2)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Coverage by Regime
methods = ['Standard CP', 'Class-Conditional CP']
regimes = ['Bear', 'Bull', 'Neutral']
colors = ['#d62728', '#2ca02c', '#1f77b4']  # Red, Green, Blue

standard_coverage = [
    results_df[results_df['method'] == 'Standard CP']['bear_coverage'].values[0] * 100,
    results_df[results_df['method'] == 'Standard CP']['bull_coverage'].values[0] * 100,
    results_df[results_df['method'] == 'Standard CP']['neutral_coverage'].values[0] * 100
]

class_cond_coverage = [
    results_df[results_df['method'] == 'Class-Conditional CP']['bear_coverage'].values[0] * 100,
    results_df[results_df['method'] == 'Class-Conditional CP']['bull_coverage'].values[0] * 100,
    results_df[results_df['method'] == 'Class-Conditional CP']['neutral_coverage'].values[0] * 100
]

x = np.arange(len(regimes))
width = 0.35

bars1 = axes[0].bar(x - width/2, standard_coverage, width, label='Standard CP', 
                     color=colors, alpha=0.6, edgecolor='black')
bars2 = axes[0].bar(x + width/2, class_cond_coverage, width, label='Class-Conditional CP',
                     color=colors, alpha=1.0, edgecolor='black')

axes[0].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
axes[0].set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Regime', fontsize=12, fontweight='bold')
axes[0].set_title('(a) Coverage by Regime', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(regimes)
axes[0].legend(loc='lower right')
axes[0].set_ylim([80, 100])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Panel B: Coverage Standard Deviation (use sample std dev with ddof=1)
std_standard = np.std(standard_coverage, ddof=1)
std_class_cond = np.std(class_cond_coverage, ddof=1)

bars = axes[1].bar(['Standard CP', 'Class-Conditional CP'],
                    [std_standard, std_class_cond],
                    color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')

axes[1].set_ylabel('Coverage Std Dev (%)', fontsize=12, fontweight='bold')
axes[1].set_title('(b) Coverage Balance Across Regimes', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Calculate reduction percentage
reduction_pct = ((std_standard - std_class_cond) / std_standard) * 100

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    if i == 1:  # Class-Conditional CP
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.2f}%\n({reduction_pct:.1f}% reduction)',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.2f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Labs-2/results/enhanced_cp/class_conditional_cp_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: Labs/results/enhanced_cp/class_conditional_cp_comparison.png")
plt.close()

# ============================================================================
# FIGURE 2: Adaptive CP - Window Size Comparison (Q3)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Coverage by Window Size
# Filter only pure Adaptive CP (exclude Hybrid)
adaptive_results = results_df[results_df['method'].str.contains('Adaptive CP \(w=')]
window_sizes = adaptive_results['window_size'].values
overall_coverage = adaptive_results['overall_coverage'].values * 100

axes[0].plot(window_sizes, overall_coverage, marker='o', linewidth=2, markersize=10,
             color='#1f77b4', label='Overall Coverage')
axes[0].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
axes[0].set_xlabel('Window Size (days)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
axes[0].set_title('(a) Coverage vs Window Size', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim([88, 93])

# Add value labels
for x, y in zip(window_sizes, overall_coverage):
    axes[0].text(x, y + 0.3, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)

# Panel B: Coverage by Regime for Different Windows
regimes = ['Bear', 'Bull', 'Neutral']
window_labels = ['126d\n(6mo)', '252d\n(1yr)', '504d\n(2yr)']

bear_coverage = adaptive_results['bear_coverage'].values * 100
bull_coverage = adaptive_results['bull_coverage'].values * 100
neutral_coverage = adaptive_results['neutral_coverage'].values * 100

x = np.arange(len(window_labels))
width = 0.25

bars1 = axes[1].bar(x - width, bear_coverage, width, label='Bear', color='#d62728', alpha=0.7, edgecolor='black')
bars2 = axes[1].bar(x, bull_coverage, width, label='Bull', color='#2ca02c', alpha=0.7, edgecolor='black')
bars3 = axes[1].bar(x + width, neutral_coverage, width, label='Neutral', color='#1f77b4', alpha=0.7, edgecolor='black')

axes[1].axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1].set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Window Size', fontsize=12, fontweight='bold')
axes[1].set_title('(b) Regime Coverage by Window Size', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(window_labels)
axes[1].legend()
axes[1].set_ylim([84, 96])  # Adjusted range to better show differences
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('Labs-2/results/enhanced_cp/adaptive_cp_window_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: Labs/results/enhanced_cp/adaptive_cp_window_comparison.png")
plt.close()

print("\n✅ All figures generated successfully!")
print("\nGenerated files:")
print("1. Labs/results/enhanced_cp/class_conditional_cp_comparison.png")
print("2. Labs/results/enhanced_cp/adaptive_cp_window_comparison.png")

