"""
VISUALIZE CONFORMAL PREDICTION RESULTS
Create publication-quality figures for paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

print("="*80)
print("📊 VISUALIZING CONFORMAL PREDICTION RESULTS")
print("="*80)

# Load results
coverage_df = pd.read_csv('Labs-2/results/conformal_prediction/coverage_results.csv')
pred_sets_df = pd.read_csv('Labs-2/results/conformal_prediction/prediction_sets.csv')

print(f"\n✅ Loaded results:")
print(f"  Coverage results: {len(coverage_df)} rows")
print(f"  Prediction sets: {len(pred_sets_df)} rows")

# ============================================================================
# Figure 1: Coverage vs Expected Coverage
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Coverage comparison
ax1 = axes[0, 0]
x = coverage_df['expected_coverage'] * 100
y = coverage_df['empirical_coverage'] * 100

ax1.plot(x, y, 'o-', markersize=10, linewidth=2, label='Empirical Coverage', color='#2E86AB')
ax1.plot([75, 100], [75, 100], '--', linewidth=2, label='Perfect Coverage', color='#A23B72', alpha=0.7)
ax1.set_xlabel('Expected Coverage (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Empirical Coverage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Conformal Prediction Coverage Guarantee', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(75, 100)
ax1.set_ylim(75, 100)

# Add annotations
for i, row in coverage_df.iterrows():
    exp = row['expected_coverage'] * 100
    emp = row['empirical_coverage'] * 100
    ax1.annotate(f"{emp:.1f}%", (exp, emp), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=10, fontweight='bold')

# Plot 2: Average set size
ax2 = axes[0, 1]
x = coverage_df['expected_coverage'] * 100
y = coverage_df['avg_set_size']

ax2.bar(x, y, width=5, color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Expected Coverage (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Set Size', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Set Size vs Coverage', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.2)

# Add value labels
for i, row in coverage_df.iterrows():
    exp = row['expected_coverage'] * 100
    size = row['avg_set_size']
    ax2.text(exp, size + 0.05, f"{size:.2f}", 
            ha='center', fontsize=11, fontweight='bold')

# Plot 3: Coverage by regime
ax3 = axes[1, 0]
regime_names = ['Bear', 'Bull', 'Neutral']
regime_coverage = []
for regime in range(3):
    regime_data = pred_sets_df[pred_sets_df['true_regime'] == regime]
    coverage = regime_data['covered'].mean()
    regime_coverage.append(coverage * 100)

colors = ['#E63946', '#06A77D', '#457B9D']
bars = ax3.bar(regime_names, regime_coverage, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=90, color='black', linestyle='--', linewidth=2, label='Target (90%)')
ax3.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
ax3.set_title('Coverage by Regime (α=0.10)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 100)

# Add value labels
for bar, val in zip(bars, regime_coverage):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Set size distribution
ax4 = axes[1, 1]
set_sizes = pred_sets_df['set_size'].value_counts().sort_index()
total = len(pred_sets_df)

sizes = set_sizes.index.tolist()
counts = set_sizes.values.tolist()
percentages = [c/total*100 for c in counts]

bars = ax4.bar(sizes, percentages, color='#C9ADA7', alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Prediction Set Size', fontsize=12, fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of Prediction Set Sizes (α=0.10)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(range(4))

# Add value labels
for bar, val in zip(bars, percentages):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Labs-2/results/conformal_prediction/conformal_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: Labs/results/conformal_prediction/conformal_analysis.png")

# ============================================================================
# Figure 2: Comparison with Point Predictions
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Calculate metrics
point_acc = (pred_sets_df['true_regime'] == pred_sets_df['point_prediction']).mean() * 100
cp_coverage = pred_sets_df['covered'].mean() * 100
avg_set_size = pred_sets_df['set_size'].mean()

metrics = ['Point Accuracy', 'CP Coverage (90%)', 'Avg Set Size']
values = [point_acc, cp_coverage, avg_set_size]
colors_bar = ['#2E86AB', '#06A77D', '#F18F01']

bars = ax.bar(metrics, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Value (%)', fontsize=12, fontweight='bold')
ax.set_title('Conformal Prediction vs Point Prediction', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 110)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
           f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('Labs-2/results/conformal_prediction/comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: Labs/results/conformal_prediction/comparison.png")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

print("\n1. Coverage Results:")
print(coverage_df.to_string(index=False))

print("\n2. Regime-Specific Coverage (α=0.10):")
for regime, name in enumerate(['Bear', 'Bull', 'Neutral']):
    regime_data = pred_sets_df[pred_sets_df['true_regime'] == regime]
    coverage = regime_data['covered'].mean()
    count = len(regime_data)
    print(f"  {name}: {coverage:.4f} ({coverage*100:.2f}%) - {count} samples")

print("\n3. Set Size Distribution (α=0.10):")
set_size_dist = pred_sets_df['set_size'].value_counts().sort_index()
for size, count in set_size_dist.items():
    pct = count / len(pred_sets_df) * 100
    print(f"  Size {size}: {count} ({pct:.1f}%)")

print("\n4. Key Findings:")
print(f"  ✅ Point Accuracy: {point_acc:.2f}%")
print(f"  ✅ 90% Coverage Achieved: {cp_coverage:.2f}%")
print(f"  ✅ Average Set Size: {avg_set_size:.2f}")
print(f"  ✅ Efficiency: {(1/avg_set_size)*100:.1f}% (lower is better)")

# Calculate efficiency
efficiency = pred_sets_df['set_size'].apply(lambda x: 1/x if x > 0 else 0).mean()
print(f"  ✅ Prediction Efficiency: {efficiency:.2%}")

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)

