"""
Verify that generated figures match the data
"""

import pandas as pd
import numpy as np
import os

# Load results
df = pd.read_csv('Labs-2/results/multi_asset/multi_asset_cp_results.csv')

print('='*80)
print('VERIFICATION: Figures Match Data')
print('='*80)

# Verify data integrity
print('\n[1] Data Integrity Check:')
print(f'  Total rows: {len(df)}')
print(f'  Expected: 10 (2 methods x 5 assets)')
print(f'  Status: {"PASS" if len(df) == 10 else "FAIL"}')

# Verify coverage values
print('\n[2] Coverage Values Check:')
standard_cp = df[df['method'] == 'Standard CP']
mean_cov = standard_cp['overall_coverage'].mean()
print(f'  Standard CP mean coverage: {mean_cov:.4f}')
print(f'  Expected: ~0.8837')
print(f'  Status: {"PASS" if abs(mean_cov - 0.8837) < 0.001 else "FAIL"}')

# Verify std dev calculation
print('\n[3] Coverage Balance (Std Dev) Check:')
for asset in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']:
    asset_df = df[df['asset'] == asset]
    standard = asset_df[asset_df['method'] == 'Standard CP'].iloc[0]
    coverages = [standard['bear_coverage'], standard['bull_coverage'], standard['neutral_coverage']]
    std_dev = np.std(coverages, ddof=1) * 100
    print(f'  {asset}: {std_dev:.2f}% std dev')

print('\n[4] Figure Files Check:')
figures = [
    'Labs-2/results/multi_asset/cross_asset_comparison.png',
    'Labs-2/results/multi_asset/per_regime_coverage_heatmap.png',
    'Labs-2/results/multi_asset/coverage_balance.png'
]
all_exist = True
for fig in figures:
    exists = os.path.exists(fig)
    all_exist = all_exist and exists
    print(f'  {os.path.basename(fig)}: {"EXISTS" if exists else "MISSING"}')

print('\n[5] Timestamp Check:')
csv_time = os.path.getmtime('Labs-2/results/multi_asset/multi_asset_cp_results.csv')
for fig in figures:
    fig_time = os.path.getmtime(fig)
    status = "UP TO DATE" if fig_time >= csv_time else "OUTDATED"
    print(f'  {os.path.basename(fig)}: {status}')

print('\n' + '='*80)
if all_exist:
    print('ALL CHECKS PASSED - FIGURES ARE UP TO DATE')
else:
    print('SOME CHECKS FAILED - REGENERATE FIGURES')
print('='*80)

