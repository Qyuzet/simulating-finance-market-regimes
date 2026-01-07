import pandas as pd
import numpy as np

print('='*80)
print('ABSTRACT VERIFICATION')
print('='*80)

# Load data
enhanced = pd.read_csv('Labs/results/enhanced_cp/enhanced_cp_results.csv')
portfolio = pd.read_csv('Labs/results/cp_portfolio/backtest_results.csv', index_col=0)
multi_asset = pd.read_csv('Labs/results/multi_asset/multi_asset_cp_results.csv')

# Extract values
std_cp = enhanced[enhanced['method'] == 'Standard CP'].iloc[0]
hybrid = enhanced[enhanced['method'].str.contains('Hybrid')].iloc[0]

print('\nABSTRACT CLAIMS:')
print('-'*80)

# Claim 1: Standard CP coverage
print(f'1. "91.09% coverage for 90% confidence intervals"')
print(f'   Actual: {std_cp["overall_coverage"]*100:.2f}%')
print(f'   Status: {"✅ CORRECT" if abs(std_cp["overall_coverage"]*100 - 91.09) < 0.01 else "❌ WRONG"}')

# Claim 2: Hybrid CP coverage
print(f'\n2. "91.80% coverage with 0.95% standard deviation"')
bear = hybrid['bear_coverage'] * 100
bull = hybrid['bull_coverage'] * 100
neutral = hybrid['neutral_coverage'] * 100
std_dev = np.std([bear, bull, neutral], ddof=1)
print(f'   Actual Coverage: {hybrid["overall_coverage"]*100:.2f}%')
print(f'   Actual Std Dev: {std_dev:.2f}%')
print(f'   Status: {"✅ CORRECT" if abs(hybrid["overall_coverage"]*100 - 91.80) < 0.01 and abs(std_dev - 0.95) < 0.01 else "❌ WRONG"}')

# Claim 3: 79.3% reduction
baseline_std = np.std([std_cp['bear_coverage']*100, std_cp['bull_coverage']*100, std_cp['neutral_coverage']*100], ddof=1)
reduction = (baseline_std - std_dev) / baseline_std * 100
print(f'\n3. "79.3% reduction from baseline"')
print(f'   Baseline Std: {baseline_std:.2f}%')
print(f'   Hybrid Std: {std_dev:.2f}%')
print(f'   Actual Reduction: {reduction:.1f}%')
print(f'   Status: {"✅ CORRECT" if abs(reduction - 79.3) < 0.5 else "❌ WRONG"}')

# Claim 4: Portfolio performance
buy_hold = portfolio.loc['Buy & Hold']
cp_cons = portfolio.loc['CP Conservative']
sharpe_improvement = (cp_cons['sharpe_ratio'] - buy_hold['sharpe_ratio']) / buy_hold['sharpe_ratio'] * 100
dd_reduction = (buy_hold['max_drawdown'] - cp_cons['max_drawdown']) / buy_hold['max_drawdown'] * 100

print(f'\n4. "30.3% higher Sharpe ratio"')
print(f'   Buy & Hold Sharpe: {buy_hold["sharpe_ratio"]:.3f}')
print(f'   CP Conservative Sharpe: {cp_cons["sharpe_ratio"]:.3f}')
print(f'   Actual Improvement: {sharpe_improvement:.1f}%')
print(f'   Status: {"✅ CORRECT" if abs(sharpe_improvement - 30.3) < 0.5 else "❌ WRONG"}')

print(f'\n5. "59.9% lower maximum drawdown"')
print(f'   Buy & Hold DD: {buy_hold["max_drawdown"]*100:.1f}%')
print(f'   CP Conservative DD: {cp_cons["max_drawdown"]*100:.1f}%')
print(f'   Actual Reduction: {dd_reduction:.1f}%')
print(f'   Status: {"✅ CORRECT" if abs(dd_reduction - 59.9) < 0.5 else "❌ WRONG"}')

# Claim 5: Multi-asset mean coverage
std_mean = multi_asset[multi_asset['method'] == 'Standard CP']['overall_coverage'].mean() * 100
print(f'\n6. "88.4% mean coverage" (multi-asset)')
print(f'   Actual: {std_mean:.1f}%')
print(f'   Status: {"✅ CORRECT" if abs(std_mean - 88.4) < 0.5 else "❌ WRONG"}')

print('\n' + '='*80)
print('ABSTRACT VERIFICATION COMPLETE')
print('='*80)

