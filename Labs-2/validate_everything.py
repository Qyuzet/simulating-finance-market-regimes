"""
COMPREHENSIVE VALIDATION SCRIPT
Check everything: files, data integrity, formatting, bugs
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("🔍 COMPREHENSIVE VALIDATION CHECK")
print("="*80)

errors = []
warnings = []
success = []

# ============================================================================
# 1. CHECK ALL EXPECTED FILES EXIST
# ============================================================================
print("\n[1] CHECKING FILE EXISTENCE...")
print("-"*80)

expected_files = {
    'Data Cleaning': [
        'Labs-2/results/data_cleaning/descriptive_stats.csv',
        'Labs-2/results/data_cleaning/outlier_summary.csv',
        'Labs-2/results/data_cleaning/dataset_info.txt'
    ],
    'Multi-Market': [
        'Labs-2/results/multi_market/results.csv'
    ],
    'Baselines': [
        'Labs-2/results/baselines/baseline_results.csv'
    ],
    'TimeGAN': [
        'Labs-2/results/timegan_synthetic.npy'
    ],
    'Visualizations': [
        'Labs-2/results/figures/multi_market_analysis.png',
        'Labs-2/results/figures/baseline_comparison.png'
    ],
    'Summary': [
        'Labs-2/results/experiment_summary_table.csv',
        'Labs-2/results/FINAL_REPORT.txt',
        'Labs-2/COMPLETE_RESULTS_SUMMARY.md'
    ]
}

for category, files in expected_files.items():
    print(f"\n{category}:")
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size:,} bytes)")
            success.append(f"{category}: {file} exists")
        else:
            print(f"  ❌ MISSING: {file}")
            errors.append(f"Missing file: {file}")

# ============================================================================
# 2. VALIDATE DATA INTEGRITY
# ============================================================================
print("\n[2] VALIDATING DATA INTEGRITY...")
print("-"*80)

# Check descriptive stats
try:
    desc_stats = pd.read_csv('Labs-2/results/data_cleaning/descriptive_stats.csv', index_col=0)
    print(f"\n✅ Descriptive Stats: {desc_stats.shape}")
    
    # Check for inf/nan in stats
    if desc_stats.isin([np.inf, -np.inf]).any().any():
        warnings.append("Descriptive stats contains inf values (expected for volume_change)")
    if desc_stats.isna().any().any():
        warnings.append("Descriptive stats contains NaN values (expected for volume_change std)")
    
    success.append("Descriptive stats loaded successfully")
except Exception as e:
    errors.append(f"Failed to load descriptive stats: {e}")

# Check outlier summary
try:
    outliers = pd.read_csv('Labs-2/results/data_cleaning/outlier_summary.csv')
    print(f"✅ Outlier Summary: {outliers.shape}")
    
    # Validate columns
    required_cols = ['Feature', 'Outliers', 'Percentage', 'Lower Bound', 'Upper Bound']
    if not all(col in outliers.columns for col in required_cols):
        errors.append(f"Outlier summary missing columns: {required_cols}")
    else:
        success.append("Outlier summary has all required columns")
        
except Exception as e:
    errors.append(f"Failed to load outlier summary: {e}")

# Check multi-market results
try:
    multi_market = pd.read_csv('Labs-2/results/multi_market/results.csv')
    print(f"✅ Multi-Market Results: {multi_market.shape}")
    
    # Validate we have 6 markets x 3 regimes = 18 rows
    if len(multi_market) != 18:
        warnings.append(f"Expected 18 rows (6 markets x 3 regimes), got {len(multi_market)}")
    else:
        success.append("Multi-market has correct number of rows (18)")
    
    # Check for NaN
    if multi_market.isna().any().any():
        errors.append("Multi-market results contain NaN values")
    else:
        success.append("Multi-market results have no NaN values")
        
    # Validate percentages sum to ~100% per market
    for market in multi_market['market'].unique():
        market_data = multi_market[multi_market['market'] == market]
        total_pct = market_data['percentage'].sum()
        if not (99.9 <= total_pct <= 100.1):
            warnings.append(f"{market}: percentages sum to {total_pct:.2f}% (expected ~100%)")
        else:
            success.append(f"{market}: percentages sum correctly ({total_pct:.2f}%)")
            
except Exception as e:
    errors.append(f"Failed to load multi-market results: {e}")

# Check baseline results
try:
    baselines = pd.read_csv('Labs-2/results/baselines/baseline_results.csv')
    print(f"✅ Baseline Results: {baselines.shape}")
    
    # Validate we have 4 models
    if len(baselines) != 4:
        errors.append(f"Expected 4 baseline models, got {len(baselines)}")
    else:
        success.append("Baseline results have 4 models")
    
    # Check accuracy bounds
    if not all((0 <= baselines['accuracy']) & (baselines['accuracy'] <= 1)):
        errors.append("Baseline accuracy values out of bounds [0, 1]")
    else:
        success.append("Baseline accuracy values in valid range")
        
    # Check for NaN
    if baselines.isna().any().any():
        errors.append("Baseline results contain NaN values")
    else:
        success.append("Baseline results have no NaN values")
        
except Exception as e:
    errors.append(f"Failed to load baseline results: {e}")

# Check TimeGAN synthetic data
try:
    synthetic = np.load('Labs-2/results/timegan_synthetic.npy')
    print(f"✅ TimeGAN Synthetic Data: {synthetic.shape}")
    
    # Validate shape (should be 1000 sequences x 30 timesteps x 5 features)
    if synthetic.shape != (1000, 30, 5):
        warnings.append(f"Expected shape (1000, 30, 5), got {synthetic.shape}")
    else:
        success.append("TimeGAN synthetic data has correct shape")
    
    # Check for NaN/inf
    if np.isnan(synthetic).any():
        errors.append("TimeGAN synthetic data contains NaN values")
    else:
        success.append("TimeGAN synthetic data has no NaN values")
        
    if np.isinf(synthetic).any():
        errors.append("TimeGAN synthetic data contains inf values")
    else:
        success.append("TimeGAN synthetic data has no inf values")
        
except Exception as e:
    errors.append(f"Failed to load TimeGAN synthetic data: {e}")

# ============================================================================
# 3. CHECK VISUALIZATIONS
# ============================================================================
print("\n[3] CHECKING VISUALIZATIONS...")
print("-"*80)

viz_files = [
    'Labs-2/results/figures/multi_market_analysis.png',
    'Labs-2/results/figures/baseline_comparison.png'
]

for viz in viz_files:
    if os.path.exists(viz):
        size = os.path.getsize(viz)
        if size < 1000:  # Less than 1KB is suspicious
            warnings.append(f"{viz} is very small ({size} bytes)")
        else:
            print(f"✅ {viz} ({size:,} bytes)")
            success.append(f"Visualization {viz} exists and has reasonable size")
    else:
        errors.append(f"Missing visualization: {viz}")

# ============================================================================
# 4. VALIDATE SUMMARY FILES
# ============================================================================
print("\n[4] VALIDATING SUMMARY FILES...")
print("-"*80)

# Check experiment summary table
try:
    summary = pd.read_csv('Labs-2/results/experiment_summary_table.csv')
    print(f"✅ Experiment Summary Table: {summary.shape}")
    
    if len(summary) != 6:
        errors.append(f"Expected 6 experiments in summary, got {len(summary)}")
    else:
        success.append("Experiment summary has 6 experiments")
        
except Exception as e:
    errors.append(f"Failed to load experiment summary: {e}")

# Check final report
try:
    with open('Labs-2/results/FINAL_REPORT.txt', 'r', encoding='utf-8') as f:
        report = f.read()
    
    if len(report) < 100:
        errors.append("Final report is too short")
    else:
        print(f"✅ Final Report: {len(report)} characters")
        success.append("Final report exists and has content")
        
except Exception as e:
    errors.append(f"Failed to load final report: {e}")

# ============================================================================
# 5. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 VALIDATION SUMMARY")
print("="*80)

print(f"\n✅ SUCCESSES: {len(success)}")
for s in success[:5]:  # Show first 5
    print(f"  • {s}")
if len(success) > 5:
    print(f"  ... and {len(success) - 5} more")

print(f"\n⚠️  WARNINGS: {len(warnings)}")
for w in warnings:
    print(f"  • {w}")

print(f"\n❌ ERRORS: {len(errors)}")
for e in errors:
    print(f"  • {e}")

# Save validation report
validation_report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'successes': len(success),
    'warnings': len(warnings),
    'errors': len(errors),
    'success_list': success,
    'warning_list': warnings,
    'error_list': errors
}

with open('Labs-2/results/VALIDATION_REPORT.json', 'w') as f:
    json.dump(validation_report, f, indent=2)

print("\n✅ Validation report saved to: Labs/results/VALIDATION_REPORT.json")

# Final verdict
print("\n" + "="*80)
if len(errors) == 0:
    print("✅ ALL CHECKS PASSED! No critical errors found.")
    print("="*80)
    exit(0)
else:
    print("❌ VALIDATION FAILED! Please fix errors above.")
    print("="*80)
    exit(1)

