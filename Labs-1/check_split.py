import sys
sys.path.append('Labs')
from data_loader import load_complete_dataset
import pandas as pd

print('='*80)
print('VERIFYING TRAIN/CAL/TEST SPLIT')
print('='*80)

df = load_complete_dataset()

# Define split dates (same as in experiments)
train_end = '2020-12-31'
cal_end = '2021-12-31'

train = df[df.index <= train_end]
cal = df[(df.index > train_end) & (df.index <= cal_end)]
test = df[df.index > cal_end]

print(f'\nTrain:')
print(f'  Period: {train.index[0].date()} to {train.index[-1].date()}')
print(f'  Observations: {len(train)}')
print(f'  Percentage: {len(train)/len(df)*100:.1f}%')

print(f'\nCalibration:')
print(f'  Period: {cal.index[0].date()} to {cal.index[-1].date()}')
print(f'  Observations: {len(cal)}')
print(f'  Percentage: {len(cal)/len(df)*100:.1f}%')

print(f'\nTest:')
print(f'  Period: {test.index[0].date()} to {test.index[-1].date()}')
print(f'  Observations: {len(test)}')
print(f'  Percentage: {len(test)/len(df)*100:.1f}%')

print(f'\nTotal: {len(train) + len(cal) + len(test)} (should be {len(df)})')

# Verify
total = len(train) + len(cal) + len(test)
if total == len(df):
    print('Verification: MATCH')
else:
    print(f'Verification: MISMATCH - {total} vs {len(df)}')

# Compare with paper
print('\n' + '='*80)
print('COMPARISON WITH PAPER')
print('='*80)
print('Paper claims:')
print('  Train: 2,537 observations (71.4%)')
print('  Calibration: 251 observations (7.1%)')
print('  Test: 767 observations (21.6%)')
print('  Total: 3,555 observations')

print('\nActual:')
print(f'  Train: {len(train)} observations ({len(train)/len(df)*100:.1f}%)')
print(f'  Calibration: {len(cal)} observations ({len(cal)/len(df)*100:.1f}%)')
print(f'  Test: {len(test)} observations ({len(test)/len(df)*100:.1f}%)')
print(f'  Total: {len(df)} observations')

print('\nStatus:')
print(f'  Train: {"MATCH" if len(train) == 2537 else "MISMATCH"}')
print(f'  Calibration: {"MATCH" if len(cal) == 251 else "MISMATCH"}')
print(f'  Test: {"MATCH" if len(test) == 767 else "MISMATCH"}')
print(f'  Total: {"MATCH" if len(df) == 3555 else "MISMATCH"}')

