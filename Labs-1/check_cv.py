import pandas as pd
import numpy as np

cv = pd.read_csv('Labs/results/statistical_tests/cv_results.csv')
acc = cv['accuracy'].values
cov = cv['coverage'].values

print('='*80)
print('CV FOLD VALUES CHECK')
print('='*80)

print('\nACCURACY:')
for i, a in enumerate(acc, 1):
    print(f'  Fold {i}: {a*100:.2f}%')
print(f'\n  Mean: {acc.mean()*100:.2f}%')
print(f'  Std (sample): {acc.std(ddof=1)*100:.2f}%')

print('\nCOVERAGE:')
for i, c in enumerate(cov, 1):
    print(f'  Fold {i}: {c*100:.2f}%')
print(f'\n  Mean: {cov.mean()*100:.2f}%')
print(f'  Std (sample): {cov.std(ddof=1)*100:.2f}%')

print('\n' + '='*80)
print('PAPER TABLE 8 VALUES:')
print('='*80)
print('Fold 1: 97.60% (Accuracy), 95.62% (Coverage)')
print('Fold 2: 98.17% (Accuracy), 95.90% (Coverage)')
print('Fold 3: 97.32% (Accuracy), 95.34% (Coverage)')
print('Fold 4: 97.60% (Accuracy), 95.48% (Coverage)')
print('Fold 5: 98.03% (Accuracy), 95.31% (Coverage)')
print('Mean: 97.74%, Std: 0.56%')

print('\n' + '='*80)
print('COMPARISON:')
print('='*80)
paper_acc = np.array([97.60, 98.17, 97.32, 97.60, 98.03])
print(f'Paper Mean: {paper_acc.mean():.2f}%')
print(f'Paper Std: {paper_acc.std(ddof=1):.2f}%')
print(f'CSV Mean: {acc.mean()*100:.2f}%')
print(f'CSV Std: {acc.std(ddof=1)*100:.2f}%')

