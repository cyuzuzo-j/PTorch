import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('memory_benchmark_results.csv')
print("MN correlation:", pearsonr(df['M'] * df['N'], df['ptorch_run_mb'])[0])
print("NK correlation:", pearsonr(df['N'] * df['K'], df['ptorch_run_mb'])[0])
print("MK correlation:", pearsonr(df['M'] * df['K'], df['ptorch_run_mb'])[0])
print("MNK correlation:", pearsonr(df['M'] * df['N'] * df['K'], df['ptorch_run_mb'])[0])
