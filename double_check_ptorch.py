import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

df = pd.read_csv('memory_benchmark_results.csv')

df['MNK'] = df['M'] * df['N'] * df['K']
df['MN'] = df['M'] * df['N']

# Calculate correlations
corr_mnk, _ = pearsonr(df['MNK'], df['ptorch_run_mb'])
corr_mn, _ = pearsonr(df['MN'], df['ptorch_run_mb'])

print(f"Pearson Correlation (ptorch vs MNK): {corr_mnk:.4f}")
print(f"Pearson Correlation (ptorch vs MN):  {corr_mn:.4f}")

# Compare pjax vs MNK to see the difference clearly
corr_pjax_mnk, _ = pearsonr(df['MNK'], df['pjax_run_mb'])
print(f"Pearson Correlation (pjax vs MNK):   {corr_pjax_mnk:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: ptorch vs MNK
sns.scatterplot(
    data=df, 
    x='MNK', 
    y='ptorch_run_mb', 
    hue='K',
    palette='viridis',
    ax=axes[0],
    s=100
)
axes[0].set_title(f'ptorch Memory vs M*N*K (Corr: {corr_mnk:.2f})')
axes[0].set_xlabel('M * N * K')
axes[0].set_ylabel('Memory (MB)')

# Plot 2: ptorch vs MN
sns.scatterplot(
    data=df, 
    x='MN', 
    y='ptorch_run_mb', 
    hue='K',
    palette='viridis',
    ax=axes[1],
    s=100
)
axes[1].set_title(f'ptorch Memory vs M*N (Corr: {corr_mn:.2f})')
axes[1].set_xlabel('M * N')
axes[1].set_ylabel('Memory (MB)')

plt.suptitle("Checking ptorch memory complexity (Is it O(MNK) or O(MN)?)", y=1.05)
plt.tight_layout()

plt.savefig('ptorch_double_check.png', bbox_inches='tight', dpi=300)
print("Saved plot to ptorch_double_check.png")
