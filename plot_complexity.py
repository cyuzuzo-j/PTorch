import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('memory_benchmark_results.csv')

# Calculate complexity proxies
df['MNK'] = df['M'] * df['N'] * df['K']
df['MN'] = df['M'] * df['N']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: pjax Memory vs O(MNK)
sns.regplot(
    data=df, 
    x='MNK', 
    y='pjax_run_mb', 
    ax=axes[0],
    scatter_kws={'alpha': 0.7},
    line_kws={'color': 'red'}
)
axes[0].set_title('pjax Memory vs M*N*K')
axes[0].set_xlabel('M * N * K')
axes[0].set_ylabel('Memory (MB)')

# Plot 2: ptorch Memory vs O(MN)
sns.regplot(
    data=df, 
    x='MN', 
    y='ptorch_run_mb', 
    ax=axes[1],
    scatter_kws={'alpha': 0.7},
    line_kws={'color': 'red'}
)
axes[1].set_title('ptorch Memory vs M*N')
axes[1].set_xlabel('M * N')
axes[1].set_ylabel('Memory (MB)')

plt.suptitle("Validating Memory Complexity Hypothesis\n(pjax -> O(MNK), ptorch -> O(MN))", y=1.05)
plt.tight_layout()

output_path = 'complexity_validation.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Plot successfully saved to {output_path}")
