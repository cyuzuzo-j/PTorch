import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# Load the data
df = pd.read_csv('memory_benchmark_results.csv')

# Calculate complexities
df['MNK'] = df['M'] * df['N'] * df['K']
df['MN'] = df['M'] * df['N']

# Frameworks to evaluate
frameworks = {
    'PyTorch': 'torch_run_mb',
    'Ours (PyTorch)': 'ptorch_run_mb',
    'Ours (JAX)': 'pjax_run_mb'
}

# Calculate correlations
results = []
for name, col in frameworks.items():
    corr_mn, _ = pearsonr(df['MN'], df[col])
    corr_mnk, _ = pearsonr(df['MNK'], df[col])
    results.append({'Framework': name, 'Complexity': 'O(MN)', 'Correlation': corr_mn})
    results.append({'Framework': name, 'Complexity': 'O(MNK)', 'Correlation': corr_mnk})

df_corr = pd.DataFrame(results)

# Create the plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

ax = sns.barplot(
    data=df_corr, 
    x='Framework', 
    y='Correlation', 
    hue='Complexity',
    palette='Set2'
)

# Add values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                xytext=(0, 5), textcoords='offset points')

plt.title('Memory Scaling Correlation by Framework')
plt.ylabel('Pearson Correlation Coefficient (r)')
plt.ylim(0, 1.1)  # Set limit slightly above 1 for annotations
plt.legend(title='Scaling Factor')

plt.tight_layout()
output_file = 'correlation_report_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")
