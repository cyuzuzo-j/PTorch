import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('memory_benchmark_results.csv')

df_melt = pd.melt(df, id_vars=['M', 'N', 'K'], 
                  value_vars=['torch_run_mb', 'ptorch_run_mb', 'pjax_run_mb'],
                  var_name='Framework', value_name='Memory (MB)')

df_melt['Framework'] = df_melt['Framework'].map({
    'torch_run_mb': 'PyTorch',
    'ptorch_run_mb': 'PTorch',
    'pjax_run_mb': 'PJAX'
})

# Filter for the largest M and N to make the scaling with K obvious
df_large = df_melt[(df_melt['M'] == 500) & (df_melt['N'] == 256)]

plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

sns.lineplot(data=df_large, x='K', y='Memory (MB)', hue='Framework', marker='o', linewidth=2)

plt.title('Memory vs K (for fixed M=500, N=256)')
plt.xlabel('K (Input Features)')
plt.ylabel('Peak Memory (MB)')
plt.ylim(bottom=0) # Start Y axis from 0 to show true scale

plt.tight_layout()
output_file = 'memory_vs_k_report.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")
