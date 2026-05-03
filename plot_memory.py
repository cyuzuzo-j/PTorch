import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('memory_benchmark_results.csv')

# Melt the dataframe for easier plotting with seaborn
df_melt = pd.melt(df, id_vars=['M', 'N', 'K'], 
                  value_vars=['torch_run_mb', 'ptorch_run_mb', 'pjax_run_mb'],
                  var_name='Framework', value_name='Memory (MB)')

# Clean up framework names for the legend
df_melt['Framework'] = df_melt['Framework'].replace({
    'torch_run_mb': 'PyTorch',
    'ptorch_run_mb': 'Ours (PyTorch)',
    'pjax_run_mb': 'Ours (JAX)'
})

# Create a grid of line plots
sns.set_theme(style="whitegrid")
g = sns.relplot(
    data=df_melt, x="K", y="Memory (MB)",
    hue="Framework", col="M", row="N", kind="line", marker="o",
    height=3.5, aspect=1.2, facet_kws={'sharey': False}
)

g.set_axis_labels("K (Hidden Dimension)", "Memory Usage (MB)")
g.fig.suptitle("Memory Usage Benchmark by Framework", y=1.05)

plt.savefig('memory_benchmark_plot.png', bbox_inches='tight')
print("Plot successfully saved to memory_benchmark_plot.png")
