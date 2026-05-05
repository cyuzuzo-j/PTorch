import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

def plot_optimizer_comparison(results_dir: str, output_path: str):
    """
    Plots validation accuracy vs step for different optimizers.
    """
    frames = []
    pattern = os.path.join(results_dir, "*.csv")
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        frames.append(df)
        
    if not frames:
        print(f"No data found in {results_dir}")
        return
        
    df = pd.concat(frames, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    unique_opts = df["optimizer"].unique()
    palette = sns.color_palette("viridis", max(3, len(unique_opts)))[:len(unique_opts)]
    
    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="optimizer", estimator="mean", errorbar=("ci", 95),
        palette=palette, linewidth=2, ax=ax,
    )
    
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Optimizer Comparison")
    ax.legend(title="Optimizer", loc="lower right", fontsize=12, frameon=False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "results", "optimizer_comparison_torch")
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../images/optimizer_comparison_torch.png"))
    plot_optimizer_comparison(results_dir, output_path)
