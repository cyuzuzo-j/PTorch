import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

def plot_norm_comparison(results_dir: str, output_path: str):
    """
    Plots validation accuracy vs step for different projection norms.
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
    
    # Rename norm names for the legend
    norm_labels = {
        "l2": r"$\ell_2$ Projection",
        "linf": r"$\ell_\infty$ Projection"
    }
    df["Projection Norm"] = df["norm"].map(norm_labels).fillna(df["norm"])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    unique_norms = df["Projection Norm"].unique()
    palette = sns.color_palette("viridis", max(3, len(unique_norms)))[:len(unique_norms)]
    
    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="Projection Norm", estimator="mean", errorbar=("ci", 95),
        palette=palette, linewidth=2, ax=ax,
    )
    
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Effect of Projection Norm")
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "results", "norm_comparison")
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../images/norm_comparison.png"))
    plot_norm_comparison(results_dir, output_path)
