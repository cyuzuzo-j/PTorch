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
    Plots validation accuracy vs wall-clock time for different projection norms.
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
    
    # Bin unaligned wall-clock times into uniform intervals for clean averaging
    time_data = df.dropna(subset=["wall_time_s", "val_acc"]).copy()
    if not time_data.empty:
        n_bins = 60
        time_data["time_bin"] = pd.cut(time_data["wall_time_s"], bins=n_bins)
        time_data["time_mid"] = time_data["time_bin"].apply(
            lambda iv: iv.mid if pd.notnull(iv) else None
        ).astype(float)

        sns.lineplot(
            data=time_data, x="time_mid", y="val_acc",
            hue="Projection Norm", estimator="mean", errorbar=("ci", 95),
            palette=palette, linewidth=2, ax=ax,
        )
    
    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Validation Accuracy")
    ax.set(xlim=(0, 50))
    ax.set(ylim=(0.8, 1.0))

    ax.set_title("Effect of Projection Norm")
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "results", "norm_comparison")
    # Updated output filename to reflect the change to time
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../images/norm_comparison_time.pdf"))
    plot_norm_comparison(results_dir, output_path)