#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Paper-ready W&B Plotter")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to W&B exported CSV")
    parser.add_argument("--output", "-o", type=str, default="figures/wandb_plot", help="Output filename (without extension)")
    parser.add_argument("--metric", "-m", type=str, default="val/val_acc", help="Metric to plot (y-axis)")
    parser.add_argument("--step", "-s", type=str, default="_step", help="Step column (x-axis)")
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing weight (0 to 1)")
    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--xlabel", type=str, help="X-axis label")
    parser.add_argument("--ylabel", type=str, help="Y-axis label")
    parser.add_argument("--frameworks", type=str, help="Comma-separated list of frameworks to include (regex)")
    return parser.parse_args()

def apply_style():
    """Apply academic paper style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    plt.rcParams.update({
        "figure.figsize": (10, 7),
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.5,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "text.usetex": False,  # Set to True if system has LaTeX
    })

def ema_smoothing(values, weight):
    """Exponential Moving Average smoothing."""
    if weight <= 0:
        return values
    smoothed = []
    last = values[0]
    for val in values:
        if np.isnan(val):
            smoothed.append(np.nan)
            continue
        new_val = last * weight + (1 - weight) * val
        smoothed.append(new_val)
        last = new_val
    return np.array(smoothed)

def extract_framework_data(df, metric_name, step_name):
    """
    Extracts (step, metric) pairs for each framework detected in the CSV.
    W&B headers often look like: 'framework: torch - val/val_acc'
    """
    framework_data = {}
    
    # Identify frameworks
    cols = df.columns
    # Pattern to match: "framework: NAME - METRIC"
    pattern = re.compile(r"framework: (.*) - (.*)")
    
    # Map (framework, metric_type) -> column_name
    # metric_type is 'val/val_acc' or '_step'
    data_map = {}
    for col in cols:
        match = pattern.search(col)
        if match:
            framework, m_name = match.groups()
            if framework not in data_map:
                data_map[framework] = {}
            data_map[framework][m_name] = col
            
    for framework, metrics in data_map.items():
        if metric_name in metrics:
            # Determine step column: framework-specific or global fallback
            s_col = metrics.get(step_name)
            if s_col is None:
                # Check for global step column
                if step_name in df.columns:
                    s_col = step_name
                elif "Step" in df.columns: # common W&B export
                    s_col = "Step"
            
            if s_col:
                f_df = df[[s_col, metrics[metric_name]]].dropna()
                if not f_df.empty:
                    # Sort by step
                    f_df = f_df.sort_values(by=s_col)
                    steps = f_df[s_col].values
                    values = f_df[metrics[metric_name]].values
                    framework_data[framework] = (steps, values)
                
    return framework_data

def main():
    args = parse_args()
    apply_style()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return
        
    df = pd.read_csv(args.input)
    
    # Try to extract framework data
    framework_data = extract_framework_data(df, args.metric, args.step)
    
    if not framework_data:
        print(f"Warning: No framework data found matching metric '{args.metric}' and step '{args.step}'.")
        print("Columns found:", df.columns.tolist())
        return

    # Filtering
    if args.frameworks:
        fw_regex = re.compile(args.frameworks)
        framework_data = {k: v for k, v in framework_data.items() if fw_regex.search(k)}

    fig, ax = plt.subplots()
    
    # Use a nice palette
    colors = sns.color_palette("husl", len(framework_data))
    
    for i, (framework, (steps, values)) in enumerate(framework_data.items()):
        if args.smooth > 0:
            values = ema_smoothing(values, args.smooth)
            
        ax.plot(steps, values, label=framework, color=colors[i], alpha=0.9, marker='o', markersize=8)
        
        # Optionally add shaded area for MIN/MAX if they exist (advanced)
        # For simplicity, we just plot the main line first.

    ax.set_xlabel(args.xlabel if args.xlabel else args.step)
    ax.set_ylabel(args.ylabel if args.ylabel else args.metric)
    if args.title:
        ax.set_title(args.title)
    
    ax.legend(frameon=True, shadow=True)
    
    # Export formats
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(f"{args.output}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    
    print(f"Successfully generated plots: {args.output}.pdf/png")

if __name__ == "__main__":
    main()
