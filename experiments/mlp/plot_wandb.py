import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import yaml
current_hash = None
print(f"Current codebase hash: {current_hash}")

# Ensure shared imports are accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments.shared.neurips_style import set_neurips_style

def fetch_wandb_runs(project_name="pjax", entity=None, config_path="config.yaml"):
    """
    Connects to Weights & Biases to pull run history for MLP experiments.
    Downloads the logged metrics for steps, validation accuracy, and training time.
    """
    print(f"Connecting to wandb project: {project_name} ...")
    api = wandb.Api()
    
    # Load current config
    valid_tasks = []
    target_experiment = "mlp_benchmark_fast"
    current_hash = "unknown"
   
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg:
                target_experiment = cfg.get("experiment_name", "mlp_benchmark")
                for t in cfg.get("tasks", []):
                    valid_tasks.append(t["name"])
    
    path = f"{entity}/{project_name}" if entity else project_name
    
    # Filter directly in the API call to save massive amounts of time
    filters = {}
    if valid_tasks:
        filters["config.task"] = {"$in": valid_tasks}
        
    runs = api.runs(path, filters=filters)
    
    records = []
    
    for i, run in enumerate(runs):
        # Filter by target experiment
        if run.config.get("experiment_name", "") != target_experiment:
            continue
            
        print(f"Processing run {i+1}/{len(runs)}: {run.name} ...")
            
        task = run.config.get("task", "unknown")
        if "mlp" not in task.lower():
            # Sometimes config might be entirely empty for a run in the python API
            # Fall back to parsing the task from the name
            for v_t in valid_tasks:
                if v_t in run.name:
                    task = v_t
                    break
                    
        framework_raw = run.config.get("framework", "unknown")
        if framework_raw == "unknown":
            if "ptorch" in run.name.lower():
                framework_raw = "ptorch"
            elif "torch" in run.name.lower():
                framework_raw = "torch"
            elif "pjax" in run.name.lower():
                framework_raw = "pjax"

        if "ptorch" in framework_raw.lower():
            framework = "Ptorch"
        elif "torch" == framework_raw.lower():
            framework = "torch"
        elif "pjax" in framework_raw.lower():
            framework = "pjax"
        else:
            framework = framework_raw

        # Final validity check
        if task == "unknown" or framework not in ["Ptorch", "torch", "pjax"]:
            continue

        code_hash = run.config.get("code_hash", "none")
        if current_hash != "unknown" and code_hash != current_hash and code_hash != "none":
            print(f"Warning: Run {run.name} ({framework}) has code_hash {code_hash[:8]} which differs from current {current_hash[:8]}")

        # Download the full metric history for this run
        code_hash = run.config.get("code_hash", "none")
        try:
            history = run.history(samples=5000)  # increase if finer granularity is needed
            batch_size = run.config.get("batch_size", "unknown")
            
            if "val/val_acc" not in history.columns:
                continue
                
            history = history.dropna(subset=["val/val_acc"])
            for _, row in history.iterrows():
                records.append({
                        "Framework": framework,
                        "Task": task,
                        "Batch_Size": batch_size,
                        "Run_ID": run.id,
                        "Code_Hash": code_hash,
                        "Step": row.get("_step"),
                        "Time (s)": row.get("training_time_s"),
                        "Validation Accuracy": row.get("val/val_acc")
                    })
        except Exception as e:
            print(f"Failed to pull data for run {run.name}: {e}")
            
    df = pd.DataFrame(records)
    return df

def plot_mlp_results(df, output_dir):
    """
    Groups data by Task and plotting metric (Step vs Time), calculates error bars,
    and produces high-quality NeurIPS styled figures.
    """
    palette = set_neurips_style()
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = df["Task"].unique()
    batch_sizes = df["Batch_Size"].unique()
    
    for task in tasks:
        for bs in batch_sizes:
            task_data = df[(df["Task"] == task) & (df["Batch_Size"] == bs)]
            if task_data.empty:
                continue
                
            print(f"Plotting configurations for task: {task} (BS: {bs})...")
            
            # Plot 1: Accuracy vs Option Step
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            sns.lineplot(
                data=task_data, 
                x="Step", 
                y="Validation Accuracy", 
                hue="Framework", 
                estimator="mean",   # Calculates the mean over seeds/runs for aligned Steps
                errorbar=("ci", 95), # Generates 95% confidence interval cleanly
                palette=palette,
                ax=ax
            )
            ax.set_title(f"{task.split('_')[0]} (Depth: {task.split('_')[-1]}) [BS: {bs}]")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Validation Accuracy")
            
            # Set simpler legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, loc="lower right", frameon=False)
            sns.despine(ax=ax)
            
            plt.tight_layout()
            out_step_pdf = os.path.join(output_dir, f"{task}_BS{bs}_STEP.pdf")
            out_step_png = os.path.join(output_dir, f"{task}_BS{bs}_STEP.png")
            fig.savefig(out_step_pdf, bbox_inches='tight')
            fig.savefig(out_step_png, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Plot 2: Accuracy vs Wall-Clock Time
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            
            # Time has unaligned x-values across runs; using estimator="mean" causes zig-zags.
            # To fix this properly, we bin the unaligned times into uniform intervals.
            # Create a copy so we don't modify the original dataframe for time
            time_data = task_data.copy().dropna(subset=["Time (s)", "Validation Accuracy"])
            if not time_data.empty:
                # Create 50 evenly spaced time bins
                bins = 50
                time_data["Time_Bin"] = pd.cut(time_data["Time (s)"], bins=bins)
                time_data["Time_Bin_Mid"] = time_data["Time_Bin"].apply(lambda intv: intv.mid).astype(float)
                
                sns.lineplot(
                    data=time_data, 
                    x="Time_Bin_Mid", 
                    y="Validation Accuracy", 
                    hue="Framework", 
                    estimator="mean",
                    errorbar=("ci", 95),
                    palette=palette,
                    ax=ax
                )
            
            ax.set_title(f"{task.split('_')[0]} (Depth: {task.split('_')[-1]}) [BS: {bs}]")
            ax.set_xlabel("Wall-clock Time (s)")
            ax.set_ylabel("Validation Accuracy")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, loc="lower right", frameon=False)
            sns.despine(ax=ax)
            
            plt.tight_layout()
            out_time_pdf = os.path.join(output_dir, f"{task}_BS{bs}_TIME.pdf")
            out_time_png = os.path.join(output_dir, f"{task}_BS{bs}_TIME.png")
            plt.savefig(out_time_pdf, bbox_inches='tight')
            plt.savefig(out_time_png, bbox_inches='tight', dpi=300)
            plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NeurIPS ready comparison plots for NeurIPS Ptorch.")
    parser.add_argument("--project", default="pjax", help="WandB project name to pull records from.")
    parser.add_argument("--entity", default=None, help="WandB user/org entity if needed.")
    parser.add_argument("--outd", default=os.path.join(os.path.dirname(__file__), "../../images/mlp"), help="Output directory for plots.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="Config file to filter runs.")
    
    args = parser.parse_args()
    
    df_metrics = fetch_wandb_runs(project_name=args.project, entity=args.entity, config_path=args.config)
    
    if df_metrics.empty:
        print("No valid mlp data found in wandb runs.")
    else:
        plot_mlp_results(df_metrics, args.outd)
        print(f"Data successfully plotted and saved to: {args.outd}")
