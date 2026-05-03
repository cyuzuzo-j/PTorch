import matplotlib.pyplot as plt
import seaborn as sns

def set_neurips_style():
    """
    Sets the matplotlib and seaborn configurations to match NeurIPS paper guidelines.
    Returns a unified color palette for the frameworks.
    """
    # Use seaborn whitegrid context as a base
    sns.set_context("paper", rc={"grid.linestyle": "--", "grid.alpha": 0.5})
    sns.set_style("whitegrid")
    
    # NeurIPS standard typography and sizes
    plt.rcParams.update({
        "text.usetex": False,  # Set to True if you have a full LaTeX distribution installed
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 10,
        "figure.figsize": (3.25, 2.5),  # Half-width figure (~3.25 inches) strictly
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.format": "pdf",  # NeurIPS highly prefers vector graphics (PDF)
        "pdf.fonttype": 42, # Type 42 fonts (TrueType) so fonts are properly embedded
        "ps.fonttype": 42
    })
    
    # Define a consistent color palette across all experiments
    framework_palette = {
        "Ptorch": "#D62728",    # Red corresponding to Ptorch (novel projection)
        "torch": "#1F77B4",     # Blue corresponding to PyTorch baseline
        "pjax": "#2CA02C"       # Green corresponding to pjax
    }
    
    return framework_palette
