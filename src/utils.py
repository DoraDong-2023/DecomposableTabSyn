import matplotlib.pyplot as plt
import numpy as np

def plot_model_scaling(num_columns, fit_times_dict, title="Model Scaling with Number of Columns", ylabel='Fit Time (seconds)'):
    """
    Plot fit times for different models against number of columns.
    
    Parameters:
    -----------
    num_columns : array-like
        Array containing the number of columns for each experiment
    fit_times_dict : dict
        Dictionary where keys are model names and values are arrays of fit times
        corresponding to num_columns
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create different marker styles and colors for each model
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for (model_name, fit_times), marker, color in zip(
        fit_times_dict.items(), 
        markers, 
        colors
    ):
        # Plot with both lines and markers
        plt.plot(num_columns, fit_times, 
                marker=marker, 
                linestyle='-', 
                label=model_name,
                color=color,
                linewidth=2,
                markersize=8,
                alpha=0.7)
    
    plt.xlabel('Number of Columns', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(loc='upper left')
    
    # Use log scale if the data spans multiple orders of magnitude
    if np.max(num_columns) / np.min(num_columns) > 100:
        plt.xscale('log')
    if any(np.max(times) / np.min(times) > 100 for times in fit_times_dict.values()):
        plt.yscale('log')
    
    plt.tight_layout()
    return plt.gcf()