import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

csv_file_path = "experiment_metrics.csv"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(csv_file_path)
data_filtered = data[
    (data['Status'] == "Success") &
    data[['Overall Score']].notnull().all(axis=1) # , 'Fitting Time', 'Sampling Time'
]

# ---------------------- Experiment 1 ----------------------
exp1_data = data_filtered[
    (data_filtered['Dataset Name'] == "covtype") &
    (data_filtered['Num Train Samples'] == 5000) &
    (data_filtered['Num Test Samples'] == 2000) &
    (data_filtered['Num Train Epochs'] == 5)
]

if exp1_data.empty:
    print("No data available for Experiment 1 under given conditions.")
else:
    exp1_agg = exp1_data.groupby(["Synthesizer Name", "Decomposer Name"]).mean(numeric_only=True).reset_index()

    quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                       "Boundary Adherence", "Range Coverage", "KS Complement",
                       "TV Complement", "Correlation Similarity", "Contingency Similarity"]
    time_metrics = ["Fitting Time", "Decomposition Time", "Sampling Time"]

    available_cols = exp1_agg.columns
    quality_metrics = [m for m in quality_metrics if m in available_cols]
    time_metrics = [m for m in time_metrics if m in available_cols]

    synthesizers = sorted(exp1_agg['Synthesizer Name'].unique())
    decomposers = sorted(exp1_agg['Decomposer Name'].unique())
    x = np.arange(len(synthesizers))

    def plot_clustered_bars(metrics, title_prefix, file_name_prefix, y_label_suffix, rows, cols):
        total_subplots = rows * cols
        metrics_count = len(metrics)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = axes[:, np.newaxis]
        axes_flat = axes.flatten()
        width = 0.8 / len(decomposers)

        cmap = plt.get_cmap("GnBu")
        colors = [cmap(i / (len(decomposers) - 1)) for i in range(len(decomposers))]
        colors.reverse()

        for i, metric in enumerate(metrics):
            if i >= total_subplots:
                break
            ax = axes_flat[i]
            metric_data = []
            for syn in synthesizers:
                row = []
                for dec in decomposers:
                    val = exp1_agg[(exp1_agg['Synthesizer Name'] == syn) & (exp1_agg['Decomposer Name'] == dec)][metric].mean()
                    row.append(val if pd.notnull(val) else 0)
                metric_data.append(row)
            metric_data = np.array(metric_data)

            for j, dec in enumerate(decomposers):
                label = dec if i == 0 else None 
                ax.bar(x + j * width, metric_data[:, j], width, color=colors[j], label=label)

            row_index = i // cols
            if row_index == rows - 1:
                ax.set_xticks(x + width * (len(decomposers) - 1) / 2)
                ax.set_xticklabels(synthesizers, rotation=45, ha='right')
            else:
                ax.set_xticks([])
            ax.set_title(metric, fontsize=10)
        
        for j in range(metrics_count, total_subplots):
            axes_flat[j].axis('off')

        fig.suptitle(title_prefix, fontsize=14)
        fig.text(0.04, 0.5, y_label_suffix, va='center', rotation='vertical', fontsize=12)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Decomposer Name", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(decomposers)))

        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])
        plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_clustered_bars.png"), bbox_inches='tight')
        plt.close()

    plot_clustered_bars(quality_metrics, "Quality Metrics", "experiment1_quality", "(Quality)", 3, 3)
    plot_clustered_bars(time_metrics, "Time Metrics", "experiment1_time", "(Time)", 3,1)
    print("Experiment 1 plots generated.")


# ---------------------- Use curve plot ----------------------
quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                   "Boundary Adherence", "Range Coverage", "KS Complement",
                   "TV Complement", "Correlation Similarity", "Contingency Similarity"]
time_metrics = ["Fitting Time", "Decomposition Time", "Sampling Time"]

def plot_experiment_metrics(data, x_axis_col, x_axis_label, title_prefix, file_name_prefix, y_label_suffix, x_values, is_time=False):
    metrics = time_metrics if is_time else quality_metrics
    rows, cols = (1, 3) if is_time else (3, 3)
    total_subplots = rows * cols
    metrics_count = len(metrics)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes[:, np.newaxis]
    axes_flat = axes.flatten()

    sns.set_palette("GnBu_d")
    decomposers = sorted(data['Decomposer Name'].unique())

    for i, metric in enumerate(metrics):
        if i >= total_subplots:
            break
        ax = axes_flat[i]

        for dec_i, dec in enumerate(decomposers):
            sub_data = data[data['Decomposer Name'] == dec]
            y_values = [sub_data[sub_data[x_axis_col] == x][metric].mean() for x in x_values]
            label = dec if i == 0 else None
            ax.plot(x_values, y_values, marker='o', label=label)

        ax.set_title(metric, fontsize=10)
        if (i // cols) == rows - 1:
            ax.set_xlabel(x_axis_label, fontsize=10)
        if (i % cols) == 0:
            ax.set_ylabel(y_label_suffix, fontsize=10)
        ax.grid(True)

    for j in range(metrics_count, total_subplots):
        axes_flat[j].axis('off')

    fig.suptitle(title_prefix, fontsize=16)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Decomposer Name", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(decomposers)))

    plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_{'time' if is_time else 'quality'}_metrics.png"))
    plt.close()

# ---------------------- Experiment 2 ----------------------
exp2_data = data_filtered[
    (data_filtered['Dataset Name'] == "covtype") &
    (data_filtered['Num Test Samples'] == 2000) &
    (data_filtered['Num Train Epochs'] == 5) &
    (data_filtered['Synthesizer Name'] == "TVAESynthesizer")
]
if not exp2_data.empty:
    train_samples = sorted(exp2_data['Num Train Samples'].unique())
    plot_experiment_metrics(exp2_data, 'Num Train Samples', "Training Samples",
                            "Experiment 2: Quality Metrics", "experiment2_quality", "Quality", train_samples, is_time=False)
    plot_experiment_metrics(exp2_data, 'Num Train Samples', "Training Samples",
                            "Experiment 2: Time Metrics", "experiment2_time", "Time (s)", train_samples, is_time=True)

# ---------------------- Experiment 3 ----------------------
exp3_data = data_filtered[
    (data_filtered['Dataset Name'] == "covtype") &
    (data_filtered['Num Train Samples'] == 5000) &
    (data_filtered['Num Train Epochs'] == 5) &
    (data_filtered['Synthesizer Name'] == "TVAESynthesizer")
]
if not exp3_data.empty:
    test_samples = sorted(exp3_data['Num Test Samples'].unique())
    plot_experiment_metrics(exp3_data, 'Num Test Samples', "Testing Samples",
                            "Experiment 3: Quality Metrics", "experiment3_quality", "Quality", test_samples, is_time=False)
    plot_experiment_metrics(exp3_data, 'Num Test Samples', "Testing Samples",
                            "Experiment 3: Time Metrics", "experiment3_time", "Time (s)", test_samples, is_time=True)

# ---------------------- Experiment 4 ----------------------
exp4_data = data_filtered[
    (data_filtered['Dataset Name'].isin(["asia", "adult", "insurance", "alarm", "covtype", "mnist12"])) &
    (data_filtered['Synthesizer Name'] == "TVAESynthesizer") &
    (data_filtered['Num Train Epochs'] == 5)
]
dataset_columns = {
    "asia": 8,
    "adult": 14,
    "insurance": 27,
    "alarm": 37,
    "covtype": 54,
    "mnist12": 12
}
if not exp4_data.empty:
    datasets = sorted(exp4_data['Dataset Name'].unique(), key=lambda x: dataset_columns[x])
    plot_experiment_metrics(exp4_data, 'Dataset Name', "Datasets (ordered by column count)",
                            "Experiment 4: Quality Metrics", "experiment4_quality", "Quality", datasets, is_time=False)
    plot_experiment_metrics(exp4_data, 'Dataset Name', "Datasets (ordered by column count)",
                            "Experiment 4: Time Metrics", "experiment4_time", "Time (s)", datasets, is_time=True)

# ---------------------- Experiment 5 ----------------------
exp5_data = data_filtered[
    (data_filtered['Dataset Name'] == "covtype") &
    (data_filtered['Synthesizer Name'] == "CTGANSynthesizer") &
    (data_filtered['Decomposer Name'] == "PCADecomposition") &
    (data_filtered['Num Train Samples'] == 5000) &
    (data_filtered['Num Test Samples'] == 2000) &
    (data_filtered['Num Train Epochs'] == 5)
]
if not exp5_data.empty:
    pca_components = sorted(exp5_data['Num Train Samples'].unique())
    plot_experiment_metrics(exp5_data, 'Num Train Samples', "PCA Components",
                            "Experiment 5: Quality Metrics", "experiment5_quality", "Quality", pca_components, is_time=False)
    plot_experiment_metrics(exp5_data, 'Num Train Samples', "PCA Components",
                            "Experiment 5: Time Metrics", "experiment5_time", "Time (s)", pca_components, is_time=True)

print("All plots have been successfully generated.")
