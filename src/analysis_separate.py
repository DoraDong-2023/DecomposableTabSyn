import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast

csv_file_path = "experiment_metrics_with_labels.csv"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

cmap = plt.get_cmap("GnBu")
colors = [cmap(i / 7) for i in range(8)]
colors.reverse()

data = pd.read_csv(csv_file_path)
data_filtered = data
#data_filtered = data[data['Overall Score'].notnull()]  # Filter only rows with valid metrics
data_filtered['n_components'] = data_filtered['Parsed Decomposer'].apply(lambda x: ast.literal_eval(x).get('n_components', None))

# ---------------------- Experiment 1 ----------------------
exp1_data = data_filtered[data_filtered['Experiment1_YN'] == 'Y']


if exp1_data.empty:
    print("No data available for Experiment 1 under given conditions.")
else:
    exp1_agg = exp1_data.groupby(["Synthesizer Name", "Parsed Decomposer Name"]).mean(numeric_only=True).reset_index()

    quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                       "Boundary Adherence", "Range Coverage", "KS Complement",
                       "TV Complement", "Correlation Similarity", "Contingency Similarity"]
    time_metrics = ["Fitting Time", "Decomposition Time", "Sampling Time"]

    available_cols = exp1_agg.columns
    quality_metrics = [m for m in quality_metrics if m in available_cols]
    time_metrics = [m for m in time_metrics if m in available_cols]

    synthesizers = sorted(exp1_agg['Synthesizer Name'].unique())
    decomposers = sorted(exp1_agg['Parsed Decomposer Name'].unique())
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

        
        #colors = [cmap(i / (len(decomposers) - 1)) for i in range(len(decomposers))]
        #colors.reverse()

        for i, metric in enumerate(metrics):
            if i >= total_subplots:
                break
            ax = axes_flat[i]
            metric_data = []
            for syn in synthesizers:
                row = []
                for dec in decomposers:
                    val = exp1_agg[(exp1_agg['Synthesizer Name'] == syn) & (exp1_agg['Parsed Decomposer Name'] == dec)][metric].mean()
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
        fig.legend(handles, labels, title="Decomposer", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(6, len(decomposers)))

        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])
        plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_clustered_bars.pdf"), bbox_inches='tight')
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
    decomposers = sorted(data['Parsed Decomposer Name'].unique())

    for i, metric in enumerate(metrics):
        if i >= total_subplots:
            break
        ax = axes_flat[i]

        for dec_i, dec in enumerate(decomposers):
            sub_data = data[data['Parsed Decomposer Name'] == dec]
            y_values = [sub_data[sub_data[x_axis_col] == x][metric].mean() for x in x_values]
            label = dec if i == 0 else None
            ax.plot(x_values, y_values, marker='o', color=colors[dec_i], label=label)

        ax.set_title(metric, fontsize=10)
        if (i // cols) == rows - 1:
            ax.set_xlabel(x_axis_label, fontsize=10)
        if (i % cols) == 0:
            ax.set_ylabel(y_label_suffix, fontsize=10)
        ax.grid(False)

    for j in range(metrics_count, total_subplots):
        axes_flat[j].axis('off')

    fig.suptitle(title_prefix, fontsize=16)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Decomposer", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(6, len(decomposers)))

    plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_{'time' if is_time else 'quality'}_metrics.pdf"))
    plt.close()

def plot_4x1_with_metrics(data, x_axis_col, x_values, title_prefix, file_name_prefix):
    """
    Plot aggregated metrics in a 4x1 layout.
    """
    metrics = ["Overall Score", "Fitting Time", "Sampling Time", "Decomposition Time"]
    decomposers = sorted(data['Parsed Decomposer Name'].unique())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for dec_i, dec in enumerate(decomposers):
            sub_data = data[data['Parsed Decomposer Name'] == dec]
            y_values = [sub_data[sub_data[x_axis_col] == x][metric].mean() for x in x_values]
            ax.plot(x_values, y_values, marker='o', color=colors[dec_i], label=dec)

        ax.set_title(metric, fontsize=12)
        ax.set_xlabel(x_axis_col, fontsize=10)
        if i == 0:
            ax.set_ylabel("Value", fontsize=10)
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Decomposer", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4)
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_4x1.pdf"))
    plt.close()
    

def plot_metrics(data, x_axis_col, x_label, metrics, title, file_name, y_label, layout, x_values, metric_type):
    """
    Generalized function to plot quality/time metrics in 3x3, 1x3, or 4x1 layouts.
    Args:
        data: Filtered data for the experiment.
        x_axis_col: Column to use for the x-axis.
        x_label: Label for the x-axis.
        metrics: List of metrics to plot (quality or time).
        title: Title for the plot.
        file_name: Output file name.
        y_label: Label for the y-axis.
        layout: Tuple (rows, cols) for subplot layout.
        x_values: Ordered values for the x-axis.
        metric_type: "quality" or "time".
    """
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    decomposers = sorted(data['Parsed Decomposer Name'].unique())

    for i, metric in enumerate(metrics):
        if i >= rows * cols:
            break
        ax = axes[i]
        for dec_i, dec in enumerate(decomposers):
            sub_data = data[data['Parsed Decomposer Name'] == dec]
            y_values = [sub_data[sub_data[x_axis_col] == x][metric].mean() for x in x_values]
            ax.plot(x_values, y_values, marker='o', color=colors[dec_i], label=dec)

        ax.set_title(metric, fontsize=10)
        if i // cols == rows - 1:
            ax.set_xlabel(x_label, fontsize=10)
        if i % cols == 0:
            ax.set_ylabel(y_label, fontsize=10)
        ax.grid(False)

    for j in range(len(metrics), rows * cols):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Decomposer", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name}_{metric_type}.pdf"))
    plt.close()


def plot_aggregated(data, x_axis_col, x_values, metrics, title, file_name):
    """
    Plot aggregated metrics in a 4x1 layout.
    """
    decomposers = sorted(data['Parsed Decomposer Name'].unique())
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for dec_i, dec in enumerate(decomposers):
            sub_data = data[data['Parsed Decomposer Name'] == dec]
            y_values = [sub_data[sub_data[x_axis_col] == x][metric].mean() for x in x_values]
            ax.plot(x_values, y_values, marker='o', color=colors[dec_i], label=dec)

        ax.set_title(metric, fontsize=10)
        ax.set_xlabel(x_axis_col, fontsize=10)
        if i == 0:
            ax.set_ylabel("Value", fontsize=10)
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Decomposer", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name}_aggregated.pdf"))
    plt.close()

# ---------------------- Experiment 2 ----------------------
exp2_data = data_filtered[data_filtered['Experiment2_YN'] == 'Y']
if not exp2_data.empty:
    train_samples = sorted(exp2_data['Num Train Samples'].unique())
    quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                   "Boundary Adherence", "Range Coverage", "KS Complement",
                   "TV Complement", "Correlation Similarity", "Contingency Similarity"]
    time_metrics = ["Fitting Time", "Sampling Time", "Decomposition Time"]

    plot_metrics(
        exp2_data, 'Num Train Samples', "Training Samples",
        quality_metrics, "Quality Metrics", "experiment2", "Quality",
        (3, 3), train_samples, "quality"
    )
    plot_metrics(
        exp2_data, 'Num Train Samples', "Training Samples",
        time_metrics, "Time Metrics", "experiment2", "Time (s)",
        (3,1), train_samples, "time"
    )
    plot_aggregated(
        exp2_data, 'Num Train Samples', train_samples,
        ["Overall Score", "Fitting Time", "Sampling Time", "Decomposition Time"], "Aggregated Metrics", "experiment2"
    )


# ---------------------- Experiment 3 ----------------------
exp3_data = data_filtered[data_filtered['Experiment3_YN'] == 'Y']
if not exp3_data.empty:
    test_samples = sorted(exp3_data['Num Test Samples'].unique())
    quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                   "Boundary Adherence", "Range Coverage", "KS Complement",
                   "TV Complement", "Correlation Similarity", "Contingency Similarity"]
    time_metrics = ["Fitting Time", "Sampling Time", "Decomposition Time"]

    plot_metrics(
        exp3_data, 'Num Test Samples', "Testing Samples",
        quality_metrics, "Quality Metrics", "experiment3", "Quality",
        (3, 3), test_samples, "quality"
    )
    plot_metrics(
        exp3_data, 'Num Test Samples', "Testing Samples",
        time_metrics, "Time Metrics", "experiment3", "Time (s)",
        (3,1), test_samples, "time"
    )
    plot_aggregated(
        exp3_data, 'Num Test Samples', test_samples,
        ["Overall Score", "Fitting Time", "Sampling Time", "Decomposition Time"], "Aggregated Metrics", "experiment3"
    )

# ---------------------- Experiment 4 ----------------------
#exp4_filtered = data_filtered[(data_filtered['Experiment4_YN'] == 'Y')]
exp4_filtered = data_filtered
exp4_filtered = exp4_filtered[exp4_filtered['Num Train Samples'] == 5000]
exp4_filtered = exp4_filtered[exp4_filtered['Num Test Samples'] == 2000]
exp4_filtered = exp4_filtered[exp4_filtered['Synthesizer Name'] == 'REaLTabFormer']
exp4_filtered = exp4_filtered[exp4_filtered['Parsed Decomposer Name'].isin(['no_decomposition', 'PCADecomposition'])]
exp4_filtered = exp4_filtered[exp4_filtered['Decomposer Name'].isin(['no_decomposition', 'PCADecomposition_n_8'])]
print(exp4_filtered)
dataset_columns = {
    "asia": 8,
    "adult": 14,
    "insurance": 27,
    "alarm": 37,
    "covtype": 54,
    "mnist12": 12
}

exp4_filtered_decomposers = ["no_decomposition", "PCADecomposition"]
#exp4_filtered = exp4_filtered[exp4_filtered['Parsed Decomposer Name'].isin(desired_decomposers)]
#print(exp4_filtered)

unique_datasets = ['asia', 'covtype', 'alarm', 'adult', 'mnist12', 'insurance']

def plot_clustered_bars_exp4_inverted(agg_data, metrics, title_prefix, file_name_prefix, y_label_suffix="Value"):
    decomposers_list = exp4_filtered_decomposers
    x = np.arange(len(decomposers_list))  # 两个decomposer对应两个x位置
    width = 0.8 / len(unique_datasets)    # 每个decomposer处插入多个dataset的bar

    # 固定为1行3列布局
    rows, cols = (3, 3)
    total_subplots = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    # 当rows=1时，axes为一维数组
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= total_subplots:
            break  # 超过3个metric的就不绘制了
        ax = axes[i]

        # 构建数据矩阵
        metric_data = []
        for dec in decomposers_list:
            row = []
            for ds in unique_datasets:
                val = agg_data[(agg_data['Dataset Name'] == ds) & (agg_data['Parsed Decomposer Name'] == dec)][metric].mean()
                row.append(val if pd.notnull(val) else 0)
            metric_data.append(row)
        metric_data = np.array(metric_data)

        for j, ds in enumerate(unique_datasets):
            label = ds if i == 0 else None
            ax.bar(x + j * width, metric_data[:, j], width, color=colors[j], label=label)

        ax.set_title(metric, fontsize=10)
        ax.set_xticks(x + width * (len(unique_datasets) - 1) / 2)
        ax.set_xticklabels(decomposers_list, rotation=45, ha='right', fontsize=8)

        if i == 0:
            ax.set_ylabel(y_label_suffix, fontsize=9)

    # 隐藏多余的子图（如果metrics少于3个，不需要，但无妨）
    for j in range(len(metrics), total_subplots):
        axes[j].axis('off')

    fig.suptitle(title_prefix, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(6, len(unique_datasets)))
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_1x3_clustered_bars.pdf"), bbox_inches='tight')
    plt.close()

exp4_agg = exp4_filtered.groupby(["Dataset Name", "Parsed Decomposer Name"]).mean(numeric_only=True).reset_index()

quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                   "Boundary Adherence", "Range Coverage", "KS Complement",
                   "TV Complement", "Correlation Similarity", "Contingency Similarity"]
time_metrics = ["Fitting Time", "Sampling Time", "Decomposition Time"]

available_cols = exp4_agg.columns
time_metrics = [m for m in time_metrics if m in available_cols]

def plot_4x1_with_metrics_exp4(data, metrics, title_prefix, file_name_prefix, y_label_suffix="Value"):
    decomposers_list = exp4_filtered_decomposers
    x = np.arange(len(decomposers_list))  
    width = 0.8 / len(unique_datasets)

    # 固定为1行4列布局
    rows, cols = (1, 4)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    for i, metric in enumerate(metrics):
        if i >= cols:  # 超过4个不绘制
            break
        ax = axes[i]

        # 构建数据矩阵
        metric_data = []
        for dec in decomposers_list:
            row = []
            for ds in unique_datasets:
                val = data[(data['Dataset Name'] == ds) & (data['Parsed Decomposer Name'] == dec)][metric].mean()
                row.append(val if pd.notnull(val) else 0)
            metric_data.append(row)
        metric_data = np.array(metric_data)

        for j, ds in enumerate(unique_datasets):
            label = ds if i == 0 else None
            ax.bar(x + j * width, metric_data[:, j], width, color=colors[j], label=label)

        ax.set_title(metric, fontsize=10)
        ax.set_xticks(x + width * (len(unique_datasets) - 1) / 2)
        ax.set_xticklabels(decomposers_list, rotation=45, ha='right', fontsize=8)

        if i == 0:
            ax.set_ylabel(y_label_suffix, fontsize=9)

    fig.suptitle(title_prefix, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(6, len(unique_datasets)))
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_4x1_clustered_bars.pdf"), bbox_inches='tight')
    plt.close()

# 绘图调用
plot_clustered_bars_exp4_inverted(exp4_agg, quality_metrics, 
                                  "Experiment 4: Quality (no_decomposition vs PCADecomposition, Train=5000, Test=2000)", 
                                  "experiment4_inverted_quality_5000_2000", 
                                  "(Quality)")

plot_clustered_bars_exp4_inverted(exp4_agg, time_metrics, 
                                  "Experiment 4: Time (no_decomposition vs PCADecomposition, Train=5000, Test=2000)", 
                                  "experiment4_inverted_time_5000_2000", 
                                  "(Time)")
agg_metrics = ["Metrics Overall Score", "Fitting Time", "Sampling Time", "Decomposition Time"]
plot_4x1_with_metrics_exp4(
    exp4_agg,
    agg_metrics,
    "Experiment 4: Aggregated Metrics (no_decomposition vs PCADecomposition, Train=5000, Test=2000)",
    "experiment4_aggregated_5000_2000",
    y_label_suffix="Value"
)
# ---------------------- Experiment 5 ----------------------
import ast
exp5_data = data_filtered[(data_filtered['Experiment5_YN'] == 'Y')]

# Extract baseline values for the dataset "covtype"
baseline_data = data_filtered[
    (data_filtered['Dataset Name'] == 'covtype') &
    (data_filtered['Parsed Decomposer Name'] == 'no_decomposition') &
    (data_filtered['Num Train Samples'] == 5000) &
    (data_filtered['Num Test Samples'] == 2000) &
    (data_filtered['Synthesizer Name'].isin(['TVAESynthesizer', 'REaLTabFormer']))
]
print(baseline_data)
# Assert only one valid value per metric
#assert len(baseline_data) == 1, "Baseline data should have exactly one row for 'covtype'!"
baseline_values = baseline_data.iloc[0].to_dict()

# Define quality and time metrics for plotting
quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                   "Boundary Adherence", "Range Coverage", "KS Complement",
                   "TV Complement", "Correlation Similarity", "Contingency Similarity"]
# Extract unique n_components and synthesizers
n_components = sorted(exp5_data['n_components'].unique())
synthesizers = exp5_data['Synthesizer Name'].unique()

time_metrics = ["Fitting Time", "Sampling Time", "Decomposition Time"]

#baseline_colors = [sns.color_palette("GnBu")[0], sns.color_palette("GnBu")[-1]]
baseline_colors = ["darkgray", "lightgray"]

def plot_experiment5_time_with_consistent_layout(data, title_prefix, file_name_prefix):
    """
    Plot Experiment 5 time metrics with consistent layout and colors.
    """
    # Define time metrics
    time_metrics = ["Fitting Time", "Sampling Time", "Decomposition Time"]

    # Colors for synthesizers
    baseline_colors = ["darkgray", "lightgray"]

    # Extract baseline values for each synthesizer
    baselines = {}
    for syn in ['TVAESynthesizer', 'REaLTabFormer']:
        baseline = baseline_data[baseline_data['Synthesizer Name'] == syn]
        assert len(baseline) == 1, f"Baseline data for synthesizer {syn} should have exactly one row!"
        baselines[syn] = baseline.iloc[0].to_dict()

    # Plot the metrics in a 3x1 layout
    rows, cols = 3, 1
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten()

    for i, metric in enumerate(time_metrics):
        ax = axes_flat[i]

        # Plot time metrics for each synthesizer
        for idx, syn in enumerate(synthesizers):
            sub_data = data[data['Synthesizer Name'] == syn]
            y_values = [sub_data[sub_data['n_components'] == c][metric].mean() for c in n_components]
            ax.plot(n_components, y_values, marker='o', color=colors[idx], label=syn)

        # Add baselines for each synthesizer
        for idx, (syn, baseline_values) in enumerate(baselines.items()):
            baseline_value = baseline_values[metric]
            ax.axhline(y=baseline_value, linestyle='--', color=baseline_colors[idx], alpha=0.8, label=f"Baseline ({syn})")

        ax.set_title(metric, fontsize=12)
        if i == rows - 1:  # Only add x-axis label on the last subplot
            ax.set_xlabel("Number of Components (n_components)", fontsize=10)
        ax.set_ylabel("Time (s)", fontsize=10)
        ax.grid(False)

    # Add a legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    baseline_handles = [plt.Line2D([0], [0], color=baseline_colors[idx], linestyle='--', label=f"Baseline ({syn})")
                        for idx, syn in enumerate(baselines.keys())]
    handles += baseline_handles
    labels += [f"Baseline ({syn})" for syn in baselines.keys()]

    fig.legend(handles, labels, title="Legend", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4)

    # Add an overall title and adjust layout
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_time_with_consistent_layout.pdf"))
    plt.close()

# Call the function to generate time metrics
plot_experiment5_time_with_consistent_layout(
    exp5_data,
    "Time Metrics with Consistent Layout",
    "experiment5_time_consistent"
)

# Function to plot Experiment 5 quality metrics with a horizontal dashed line
def plot_experiment5_quality_with_baselines(data, title_prefix, file_name_prefix):
    """
    Plot Experiment 5 quality metrics with two baselines, one for each synthesizer, and distinguish them by color.
    """
    # Define quality metrics
    quality_metrics = ["Overall Score", "Column Shapes", "Column Pair Trends",
                       "Boundary Adherence", "Range Coverage", "KS Complement",
                       "TV Complement", "Correlation Similarity", "Contingency Similarity"]

    # Extract baseline values for each synthesizer
    baselines = {}
    for syn in ['TVAESynthesizer', 'REaLTabFormer']:
        baseline = baseline_data[baseline_data['Synthesizer Name'] == syn]
        assert len(baseline) == 1, f"Baseline data for synthesizer {syn} should have exactly one row!"
        baselines[syn] = baseline.iloc[0].to_dict()

    # Plot the metrics
    rows, cols = 3, 3
    total_subplots = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten()

    for i, metric in enumerate(quality_metrics):
        if i >= total_subplots:
            break
        ax = axes_flat[i]

        # Plot the quality metrics for each synthesizer
        for dec_i, syn in enumerate(synthesizers):
            sub_data = data[data['Synthesizer Name'] == syn]
            y_values = [sub_data[sub_data['n_components'] == c][metric].mean() for c in n_components]
            ax.plot(n_components, y_values, marker='o', color=colors[dec_i], label=syn)

        # Add baselines for each synthesizer
        for idx, (syn, baseline_values) in enumerate(baselines.items()):
            baseline_value = baseline_values[metric]
            ax.axhline(y=baseline_value, linestyle='--', color=baseline_colors[idx], alpha=0.8, label=f"Baseline ({syn})")

        ax.set_title(metric, fontsize=10)
        if (i // cols) == rows - 1:
            ax.set_xlabel("Number of Components (n_components)", fontsize=10)
        if (i % cols) == 0:
            ax.set_ylabel("Quality", fontsize=10)
        ax.grid(False)

    for j in range(len(quality_metrics), total_subplots):
        axes_flat[j].axis('off')

    # Add a legend
    #handles, labels = axes_flat[0].get_legend_handles_labels()
    handles, labels = ax.get_legend_handles_labels()
    # Create custom handles for baselines
    baseline_handles = [plt.Line2D([0], [0], color=baseline_colors[idx], linestyle='--', label=f"Baseline ({syn})") 
                        for idx, syn in enumerate(baselines.keys())]
    # Combine original handles with baseline handles
    #handles += baseline_handles
    #labels += [f"Baseline ({syn})" for syn in baselines.keys()]

    fig.legend(handles, labels, title="Legend", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4)

    # Add an overall title and adjust layout
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_quality_with_baselines.pdf"))
    plt.close()

# Call the updated function
plot_experiment5_quality_with_baselines(exp5_data, 
                                        "Quality Metrics with Two Baselines", 
                                        "experiment5_quality_with_two_baselines")

print("Experiment 5 quality metrics plot with baseline has been generated and saved.")

# ---------------------- Plot aggregated metrics ----------------------
def plot_4x1_with_metrics_exp5(data, title_prefix, file_name_prefix):
    """
    Plot a 4x1 layout for Experiment 5 metrics with two baselines for each metric.
    """
    # Define the metrics to be plotted
    metrics = ["Metrics Overall Score", "Fitting Time", "Sampling Time", "Decomposition Time"]

    # Extract baselines for each synthesizer
    baselines = {}
    for syn in ['TVAESynthesizer', 'REaLTabFormer']:
        baseline = baseline_data[baseline_data['Synthesizer Name'] == syn]
        assert len(baseline) == 1, f"Baseline data for synthesizer {syn} should have exactly one row!"
        baselines[syn] = baseline.iloc[0].to_dict()

    # Create the layout for 4x1 plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sns.set_palette("GnBu_d")

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for dec_i, syn in enumerate(synthesizers):
            sub_data = data[data['Synthesizer Name'] == syn]
            y_values = [sub_data[sub_data['n_components'] == c][metric].mean() for c in n_components]
            ax.plot(n_components, y_values, marker='o', color=colors[dec_i], label=syn)

        # Add baselines for each synthesizer
        for idx, (syn, baseline_values) in enumerate(baselines.items()):
            baseline_value = baseline_values[metric]
            ax.axhline(y=baseline_value, linestyle='--', color=baseline_colors[idx], alpha=0.8, label=f"Baseline ({syn})")

        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("Number of Components (n_components)", fontsize=10)
        if i == 0:
            ax.set_ylabel(metric, fontsize=10)
        ax.grid(False)

    # Add a legend for the first plot only
    #handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Legend", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4)

    # Add an overall title and adjust layout
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_4x1_with_baselines.pdf"))
    plt.close()

# Function to generate 4x1 layout for Experiment 1
def plot_4x1_with_metrics_exp1(data, title_prefix, file_name_prefix):
    # Define the metrics to be plotted
    metrics = ["Metrics Overall Score", "Fitting Time", "Sampling Time", "Decomposition Time"]

    # Aggregated data for Experiment 1
    aggregated_data = data.groupby(["Synthesizer Name", "Parsed Decomposer Name"]).mean(numeric_only=True).reset_index()

    # Unique synthesizers and decomposers
    synthesizers = sorted(aggregated_data['Synthesizer Name'].unique())
    decomposers = sorted(aggregated_data['Parsed Decomposer Name'].unique())
    x = np.arange(len(synthesizers))  # x-axis positions

    # Create the layout for 4x1 plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5))
    sns.set_palette("GnBu_d")

    for i, metric in enumerate(metrics):
        ax = axes[i]
        width = 0.8 / len(decomposers)  # Bar width for clustered bars

        # Plot metrics for each decomposer
        for j, dec in enumerate(decomposers):
            values = aggregated_data[aggregated_data['Parsed Decomposer Name'] == dec][metric].values
            ax.bar(x + j * width, values, width, color=colors[j], label=dec)

        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x + width * (len(decomposers) - 1) / 2)
        ax.set_xticklabels(synthesizers, rotation=45, ha='right', fontsize=10)
        #ax.set_xlabel("Synthesizer Name", fontsize=10)
        ax.set_ylabel(metric if metric == "Metrics Overall Score" else "Time (s)", fontsize=10)
        #ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Decomposer", loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(decomposers))

    # Add an overall title and adjust layout
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_4x1_layout.pdf"))
    plt.close()

# Generate the 4x1 layout plot for Experiment 1
plot_4x1_with_metrics_exp1(exp1_data, "Aggregated Metrics", "experiment1_4x1")

# Generate 4x1 layout for Experiment 5
plot_4x1_with_metrics_exp5(exp5_data, 
                           "Aggregated Metrics", 
                           "experiment5_4x1_with_two_baselines")

print("4x1 metrics layout plots for Experiments 1 and 5 have been generated and saved.")
