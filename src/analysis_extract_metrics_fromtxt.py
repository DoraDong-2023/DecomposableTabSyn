import os
import re
import pandas as pd

def extract_experiment_metrics(file_path):
    """
    Extract metrics, settings, and time from the experiment log file. Includes error handling.

    :param file_path: Path to the log file.
    :return: DataFrame with extracted data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content into experiment blocks
    experiment_blocks = re.split(r"Running experiment with", content)
    experiment_data = []

    for block in experiment_blocks[1:]:  # Skip the first part (not an experiment block)
        # Initialize a dictionary to store the experiment result
        result = {
            "Dataset Name": None,
            "Synthesizer Name": None,
            "Num Train Samples": None,
            "Num Test Samples": None,
            "Num Train Epochs": None,
            "Seed": None,
            "Decomposer Name": None,
            "Fitting Time": None,
            "Decomposition Time": None,
            "Sampling Time": None,
            "Overall Score": None,
            "Column Shapes": None,
            "Column Pair Trends": None,
            "Boundary Adherence": None,
            "Range Coverage": None,
            "Category Coverage": None,
            "KS Complement": None,
            "TV Complement": None,
            "Statistic Similarity": None,
            "Correlation Similarity": None,
            "Contingency Similarity": None,
            "Table Structure": None,
            "New Row Synthesis": None,
            "Metrics Overall Score": None,
            "Status": "Success"
        }

        # Extract settings
        dataset_match = re.search(r"dataset_name: (\w+)", block)
        synthesizer_match = re.search(r"synthesizer_name: (\w+)", block)
        train_samples_match = re.search(r"num_train_samples: (\d+)", block)
        test_samples_match = re.search(r"num_test_samples: (\d+)", block)
        train_epochs_match = re.search(r"num_train_epochs: (\d+)", block)
        seed_match = re.search(r"seed: (\d+)", block)
        decomposer_match = re.search(r"decomposer_name: (\w+)", block)

        result["Dataset Name"] = dataset_match.group(1) if dataset_match else None
        result["Synthesizer Name"] = synthesizer_match.group(1) if synthesizer_match else None
        result["Num Train Samples"] = int(train_samples_match.group(1)) if train_samples_match else None
        result["Num Test Samples"] = int(test_samples_match.group(1)) if test_samples_match else None
        result["Num Train Epochs"] = int(train_epochs_match.group(1)) if train_epochs_match else None
        result["Seed"] = int(seed_match.group(1)) if seed_match else None
        result["Decomposer Name"] = decomposer_match.group(1) if decomposer_match else None

        # Extract time-related metrics
        fit_time_match = re.search(r"Fitting Time taken: ([\d.]+)s", block)
        decomp_time_match = re.search(r"Decomposition Time taken: ([\d.]+)s", block)
        sample_time_match = re.search(r"Sampling Time taken: ([\d.]+)s", block)

        result["Fitting Time"] = float(fit_time_match.group(1)) if fit_time_match else None
        result["Decomposition Time"] = float(decomp_time_match.group(1)) if decomp_time_match else None
        result["Sampling Time"] = float(sample_time_match.group(1)) if sample_time_match else None

        # Extract quality report metrics
        overall_score_match = re.search(r"Overall Score: ([\d.]+)", block)
        col_shapes_match = re.search(r"Column Shapes: ([\d.]+)", block)
        col_pair_trends_match = re.search(r"Column Pair Trends: ([\d.]+)", block)

        result["Overall Score"] = float(overall_score_match.group(1)) if overall_score_match else None
        result["Column Shapes"] = float(col_shapes_match.group(1)) if col_shapes_match else None
        result["Column Pair Trends"] = float(col_pair_trends_match.group(1)) if col_pair_trends_match else None

        # Extract detailed metrics
        metrics = {
            "Boundary Adherence": r"BoundaryAdherence: ([\d.]+)",
            "Range Coverage": r"RangeCoverage: ([\d.]+)",
            "Category Coverage": r"CategoryCoverage: (\w+)",
            "KS Complement": r"KSComplement: ([\d.]+)",
            "TV Complement": r"TVComplement: ([\d.]+)",
            "Statistic Similarity": r"StatisticSimilarity: (\w+)",
            "Correlation Similarity": r"CorrelationSimilarity: ([\d.]+)",
            "Contingency Similarity": r"ContingencySimilarity: ([\d.]+)",
            "Table Structure": r"TableStructure: ([\d.]+)",
            "New Row Synthesis": r"NewRowSynthesis: ([\d.]+)",
            "Metrics Overall Score": r"Metrics Overall Score: ([\d.]+)"
        }

        for key, pattern in metrics.items():
            match = re.search(pattern, block)
            result[key] = float(match.group(1)) if match and match.group(1) != "None" else None

        # Check for errors
        if "Traceback" in block:
            result["Status"] = "Failed"

        experiment_data.append(result)

    # Convert results to DataFrame
    return pd.DataFrame(experiment_data)

# Example usage
file_path = "eval_synthesizers.txt"  # Replace with your file path
df = extract_experiment_metrics(file_path)

# Save DataFrame to CSV
output_csv_path = "experiment_metrics.csv"
df.to_csv(output_csv_path, index=False)
print(f"Extracted data saved to {output_csv_path}")
