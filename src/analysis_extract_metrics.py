import os
import json
import pandas as pd
import re


def safe_extract_n_components(decomposer):
    """Extract numeric components from decomposer if present."""
    match = re.search(r"_n_(\d+)", decomposer)
    return int(match.group(1)) if match else None


def parse_decomposer(decomposer):
    """
    Parse decomposer to a normalized form and extract key components.
    """
    if decomposer == "no_decomposition":
        return {"base": "no_decomposition", "n_components": None}
    if "_" in decomposer:
        base, *details = decomposer.split('_')
        n_components = safe_extract_n_components(decomposer)
        return {"base": base, "n_components": n_components}
    return {"base": decomposer, "n_components": None}


def is_experiment_1(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples):
    """Experiment 1: Fixed conditions."""
    return (
        dataset == "covtype" and
        synthesizer in [
            "CTGANSynthesizer", "TVAESynthesizer", "CopulaGANSynthesizer", "GaussianCopulaSynthesizer",
            "REaLTabFormer", "Tabula"
        ] and
        parsed_decomposer["base"] in [
            "no_decomposition", "PCADecomposition", "SVDDecomposition", "FactorAnalysisDecomposition",
            "ICADecomposition", "TruncateDecomposition"
        ] and
        (
            parsed_decomposer["base"] != "PCADecomposition" or 
            parsed_decomposer["n_components"] == 8  # Only check for PCADecomposition
        ) and
        num_train_samples == 5000 and num_test_samples == 2000
    )


def is_experiment_2(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples):
    """Experiment 2: Train samples vary."""
    return (
        dataset == "covtype" and
        synthesizer in ["TVAESynthesizer", "REaLTabFormer"] and
        parsed_decomposer["base"] in ["no_decomposition", "PCADecomposition", "TruncateDecomposition"] and
        (
            parsed_decomposer["base"] != "PCADecomposition" or
            parsed_decomposer["n_components"] == 8  # Only check for PCADecomposition
        ) and
        num_test_samples == 2000
    )


def is_experiment_3(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples):
    """Experiment 3: Test samples vary."""
    return (
        dataset == "covtype" and
        synthesizer in ["TVAESynthesizer", "REaLTabFormer"] and
        parsed_decomposer["base"] in ["no_decomposition", "PCADecomposition", "TruncateDecomposition"] and
        (
            parsed_decomposer["base"] != "PCADecomposition" or
            parsed_decomposer["n_components"] == 8  # Only check for PCADecomposition
        ) and
        num_train_samples == 5000
    )


def is_experiment_4(dataset, synthesizer, parsed_decomposer):
    """Experiment 4: Different datasets."""
    return (
        dataset in ["asia", "adult", "insurance", "alarm", "covtype", "mnist12"] and
        synthesizer in ["TVAESynthesizer", "REaLTabFormer"] and
        parsed_decomposer["base"] in ["no_decomposition", "PCADecomposition"] and
        (
            parsed_decomposer["base"] != "PCADecomposition" or
            parsed_decomposer["n_components"] == 8  # Only check for PCADecomposition
        )
    )


def is_experiment_5(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples):
    """Experiment 5: Allow PCADecomposition with all n_components."""
    return (
        dataset == "covtype" and
        synthesizer in ["TVAESynthesizer", "REaLTabFormer"] and
        parsed_decomposer["base"] == "PCADecomposition" and num_train_samples == 5000 and num_test_samples == 2000
    )


def extract_results_with_labels(base_dir, output_csv):
    """
    Extract metrics from results.json files and assign experiment labels.
    """
    results = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "results.json":
                file_path = os.path.join(root, file)

                path_parts = root.split(os.sep)
                num_train_samples = None
                num_test_samples = None
                last_part = path_parts[-1]
                match = re.match(r"(\d+)_train_(\d+)_test", last_part)
                if match:
                    num_train_samples = int(match.group(1))
                    num_test_samples = int(match.group(2))
                    # 如果匹配到该模式，说明最后一层是train/test信息，
                    # 那么 dataset = path_parts[-2]， synthesizer = path_parts[-3], decomposer = path_parts[-4]
                    if len(path_parts) >= 4:
                        dataset = path_parts[-2]
                        synthesizer = path_parts[-3]
                        decomposer = path_parts[-4]
                    else:
                        # 如果路径不符合预期，fallback
                        dataset = "Unknown"
                        synthesizer = "Unknown"
                        decomposer = "Unknown"
                else:
                    # 否则维持原有逻辑
                    decomposer = path_parts[-3] if len(path_parts) > 2 else "Unknown"
                    synthesizer = path_parts[-2] if len(path_parts) > 1 else "Unknown"
                    dataset = path_parts[-1] if len(path_parts) > 0 else "Unknown"

                parsed_decomposer = parse_decomposer(decomposer)

                with open(file_path, 'r') as f:
                    try:
                        entry = json.load(f)
                        data = entry
                        #for entry in data:
                        if True:
                            num_train_samples = entry.get("Num Rows", 5000)
                            num_test_samples = entry.get("Num Test Samples", 2000)

                            # Handle exclusion based on PCADecomposition and n_components logic
                            #print(dataset, synthesizer, parsed_decomposer)
                            if parsed_decomposer["base"] == "PCADecomposition" and parsed_decomposer["n_components"] != 8:
                                if not is_experiment_5(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples):
                                    continue
                            result = {
                                "Dataset Name": dataset,
                                "Synthesizer Name": synthesizer,
                                "Decomposer Name": decomposer,
                                "Parsed Decomposer": parsed_decomposer,
                                "Parsed Decomposer Name": parsed_decomposer['base'],
                                "Num Train Samples": num_train_samples,
                                "Num Test Samples": num_test_samples,
                                "Num Train Epochs": entry.get("Num Train Epochs", 5),
                                "Overall Score": entry.get("Overall Score", None),
                                "Fitting Time": entry.get("Fit Time (s)", None),
                                "Decomposition Time": entry.get("Decomposition Time (s)", None),
                                "Sampling Time": entry.get("Sample Time (s)", None),
                                "Column Shapes": entry.get("Properties", {}).get("Column Shapes", None),
                                "Column Pair Trends": entry.get("Properties", {}).get("Column Pair Trends", None),
                                "Boundary Adherence": entry.get("Metrics", {}).get("BoundaryAdherence", None),
                                "Range Coverage": entry.get("Metrics", {}).get("RangeCoverage", None),
                                "Category Coverage": entry.get("Metrics", {}).get("CategoryCoverage", None),
                                "KS Complement": entry.get("Metrics", {}).get("KSComplement", None),
                                "TV Complement": entry.get("Metrics", {}).get("TVComplement", None),
                                "Statistic Similarity": entry.get("Metrics", {}).get("StatisticSimilarity", None),
                                "Correlation Similarity": entry.get("Metrics", {}).get("CorrelationSimilarity", None),
                                "Contingency Similarity": entry.get("Metrics", {}).get("ContingencySimilarity", None),
                                "Table Structure": entry.get("Metrics", {}).get("TableStructure", None),
                                "New Row Synthesis": entry.get("Metrics", {}).get("NewRowSynthesis", None),
                                "Metrics Overall Score": entry.get("Metrics Overall Score", None),
                                "Experiment1_YN": "Y" if is_experiment_1(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples) else "N",
                                "Experiment2_YN": "Y" if is_experiment_2(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples) else "N",
                                "Experiment3_YN": "Y" if is_experiment_3(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples) else "N",
                                "Experiment4_YN": "Y" if is_experiment_4(dataset, synthesizer, parsed_decomposer) else "N",
                                "Experiment5_YN": "Y" if is_experiment_5(dataset, synthesizer, parsed_decomposer, num_train_samples, num_test_samples) else "N",
                                "Path": file_path,
                            }
                            results.append(result)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage
base_directory = "benchmark_results"
output_csv_path = "experiment_metrics_with_labels.csv"
extract_results_with_labels(base_directory, output_csv_path)
