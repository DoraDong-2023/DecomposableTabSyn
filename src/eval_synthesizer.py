import sdmetrics
import fire
import time
import json
import numpy as np
import pandas as pd
import prettytable as pt
from functools import partial
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from pathlib import Path
from sdv.datasets.demo import download_demo, get_available_demos
from synthesizers import TableSynthesizer, DecompositionSynthesizer, decomposier_name_to_class
from utils import plot_model_scaling

# https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary
from sdmetrics.single_table import (
    BoundaryAdherence, KSComplement, TVComplement, TableStructure, NewRowSynthesis
)
from sdmetrics.single_column import (
    StatisticSimilarity, SequenceLengthSimilarity, RangeCoverage, MissingValueSimilarity, KeyUniqueness, CategoryCoverage
)
from sdmetrics.column_pairs import (
    ReferentialIntegrity, CorrelationSimilarity, ContingencySimilarity
)
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport

def evaluate_single_table_metrics(real_data, synthetic_data, metadata, save_path="results"):
    """
    Evaluate various single-table metrics for real and synthetic data.
    """
    
    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, list(metadata.tables.values())[0].to_dict())
    quality_report.save(save_path / "quality_report.pkl")
    score = quality_report.get_score()
    properties = quality_report.get_properties() # dataframe
    property_score = {k: v for k, v in zip(properties['Property'], properties['Score'])}
    details = defaultdict(dict) # [metric_name][column_name] = score
    (save_path / "quality_report").mkdir(parents=True, exist_ok=True)
    for k in properties['Property']:
        assert k in ['Column Shapes', 'Column Pair Trends'], f"Invalid property name: {k}" 
        cur_details = quality_report.get_details(property_name=k)
        for index, row in cur_details.iterrows():
            if k == 'Column Shapes':
                metric_name = row['Metric']
                column_name = row['Column']
                score = row['Score']
                details[metric_name][column_name] = score
            else:
                metric_name = row['Metric']
                column1 = row['Column 1']
                column2 = row['Column 2']
                score = row['Score']
                if score is not None and not np.isnan(score):
                    details[metric_name][f"{column1} - {column2}"] = score
        fig = quality_report.get_visualization(property_name=k)
        fig.write_image(save_path / "quality_report" / f"{k}.png")
    metrics = {
        "Overall Score": score,
        "Properties": property_score,
        "Details": details
    }
    
    # Align columns
    common_columns = real_data.columns.intersection(synthetic_data.columns)
    real_data = real_data[common_columns]
    synthetic_data = synthetic_data[common_columns]

    metrics = {}

    # Single-column metrics
    for column in common_columns:
        # Convert Series to DataFrame
        real_col = real_data[column]
        synthetic_col = synthetic_data[column]

        if pd.api.types.is_numeric_dtype(real_data[column]):
            try:
                details["BoundaryAdherence"][column] = BoundaryAdherence.compute(real_col.to_frame(), synthetic_col.to_frame())
            except Exception as e:
                details["BoundaryAdherence"][column] = None
            try:
                details["RangeCoverage"][column] = RangeCoverage.compute(real_col, synthetic_col)
            except Exception as e:
                details["RangeCoverage"][column] = None
        else:
            details["CategoryCoverage"][column] = CategoryCoverage.compute(real_col, synthetic_col)
        # KSComplement and TVComplement are computed in the quality report
    # Column-pairs metrics
    if len(common_columns) > 1:
        for col1 in common_columns:
            for col2 in common_columns:
                if col1 == col2:
                    continue
                try:
                    score = StatisticSimilarity.compute(
                        real_data[col1],
                        synthetic_data[col1],
                    )
                except Exception as e:
                    score = None
                if score is not None and not np.isnan(score):
                    details['StatisticalSimilarity'][f"{col1} - {col2}"] = score
        # CorrelationSimilarity and ContingencySimilarity are computed in the quality report
        
    metrics = {}
    metrics_names = ["BoundaryAdherence", "RangeCoverage", "CategoryCoverage", "KSComplement", "TVComplement", "StatisticSimilarity", "CorrelationSimilarity", "ContingencySimilarity"]
    for metric_name in metrics_names:
        if metric_name in details:
            metrics[metric_name] = np.mean([v for v in details[metric_name].values() if v is not None and not np.isnan(v)])
        else:
            metrics[metric_name] = None
    # table-wide metrics
    metrics['TableStructure'] = TableStructure.compute(real_data, synthetic_data)
    metrics['NewRowSynthesis'] = NewRowSynthesis.compute(real_data, synthetic_data, list(metadata.tables.values())[0].to_dict())
    
    summary = {
        "Overall Score": score,
        "Properties": property_score,
        "Metrics": metrics,
        "Metrics Overall Score": np.mean([v for v in metrics.values() if v is not None and v is not np.nan]),
        "Details": details,
    }
    print("---" * 20)
    print("Quality Report:")
    print(f" - Overall Score: {score}")
    print(" - Properties:")
    for k, v in property_score.items():
        print(f"   - {k}: {v}")
    print(" - Metrics:")
    for k, v in metrics.items():
        print(f"   - {k}: {v}")
    print(" - Metrics Overall Score:", summary["Metrics Overall Score"])
    return summary

def is_constant_column(column):
    """
    Check whether columns are constant
    """
    return column.nunique() <= 1

def train_and_evaluate_synthesizer(synthesizer, real_data, metadata, num_samples, save_path):
    start_fit = time.time()
    synthesizer.fit(real_data)
    fit_time = time.time() - start_fit
    if isinstance(synthesizer, DecompositionSynthesizer):
        print(f"Decomposing and fitting time: {fit_time:.2f}s")
        decomposition_time = synthesizer.decomposition_time
        fit_time = synthesizer.total_fit_time
    else:
        decomposition_time = 0
        fit_time = fit_time
    print(f"Decomposition Time taken: {decomposition_time:.2f}s")
    print(f"Fitting Time taken: {fit_time:.2f}s")
    
    start_sample = time.time()
    synthetic_data = synthesizer.sample(num_samples)
    sample_time = time.time() - start_sample
    print(f"Sampling Time taken: {sample_time:.2f}s")
    
    metrics = evaluate_single_table_metrics(real_data, synthetic_data, metadata, save_path)
    return metrics, fit_time, sample_time, decomposition_time

def benchmark_single_synthesizer(datasets, new_synthesizer_fn, num_samples=1000, row_sizes=[1000], save_path="results", seed=42):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    for dataset_name in tqdm(datasets, desc=f"Running experiments on {datasets}"):
        dataset_save_path = save_path / dataset_name
        dataset_save_path.mkdir(parents=True, exist_ok=True)
        # Load dataset
        real_data, metadata = download_demo(modality='single_table', dataset_name=dataset_name)
        num_columns = len(real_data.columns)
        print(f"Running experiments on {dataset_name} dataset ...")
        dataset_results = []
        for num_rows in row_sizes:
            print(seed)
            data = real_data.sample(num_rows, random_state=seed)
            print(f"Using {num_rows} rows from dataset {dataset_name}")
            
            # Initialize synthesizer
            synthesizer = new_synthesizer_fn(metadata)
            # Train and evaluate
            metrics, fit_time, sample_time, decomposition_time = train_and_evaluate_synthesizer(synthesizer, data, metadata, num_samples, dataset_save_path)
            
            # Record results
            row = {
                "Dataset": dataset_name,
                "Num Rows": num_rows,
                "Synthesizer": synthesizer.synthesizer_name,
                "Fit Time (s)": fit_time,
                "Sample Time (s)": sample_time,
                "Decomposition Time (s)": decomposition_time,
            }
            row.update(metrics)  # Add metrics to the row
            dataset_results.append(row)
        # Convert results to DataFrame
        # save results to 
        with open(dataset_save_path / "results.json", "w") as f:
            json.dump(dataset_results, f, indent=4)
            print(f"Results on {dataset_name} saved to {dataset_save_path}")

def get_synthesizer(meta_data, synthesizer_name, decomposer_name=None, num_train_epochs=1, default_n_components=4, row_fraction=0.5, col_fraction=0.5, nf_level=3):
    if decomposer_name:
        assert decomposer_name in decomposier_name_to_class, f"Invalid decomposer name. Available decomposer names are: {decomposier_name_to_class.keys()}"
        synthesizer_init_kwargs = {"epochs": num_train_epochs}
        decomposer_init_kwargs = {}
        decomposer_split_kwargs = {}
        if decomposer_name in ["NMFDecomposition", "PCADecomposition", "SVDDecomposition", "ICADecomposition", "FactorAnalysisDecomposition", "DictionaryLearningDecomposition"]:
            decomposer_init_kwargs["n_components"] = default_n_components
        elif decomposer_name == "TruncateDecomposition":
            decomposer_init_kwargs["row_fraction"] = row_fraction
            decomposer_init_kwargs["col_fraction"] = col_fraction
        elif decomposer_name == "NFDecomposition":
            decomposer_split_kwargs = {"nf_level": nf_level}
        
        synthesizer = DecompositionSynthesizer(meta_data, synthesizer_name, decomposer_name, synthesizer_init_kwargs, decomposer_init_kwargs, decomposer_split_kwargs)
    else:
        synthesizer_init_kwargs = {"epochs": num_train_epochs}
        synthesizer = TableSynthesizer(synthesizer_name, meta_data, synthesizer_config=synthesizer_init_kwargs)
    return synthesizer

def main(
    dataset_name: str,
    synthesizer_name: str,
    decomposer_name: str = None,
    num_train_samples: int = 1000,
    num_test_samples: int = 1000,
    num_train_epochs: int = 1,
    seed: int = 42,
    default_n_components: int = 8,
    row_fraction: float = 0.5,
    col_fraction: float = 0.5,
    nf_level: int = 3,
):
    dataset_names = dataset_name
    available_dataset_names = get_available_demos(modality='single_table')['dataset_name'].tolist()
    assert all(dataset_name in available_dataset_names for dataset_name in dataset_names), \
        f"Invalid dataset name {dataset_name}. Available dataset names are: {available_dataset_names}"
    if decomposer_name == "no_decomposition":
        decomposer_name = None
    get_new_synthesizer = partial(
        get_synthesizer,
        synthesizer_name=synthesizer_name,
        decomposer_name=decomposer_name,
        num_train_epochs=num_train_epochs,
        default_n_components=default_n_components,
        row_fraction=row_fraction,
        col_fraction=col_fraction,
        nf_level=nf_level,
    )
        
    to_save_decomposition_name = decomposer_name if decomposer_name else "no_decomposition"
    if to_save_decomposition_name == "TruncateDecomposition":
        to_save_decomposition_name += f"_rf_{row_fraction}_cf_{col_fraction}"
    elif to_save_decomposition_name == "NFDecomposition":
        to_save_decomposition_name += f"_nf_{nf_level}"
    else:
        if to_save_decomposition_name in ["NMFDecomposition", "PCADecomposition", "SVDDecomposition", "ICADecomposition", "FactorAnalysisDecomposition", "DictionaryLearningDecomposition"]:
            to_save_decomposition_name += f"_n_{default_n_components}"

    save_path = Path("benchmark_results") / (to_save_decomposition_name) / synthesizer_name
    save_path.mkdir(parents=True, exist_ok=True)
    benchmark_single_synthesizer(
        datasets=dataset_names,
        new_synthesizer_fn=get_new_synthesizer,
        num_samples=num_test_samples,
        row_sizes=[num_train_samples],
        save_path=save_path,
        seed=seed,
    )
    
    
if __name__ == '__main__':
    fire.Fire(main)
    
    
"""
python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 1 --seed 42
"""