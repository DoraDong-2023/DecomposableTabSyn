from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import time, json
import prettytable as pt
from tqdm import tqdm
from pathlib import Path
from sdv.datasets.demo import download_demo, get_available_demos
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
from sdv.datasets.demo import download_demo
from sdmetrics.reports.single_table import QualityReport

class BaseSynthesis:
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(self, num_rows: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")

class TableSynthesizer(BaseSynthesis):
    synthesizer_configs = {
        "GaussianCopulaSynthesizer": {},
        "CTGANSynthesizer": {
            "epochs": 1
        },
        "TVAESynthesizer": {
            "epochs": 1
        },
        "CopulaGANSynthesizer": {
            "epochs": 1    
        },
    }
    def __init__(self, synthesizer_name, metadata, filepath=None):
        assert synthesizer_name in self.synthesizer_configs, f"Invalid synthesizer name: {synthesizer_name}"
        synthesizer_config = self.synthesizer_configs[synthesizer_name]
        synthesizer_class = globals()[synthesizer_name]
        self.synthesizer_name = synthesizer_name
        self.synthesizer_config = synthesizer_config
        if filepath:
            self.synthesizer = synthesizer_class.load(filepath)
        else:
            self.synthesizer = synthesizer_class(metadata, **synthesizer_config)
    def fit(self, data: pd.DataFrame):
        self.synthesizer.fit(data)
    def sample(self, num_rows: int) -> pd.DataFrame:
        return self.synthesizer.sample(num_rows)

from decomposition import TruncateDecomposition, NMFDecomposition, PCADecomposition, SVDDecomposition, ICADecomposition, FactorAnalysisDecomposition, DictionaryLearningDecomposition
from sdv.metadata import SingleTableMetadata
class DecompositionSynthesizer(BaseSynthesis):
    def __init__(self, metadata, synthesizer_name="GaussianCopulaSynthesizer", row_fraction=0.5, col_fraction=0.5, decomposer=None, **decomposer_kwargs):
        """
        Initialize with metadata and decomposition parameters.
        """
        self.metadata = metadata
        self.row_fraction = row_fraction
        self.col_fraction = col_fraction
        if not decomposer:
            self.decomposer = TruncateDecomposition(row_fraction, col_fraction)
        else:
            #self.decomposer = NMFDecomposition(n_components=2)
            self.decomposer = decomposer(**decomposer_kwargs)
        self.synthesizers = []
        self.synthesizer_name = synthesizer_name
        self.synthesizer_config = TableSynthesizer.synthesizer_configs.get(synthesizer_name, {})
        self.synthesizer_class = globals()[synthesizer_name]

    def _update_metadata(self, data_part: pd.DataFrame) -> SingleTableMetadata:
        """
        Create a new metadata object for each decomposed part based on its columns.
        """
        part_metadata = SingleTableMetadata()
        part_metadata.detect_from_dataframe(data_part)
        return part_metadata
    
    def fit(self, data: pd.DataFrame):
        """
        Split the data using decomposition and fit individual synthesizers on each piece.
        """
        start_time = time.time()
        self.data_parts = self.decomposer.split(data)
        decomposition_time = time.time() - start_time
        print(f"Data decomposition completed in {decomposition_time:.2f} seconds.")

        self.synthesizers = []
        fit_times = []
        for i, part in enumerate(self.data_parts):
            part_metadata = self._update_metadata(part)
            synthesizer = self.synthesizer_class(part_metadata, **self.synthesizer_config)
            start_time = time.time()
            synthesizer.fit(part)
            fit_time = time.time() - start_time
            self.synthesizers.append(synthesizer)
            fit_times.append(fit_time)
        total_fit_time = sum(fit_times)
        print(f"Total fit time for all parts: {total_fit_time:.2f} seconds.")
        self.decomposition_time = decomposition_time
        self.fit_times = fit_times
        self.total_fit_time = total_fit_time

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Sample data from each fitted synthesizer and join the samples back together.
        """
        sampled_parts = []
        num_samples_per_part = num_rows // len(self.synthesizers)
        sample_times = []

        for i, synthesizer in enumerate(self.synthesizers):
            start_time = time.time()
            sampled_part = synthesizer.sample(num_samples_per_part)
            sample_time = time.time() - start_time
            #print(f"Sample time for part {i + 1}: {sample_time:.2f} seconds.")
            sampled_parts.append(sampled_part)
            sample_times.append(sample_time)
        total_sample_time = sum(sample_times)
        #print(f"Total sample time for all parts: {total_sample_time:.2f} seconds.")

        # Join the sampled parts using the decomposition join method
        start_time = time.time()
        #print("Joining sampled parts...")
        joined_data = self.decomposer.join(sampled_parts)
        join_time = time.time() - start_time
        #print(f"Join completed in {join_time:.2f} seconds.")
        self.sample_times = sample_times
        self.total_sample_time = total_sample_time
        self.join_time = join_time
        return joined_data
    def get_timing_details(self):
        """
        Return a summary of the time costs for decomposition, fitting, sampling, and joining.
        """
        return {
            "decomposition_time": self.decomposition_time,
            "fit_times": self.fit_times,
            "total_fit_time": self.total_fit_time,
            "sample_times": self.sample_times,
            "total_sample_time": self.total_sample_time,
            "join_time": self.join_time,
        }


def evaluate_single_table_metrics(real_data, synthetic_data, metadata):
    """
    Evaluate various single-table metrics for real and synthetic data.
    """
    # Align columns
    common_columns = real_data.columns.intersection(synthetic_data.columns)
    real_data = real_data[common_columns]
    synthetic_data = synthetic_data[common_columns]

    metrics = {}

    # Single-column metrics
    for column in common_columns:
        real_col = real_data[[column]]  # Convert Series to DataFrame
        synthetic_col = synthetic_data[[column]]  # Convert Series to DataFrame

        if pd.api.types.is_numeric_dtype(real_data[column]):
            metrics[f'{column}_BoundaryAdherence'] = BoundaryAdherence.compute(real_col, synthetic_col)
            metrics[f'{column}_KSComplement'] = KSComplement.compute(real_col, synthetic_col)
            #metrics[f'{column}_RangeCoverage'] = RangeCoverage.compute(real_col, synthetic_col)
        else:
            metrics[f'{column}_CategoryCoverage'] = CategoryCoverage.compute(real_col, synthetic_col)

    # Column-pairs metrics
    if len(common_columns) > 1:
        for col1, col2 in zip(common_columns[:-1], common_columns[1:]):
            """if is_constant_column(real_data[col1]) or is_constant_column(real_data[col2]) or \
                is_constant_column(synthetic_data[col1]) or is_constant_column(synthetic_data[col2]):
                    #print(f"Skipping CorrelationSimilarity for constant column pair: {col1}, {col2}")
                    #continue
                    pass"""
            metrics[f'{col1}_{col2}_CorrelationSimilarity'] = CorrelationSimilarity.compute(
                real_data[[col1, col2]],
                synthetic_data[[col1, col2]],
                coefficient='Pearson'
            )
    # Table-wide metrics
    metrics['TableStructure'] = TableStructure.compute(real_data, synthetic_data)
    #metrics['NewRowSynthesis'] = NewRowSynthesis.compute(real_data, synthetic_data, metadata)

    return metrics
def is_constant_column(column):
    """
    Check whether columns are constant
    """
    return column.nunique() <= 1

def train_and_evaluate_synthesizer(synthesizer, real_data, metadata, num_samples):
    start_fit = time.time()
    synthesizer.fit(real_data)
    fit_time = time.time() - start_fit
    print(f"Fitting Time taken: {fit_time:.2f}s")
    start_sample = time.time()
    synthetic_data = synthesizer.sample(num_samples)
    sample_time = time.time() - start_sample
    print(f"Sampling Time taken: {sample_time:.2f}s")
    metrics = evaluate_single_table_metrics(real_data, synthetic_data, metadata)
    return metrics, fit_time, sample_time
def benchmark_datasets_and_synthesizers(datasets, synthesizers, num_samples=1000, row_sizes=[1000], save_path="results", numerical_only=False):
    all_results = {}
    for dataset_name in tqdm(datasets, desc="Datasets"):
        # Load dataset
        real_data, metadata = download_demo(modality='single_table', dataset_name=dataset_name)
        if numerical_only:
            # Keep only numerical columns
            real_data = real_data.select_dtypes(include=[np.number])
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_data)
        num_columns = len(real_data.columns)
        print(f"Running experiments on {dataset_name} dataset ...")
        dataset_results = []
        for num_rows in row_sizes:
            data = real_data[:num_rows]
            print(f"Using {num_rows} rows from dataset {dataset_name}")
            for synthesizer_name, synthesizer_class in synthesizers.items():
                print(f"Processing dataset {dataset_name} with synthesizer {synthesizer_name} and {num_rows} rows...")
                try:
                    # Initialize synthesizer
                    if synthesizer_name.startswith('Decomposition'):
                        base_synthesizer_name = synthesizer_name.split('_')[1]
                        synthesizer = synthesizer_class(metadata, synthesizer_name=base_synthesizer_name)
                    else:
                        synthesizer = synthesizer_class(metadata)
                    # Train and evaluate
                    metrics, fit_time, sample_time = train_and_evaluate_synthesizer(synthesizer, data, metadata, num_samples)
                    # Record results
                    row = {
                        "Dataset": dataset_name,
                        "Num Rows": num_rows,
                        "Synthesizer": synthesizer_name,
                        "Fit Time (s)": fit_time,
                        "Sample Time (s)": sample_time,
                    }
                    row.update(metrics)  # Add metrics to the row
                    dataset_results.append(row)
                except Exception as e:
                    print(f"Error processing {dataset_name} with {synthesizer_name}: {e}")
                    continue
        # Convert results to DataFrame
        results_df = pd.DataFrame(dataset_results)
        # Store in dictionary
        all_results[dataset_name] = results_df
    # save results to json
    with open(save_path+'.json', "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    # Save results to Excel with multiple sheets
    """with pd.ExcelWriter(save_path+'.xlsx') as writer:
        for dataset_name, results_df in all_results.items():
            sheet_name = dataset_name[:31]  # Excel sheet names can't exceed 31 characters
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)"""
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    dataset_names = get_available_demos(modality='single_table')['dataset_name']
    exclude_datasets = ['intrusion']
    dataset_names = [dataset_name for dataset_name in dataset_names if dataset_name not in exclude_datasets]
    # dataset_names = ["ring", "asia", "fake_companies", "census_extended", "insurance", "alarm", "census", "covtype", "news", ]
    
    def create_decomposition_synthesizer_factory(base_synth_name, decomposer_class):
        return lambda metadata: DecompositionSynthesizer(metadata, synthesizer_name=base_synth_name, decomposer=decomposer_class)
    
    synthesizer_classes = {
        "GaussianCopulaSynthesizer": lambda metadata: TableSynthesizer("GaussianCopulaSynthesizer", metadata),
        "CTGANSynthesizer": lambda metadata: TableSynthesizer("CTGANSynthesizer", metadata),
        "TVAESynthesizer": lambda metadata: TableSynthesizer("TVAESynthesizer", metadata),
        "CopulaGANSynthesizer": lambda metadata: TableSynthesizer("CopulaGANSynthesizer", metadata),
        # Decomposition synthesizers with different base synthesizers
        "Decomposition_GaussianCopulaSynthesizer": lambda metadata, **kwargs: DecompositionSynthesizer(metadata, **kwargs),
        "Decomposition_CTGANSynthesizer": lambda metadata, **kwargs: DecompositionSynthesizer(metadata, **kwargs),
        "Decomposition_TVAESynthesizer": lambda metadata, **kwargs: DecompositionSynthesizer(metadata, **kwargs),
        "Decomposition_CopulaGANSynthesizer": lambda metadata, **kwargs: DecompositionSynthesizer(metadata, **kwargs),
    }

    # Define the row sizes you want to test
    row_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # Adjust as needed

    print('available dataset_names are: ', dataset_names)
    # Run benchmark
    benchmark_datasets_and_synthesizers(
        #datasets=dataset_names,
        datasets=['covtype'], ###### TODO: modify
        synthesizers=synthesizer_classes,
        num_samples=1000,
        row_sizes=row_sizes,
        save_path="benchmark_results",
        numerical_only=False
    )
    synthesizer_classes = {
        "GaussianCopulaSynthesizer": lambda metadata: TableSynthesizer("GaussianCopulaSynthesizer", metadata),
        "CTGANSynthesizer": lambda metadata: TableSynthesizer("CTGANSynthesizer", metadata),
        "TVAESynthesizer": lambda metadata: TableSynthesizer("TVAESynthesizer", metadata),
        "CopulaGANSynthesizer": lambda metadata: TableSynthesizer("CopulaGANSynthesizer", metadata),
    }





    # next test matrix decomposition (which requires numerical_only=True)
    base_synthesizers = ["GaussianCopulaSynthesizer", "CTGANSynthesizer", "TVAESynthesizer", "CopulaGANSynthesizer"]
    decomposer_names = ["Truncate", "NMF", "PCA", "SVD", "ICA", "FactorAnalysis", "DictionaryLearning"]
    decomposer_kwargs_mapping = {
        "NMF": {"n_components": 2},
        "PCA": {"n_components": 2},
        "SVD": {"n_components": 2},
        "ICA": {"n_components": 2},
        "FactorAnalysis": {"n_components": 2},
        "DictionaryLearning": {"n_components": 2},
        "Truncate": {"row_fraction": 0.5, "col_fraction": 0.5},
    }
    # Add decomposition synthesizers with different base synthesizers and decomposers
    for base_synth_name in base_synthesizers:
        for decomposer_name in decomposer_names:
            synthesizer_name = f"Decomposition_{base_synth_name}_{decomposer_name}"
            decomposer_kwargs = decomposer_kwargs_mapping.get(decomposer_name, {})
            synthesizer_classes[synthesizer_name] = create_decomposition_synthesizer_factory(
                base_synth_name,
                decomposer_name,
                **decomposer_kwargs
            )
    # Run benchmark for decomposer synthesizers
    benchmark_datasets_and_synthesizers(
        datasets=['covtype'], ###### TODO: modify
        synthesizers={key: synthesizer_classes[key] for key in synthesizer_classes if key.startswith('Decomposition')},
        num_samples=1000,
        row_sizes=row_sizes,
        save_path="benchmark_results_decomposers",
        numerical_only=True  # Filter data to numerical columns
    )