from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from decomposition import TruncateDecomposition
from sdv.metadata import SingleTableMetadata
class DecompositionSynthesizer(BaseSynthesis):
    def __init__(self, metadata, row_fraction=0.5, col_fraction=0.5):
        """
        Initialize with metadata and decomposition parameters.
        """
        self.metadata = metadata
        self.row_fraction = row_fraction
        self.col_fraction = col_fraction
        self.decomposer = TruncateDecomposition(row_fraction, col_fraction)
        self.synthesizers = []

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
            synthesizer = GaussianCopulaSynthesizer(part_metadata)
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

if __name__ == "__main__":
    import time
    import json
    import prettytable as pt
    from tqdm import tqdm
    from pathlib import Path
    from sdv.datasets.demo import download_demo, get_available_demos
    from utils import plot_model_scaling
    dataset_names = get_available_demos(modality='single_table')['dataset_name']
    exclude_datasets = ['intrusion']
    dataset_names = [dataset_name for dataset_name in dataset_names if dataset_name not in exclude_datasets]
    # dataset_names = ["ring", "asia", "fake_companies", "census_extended", "insurance", "alarm", "census", "covtype", "news", ]
    results_file = Path("synthesis_speed_results.json")
    if False: #results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
        dataset_name = 'covtype'
        real_data, metadata = download_demo(modality='single_table', dataset_name=dataset_name)
        num_columns = len(real_data.columns)
        print(f"Running experiments on {dataset_name} dataset ...")
        # restric the number of rows to 10000
        data = real_data[:1000]
        print(f"Cut down the number of rows to {len(data)} from {len(real_data)} for faster synthesis.")
        # test the speed of different synthesizers on fitting and sampling
        ds_results = {}
        print('='*10)
        synthesizer_name = 'GaussianCopulaSynthesizer'
        assert synthesizer_name not in results.get(dataset_name, {})
        save_path = Path("saves") / dataset_name / f"{synthesizer_name}.pkl"
        fit_time_path = Path("saves") / dataset_name / f"{synthesizer_name}_fit_time.txt"
        if False: #save_path.exists():
            synthesizer = TableSynthesizer(synthesizer_name, metadata, filepath=save_path)
            print(f"Loaded {synthesizer_name} synthesizer from {save_path}")
            with open(fit_time_path, "r") as f:
                fit_time = float(f.read())
        else:
            synthesizer = TableSynthesizer(synthesizer_name, metadata)
            print(f"Fitting the {synthesizer_name} synthesizer ...")
            start = time.time()
            synthesizer.fit(data)
            fit_time = time.time()- start
            print(f"Fitting Time taken: {fit_time:.2f}s")
            # save synthesizer
            save_path.parent.mkdir(parents=True, exist_ok=True)
            synthesizer.synthesizer.save(save_path)
            with open(fit_time_path, "w") as f:
                f.write(str(fit_time))
        # synthesis period
        num_samples = 10000
        start = time.time()
        synthetic_data = synthesizer.sample(num_samples)
        sample_time = time.time()- start
        sample_time_per_row = sample_time / num_samples
        print(f"Time taken to generate {num_samples} samples: {sample_time:.2f}s")
        print(f"Average time taken to generate a sample: {(sample_time) * 1000 / num_samples:.2f}ms")
        print('='*10)
        
        # another synthesizer
        decomposed_synthesizer = DecompositionSynthesizer(metadata, row_fraction=0.5, col_fraction=0.5)
        start = time.time()
        decomposed_synthesizer.fit(data)
        decomposed_fit_time = time.time() - start

        start = time.time()
        decomposed_samples = decomposed_synthesizer.sample(num_samples)
        decomposed_sample_time = time.time() - start
        print('='*10)
        print(f"Decomposed Synthesizer Fit Time: {decomposed_fit_time:.2f}s")
        print(f"Decomposed Synthesizer Sample Time: {decomposed_sample_time:.2f}s")
        timing_details = decomposed_synthesizer.get_timing_details()
        print(json.dumps(timing_details,indent=4))
        print('='*10)

    row_sizes = list(range(1000, 11000, 1000))
    fit_times_original = []
    fit_times_decomposed = []
    sample_times_original = []
    sample_times_decomposed = []

    # Loop through each row size and measure times
    for num_rows in tqdm(row_sizes, desc="Benchmarking"):
        # Prepare data of the current size
        data_subset = real_data[:num_rows]

        # Original Synthesizer
        print(f"Testing original synthesizer with {num_rows} rows...")
        synthesizer = TableSynthesizer("GaussianCopulaSynthesizer", metadata)
        start = time.time()
        synthesizer.fit(data_subset)
        fit_time_original = time.time() - start

        start = time.time()
        synthetic_data = synthesizer.sample(num_samples)
        sample_time_original = time.time() - start

        # Decomposed Synthesizer
        print(f"Testing decomposed synthesizer with {num_rows} rows...")
        decomposed_synthesizer = DecompositionSynthesizer(metadata, row_fraction=0.5, col_fraction=0.5)
        start = time.time()
        decomposed_synthesizer.fit(data_subset)
        fit_time_decomposed = time.time() - start

        start = time.time()
        decomposed_samples = decomposed_synthesizer.sample(num_samples)
        sample_time_decomposed = time.time() - start

        # Store the results
        fit_times_original.append(fit_time_original)
        sample_times_original.append(sample_time_original)
        fit_times_decomposed.append(fit_time_decomposed)
        sample_times_decomposed.append(sample_time_decomposed)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Fit Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(row_sizes, fit_times_original, label='Original Synthesizer', marker='o')
    plt.plot(row_sizes, fit_times_decomposed, label='Decomposed Synthesizer', marker='o')
    plt.xlabel('Number of Rows')
    plt.ylabel('Fit Time (seconds)')
    plt.title('Fit Time vs Number of Rows')
    plt.legend()
    plt.grid()

    # Sample Time Plot
    plt.subplot(1, 2, 2)
    plt.plot(row_sizes, sample_times_original, label='Original Synthesizer', marker='o')
    plt.plot(row_sizes, sample_times_decomposed, label='Decomposed Synthesizer', marker='o')
    plt.xlabel('Number of Rows')
    plt.ylabel('Sample Time (seconds)')
    plt.title('Sample Time vs Number of Rows')
    plt.legend()
    plt.grid()

    # Show and save the plot
    plt.tight_layout()
    plt.show()
    plt.savefig("synthesizer_scaling_comparison.png")