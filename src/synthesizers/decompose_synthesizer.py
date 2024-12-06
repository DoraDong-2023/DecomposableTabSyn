import time
import pandas as pd
from src.synthesizers.base_synthesizer import BaseSynthesis, TableSynthesizer
from src.decomposition import (
    TruncateDecomposition,
    NMFDecomposition,
    PCADecomposition,
    SVDDecomposition,
    ICADecomposition,
    FactorAnalysisDecomposition,
    DictionaryLearningDecomposition,
    NFDecomposition,
    BayesianDecomposition,
)
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata

decomposier_name_to_class = {
    # 'BayesianDecomposition': BayesianDecomposition,
    'NFDecomposition': NFDecomposition,
    "TruncateDecomposition": TruncateDecomposition,
    
    'NMFDecomposition': NMFDecomposition,
    'PCADecomposition': PCADecomposition,
    'SVDDecomposition': SVDDecomposition,
    'ICADecomposition': ICADecomposition,
    'FactorAnalysisDecomposition': FactorAnalysisDecomposition,
    'DictionaryLearningDecomposition': DictionaryLearningDecomposition,
}
class DecompositionSynthesizer(BaseSynthesis):
    
    def __init__(self, metadata, synthesizer_name="GaussianCopulaSynthesizer", decomposer_name=None, synthesizer_init_kwargs={}, decomposer_init_kwargs={}, decomposer_split_kwargs={}):
        """
        Initialize with metadata and decomposition parameters.
        """
        self.metadata = metadata
        
        if not decomposer_name:
            decomposer_name = "TruncateDecomposition"
        self.decomposer_name = decomposer_name
        self.decomposer = decomposier_name_to_class[decomposer_name](**decomposer_init_kwargs)
        
        self.decomposer_split_kwargs = decomposer_split_kwargs
        self.synthesizers = []
        self.synthesizer_name = synthesizer_name
        self.synthesizer_init_kwargs = synthesizer_init_kwargs or TableSynthesizer.synthesizer_configs.get(synthesizer_name, {})
        

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
        data = data.copy(deep=True)
        start_time = time.time()
        sub_tables = self.decomposer.split(data, **self.decomposer_split_kwargs)
        decomposition_time = time.time() - start_time
        # print(f"Data decomposition completed in {decomposition_time:.2f} seconds.")

        self.synthesizers = []
        fit_times = []
        for i, sub_table in tqdm(enumerate(sub_tables), total=len(sub_tables), desc="Fitting synthesizers"):
            
            sub_table_metadata = self._update_metadata(sub_table)
            synthesizer = TableSynthesizer(self.synthesizer_name, sub_table_metadata, synthesizer_config=self.synthesizer_init_kwargs)
            start_time = time.time()
            synthesizer.fit(sub_table)
            fit_time = time.time() - start_time
            self.synthesizers.append(synthesizer)
            fit_times.append(fit_time)
        total_fit_time = sum(fit_times)
        # print(f"Total fit time for all parts: {total_fit_time:.2f} seconds.")
        self.decomposition_time = decomposition_time
        self.fit_times = fit_times
        self.total_fit_time = total_fit_time

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Sample data from each fitted synthesizer and join the samples back together.
        """
        sampled_sub_tables = []
        num_samples_per_sub_tables = []
        sample_times = []
        
        if hasattr(self.decomposer, "num_rows_to_sample_per_subtable"):
            num_samples_per_sub_tables = self.decomposer.num_rows_to_sample_per_subtable(num_rows)
        else:
            num_samples_per_sub_tables = [num_rows] * len(self.synthesizers)
            

        for i, synthesizer in enumerate(self.synthesizers):
            start_time = time.time()
            sampled_sub_table = synthesizer.sample(num_samples_per_sub_tables[i])
            sample_time = time.time() - start_time
            #print(f"Sample time for part {i + 1}: {sample_time:.2f} seconds.")
            sampled_sub_tables.append(sampled_sub_table)
            sample_times.append(sample_time)
        total_sample_time = sum(sample_times)
        #print(f"Total sample time for all parts: {total_sample_time:.2f} seconds.")

        # Join the sampled parts using the decomposition join method
        start_time = time.time()
        #print("Joining sampled parts...")
        joined_data = self.decomposer.join(sampled_sub_tables)
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
    from src.dataloader import DemoDataLoader
    real_data, meta_data = DemoDataLoader(dataset_name="covtype").load_data()
    real_data = real_data.sample(5000)
    
    print("Real Data:")
    print(real_data.head())
    decomposer_name = "PCADecomposition"
    synthesizer_name = "REaLTabFormer"
    decomposer_init_kwargs = {"n_components": 8}
    decomposer_split_kwargs = {}
    synthesizer_init_kwargs = {"epochs": 2}
    
    synthesizer = DecompositionSynthesizer(
        meta_data, synthesizer_name=synthesizer_name, decomposer_name=decomposer_name,
        decomposer_init_kwargs=decomposer_init_kwargs, decomposer_split_kwargs=decomposer_split_kwargs,
        synthesizer_init_kwargs=synthesizer_init_kwargs
    )
    synthesizer.fit(real_data)
    sampled_data = synthesizer.sample(1000)
    
    print("Sampled Data:")
    print(sampled_data.head())
    
    print("Timing Details:")
    print(synthesizer.get_timing_details())
    print("Decomposition Synthesizer Test Completed.")
    

