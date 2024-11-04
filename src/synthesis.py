from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd

class BaseSynthesis:
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(self, num_rows: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")

class TableSynthesis(BaseSynthesis):
    def __init__(self, metadata):
        self.synthesizer = GaussianCopulaSynthesizer(metadata)
    def fit(self, data: pd.DataFrame):
        self.synthesizer.fit(data)
    def sample(self, num_rows: int) -> pd.DataFrame:
        return self.synthesizer.sample(num_rows)

if __name__ == "__main__":
    from sdv.datasets.demo import download_demo
    real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    synthesizer = TableSynthesis(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(5)
    print("Real Data Sample:")
    print(real_data.head())
    print("\nSynthetic Data Sample:")
    print(synthetic_data.head())

