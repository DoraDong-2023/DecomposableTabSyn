
from sdv.datasets.demo import download_demo

class BaseDataLoader:
    """Base class for data loading."""
    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

class DemoDataLoader(BaseDataLoader):
    def __init__(self, modality='single_table', dataset_name='fake_hotel_guests'):
        self.modality = modality
        self.dataset_name = dataset_name
    def load_data(self):
        print(f"\nLoading {self.dataset_name} dataset ...")
        real_data, metadata = download_demo(modality=self.modality, dataset_name=self.dataset_name)
        print(f"Dataset {self.dataset_name} loaded successfully.")
        return real_data, metadata

if __name__ == "__main__":
    dataloader = DemoDataLoader()
    real_data, metadata = dataloader.load_data()
    print("Loaded Data Sample:")
    print(real_data.head())
    print("\nMetadata:")
    print(metadata)

