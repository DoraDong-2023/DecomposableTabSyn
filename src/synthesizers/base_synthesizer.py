import pandas as pd
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
)
from sdv.metadata import SingleTableMetadata
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
    def __init__(self, synthesizer_name, metadata, filepath=None, synthesizer_config={}):
        assert synthesizer_name in self.synthesizer_configs, f"Invalid synthesizer name: {synthesizer_name}"
        synthesizer_config = synthesizer_config or self.synthesizer_configs[synthesizer_name]
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