from sdv.evaluation.single_table import evaluate_quality, get_column_plot
import pandas as pd

class BaseEvaluation:
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata):
        raise NotImplementedError("Subclasses should implement this method.")

class QualityEvaluation(BaseEvaluation):
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata):
        quality_report = evaluate_quality(real_data, synthetic_data, metadata)
        print("\nGenerating report ...")
        print(f"\nOverall Score (Average): {quality_report.get_score() * 100:.2f}%\n")
        return quality_report

    def plot_column(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column_name: str, metadata):
        if not hasattr(metadata, 'columns') or column_name not in metadata.columns:
            raise ValueError("The metadata does not contain the specified column or is improperly formatted.")
        return get_column_plot(real_data, synthetic_data, column_name, metadata)

if __name__ == "__main__":
    from sdv.datasets.demo import download_demo
    from synthesis import TableSynthesis
    
    real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    print(real_data)
    synthesizer = TableSynthesis(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(100)
    
    evaluator = QualityEvaluation()
    quality_report = evaluator.evaluate(real_data, synthetic_data, metadata)
    print("\nSynthetic Data Quality Report:")
    print(quality_report)