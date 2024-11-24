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
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
        for dataset_name in tqdm(dataset_names, desc="Synthesizing Datasets"):
            real_data, metadata = download_demo(modality='single_table', dataset_name=dataset_name)
            num_columns = len(real_data.columns)
            if num_columns > 100:
                continue
            
            print(f"Running experiments on {dataset_name} dataset ...")
            
            # restric the number of rows to 10000
            data = real_data[:1000]
            print(f"Cut down the number of rows to {len(data)} from {len(real_data)} for faster synthesis.")
            # test the speed of different synthesizers on fitting and sampling
            ds_results = {}
            for synthesizer_name in TableSynthesizer.synthesizer_configs:
                samples_path = Path("saves") / dataset_name / f"{synthesizer_name}_samples.csv"
                quality_report_path = Path("saves") / dataset_name / f"{synthesizer_name}_quality_report.pkl"
                if synthesizer_name in results.get(dataset_name, {}):
                    ds_results.append(results[dataset_name][synthesizer_name])
                    continue
                save_path = Path("saves") / dataset_name / f"{synthesizer_name}.pkl"
                fit_time_path = Path("saves") / dataset_name / f"{synthesizer_name}_fit_time.txt"
                if save_path.exists():
                    synthesizer = TableSynthesizer(synthesizer_name, metadata, filepath=save_path)
                    print(f"Loaded {synthesizer_name} synthesizer from {save_path}")
                    with open(fit_time_path, "r") as f:
                        fit_time = float(f.read())
                else:
                    synthesizer = TableSynthesizer(synthesizer_name, metadata)
                    start = time.time()
                    print(f"Fitting the {synthesizer_name} synthesizer ...")
                    synthesizer.fit(data)
                    end = time.time()
                    fit_time = end - start
                    print(f"Fitting Time taken: {end - start:.2f}s")
                    
                    # save synthesizer
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    synthesizer.synthesizer.save(save_path)
                    with open(fit_time_path, "w") as f:
                        f.write(str(fit_time))
                
                start = time.time()
                num_samples = 10000
                synthetic_data = synthesizer.sample(num_samples)
                end = time.time()
                sample_time = end - start
                sample_time_per_row = sample_time / num_samples
                print(f"Time taken to generate {num_samples} samples: {end - start:.2f}s")
                print(f"Average time taken to generate a sample: {(end - start) * 1000 / num_samples:.2f}ms")
                # save samples
                samples_path.parent.mkdir(parents=True, exist_ok=True)
                synthetic_data.to_csv(samples_path, index=False)
                
                ## Quality Report
                from sdmetrics.reports.single_table import QualityReport
                quality_report = QualityReport()
                quality_report.generate(data, synthetic_data, list(metadata.tables.values())[0].to_dict())
                score = quality_report.get_score()
                properties = quality_report.get_properties() # dataframe
                property_score = {k: v for k, v in zip(properties['Property'], properties['Score'])}
                details = defaultdict(dict)
                for k in properties['Property']:
                    cur_details = quality_report.get_details(property_name=k)
                    for index, row in cur_details.iterrows():
                        details[row.iloc[1]][row.iloc[0]] = row.iloc[2]
                    fig = quality_report.get_visualization(property_name=k)
                    fig.write_image(quality_report_path.parent / f"{quality_report_path.stem}_{k}.png")
                
                ds_results[synthesizer_name] = {
                    "synthesizer_name": synthesizer_name,
                    "fit_time": fit_time,
                    "num_samples": num_samples,
                    f"sample_time": sample_time,
                    "sample_time_per_row": sample_time_per_row,
                    "quality_report": {
                        "score": score, # overall score
                        "properties": property_score,
                        "details": details
                    }
                }
            results[dataset_name] = {
                "num_columns": num_columns,
                "results": ds_results
            }
                
            table = pt.PrettyTable()
            table.field_names = ["Synthesizer", "Fit Time (s)", "Num Samples", "Sample Time (1k samples)", "Sample Time (per row)"]
            for result in ds_results.values():
                table.add_row([
                    result["synthesizer_name"],
                    result["fit_time"],
                    result["num_samples"],
                    result["sample_time"],
                    result["sample_time_per_row"]
                ])
            print(table)
        
    sort_results = sorted(results.items(), key=lambda x: (x[1]["num_columns"], x[0]))
    new_results = {}
    existing_num_columns = []
    for dataset_name, dataset_results in sort_results:
        num_columns = dataset_results["num_columns"]
        if num_columns in existing_num_columns:
            continue
        existing_num_columns.append(num_columns)
        new_results[dataset_name] = dataset_results
    results = new_results
    with open("synthesis_speed_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    
    
    # plot the results
    # plot fit times against number of columns
    fit_times_dict = {}
    num_columns = []
    
    for dataset_name, dataset_results in results.items():
        num_columns.append(dataset_results["num_columns"])
        ds_results = dataset_results["results"]
        for result in ds_results.values():
            synthesizer_name = result["synthesizer_name"]
            fit_time = result["fit_time"]
            if synthesizer_name not in fit_times_dict:
                fit_times_dict[synthesizer_name] = []
            fit_times_dict[synthesizer_name].append(fit_time)
    fig = plot_model_scaling(num_columns, fit_times_dict, ylabel="Fit Time (seconds)", title="Fit Time Scaling with Number of Columns")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.show()
    plt.savefig(plots_dir / "fit_times_scaling_against_num_columns.png")
    
    # plot sample times against number of columns
    sample_times_dict = {}
    num_columns = []
    for dataset_name, dataset_results in results.items():
        num_columns.append(dataset_results["num_columns"])
        ds_results = dataset_results["results"]
        for result in ds_results.values():
            synthesizer_name = result["synthesizer_name"]
            sample_time_per_row = result["sample_time_per_row"]
            if synthesizer_name not in sample_times_dict:
                sample_times_dict[synthesizer_name] = []
            sample_times_dict[synthesizer_name].append(sample_time_per_row * 1000)
    fig = plot_model_scaling(num_columns, sample_times_dict, title="Sample Time per Row Scaling with Number of Columns", ylabel="Sample Time per row (ms)")
    plt.show()
    plt.savefig(plots_dir / "sample_times_scaling_against_num_columns.png")

    
    # print the results
    for dataset_name, dataset_results in results.items():
        num_columns = dataset_results["num_columns"]
        ds_results = dataset_results["results"]
        print(f"Results for {dataset_name} with {num_columns} columns:")
        table = pt.PrettyTable()
        table.field_names = ["Synthesizer", "Fit Time (s)", "Num Samples", "Sample Time (1k samples)", "Sample Time (per row)", "Quality Score"]
        for result in ds_results.values():
            table.add_row([
                result["synthesizer_name"],
                result["fit_time"],
                result["num_samples"],
                result["sample_time"],
                result["sample_time_per_row"],
                result["quality_report"]["score"]
            ])
        print(table)    
    

