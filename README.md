# DecomposableTabSyn
The implementation of the Waterloo CS848 project: `Raw Table Synthesis through Decomposition`

## Overview
This project implements a framework for decomposing and synthesizing tabular data. The structure and functionality are designed to allow for extendability, enabling the integration of custom synthesizers and decomposers by inheriting basic classifiers provided in the `synthesizers` and `decomposers` modules.

## Repository Structure
```bash
base ❯ tree -L 2
.
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── src
    ├── __init__.py
    ├── analysis_extract_metrics.py
    ├── analysis_separate.py
    ├── dataloader.py
    ├── decompose.py
    ├── decomposition.py
    ├── eval_synthesizer.py
    ├── eval_synthesizers.sh
    ├── evaluation.py
    ├── synthesis2.py
    ├── synthesizers
    └── utils.py
```

You can test the decomposition module independently:
```bash
python -m src.decomposition
```

## Installation
Install the package in editable mode to enable development and testing:
```bash
pip install -e .
```

## Running Experiments
The framework includes scripts to conduct five designed experiments and analyze their results.

### Step 1: Execute the Experiments
Navigate to the `src` directory and run the evaluation script:
```bash
cd src
sh eval_synthesizers.sh
```
This script runs five predefined experiments using the implemented synthesizers.

### Step 2: Extract Metrics and Integrate Results
Once the experiments are complete, extract the metrics:
```bash
python analysis_extract_metrics.py
```
This generates the file `experiment_metrics_with_labels.csv` that consolidates all experiment results.

### Step 3: Visualize and Plot Results
Generate plots from the extracted metrics and save them into the `images` directory:
```bash
python analysis_separate.py
```
The resulting visualizations are stored as PNG files in the `images` folder.

## Key Features
- Extendable synthesizers and decomposers: Create custom extensions by inheriting from base classifiers in the `synthesizers` and `decomposers` modules.
- Automated experiment execution: Run multiple predefined experiments using a single shell script.
- Integrated analysis: Extract metrics and visualize results for better interpretability.
- Modular design: Analyze, synthesize, and decompose tabular data through standalone modules.

## Requirements
All dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

