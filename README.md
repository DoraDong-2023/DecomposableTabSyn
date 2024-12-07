# DecomposableTabSyn
The implementation of the decomposable tabular synthesis

## Installation
```bash
pip install -e .
```

## Analysis
```bash
python -m analysis.TimevsRowsinDecomposition
```

```bash
cd src
python synthesis2.py
```

## Test decomposition files
```bash
python -m src.decomposition
```

## Extract metrics and plot figures
```bash
cd src
python analysis_extract_metrics.py # get experiment_metrics.csv
python analysis_separate.py # get images/*.png files
```