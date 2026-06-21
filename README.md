# Subnational Drought Stress Classification — Rwanda

A compact machine learning pipeline to classify seasonal drought stress levels across Rwanda's districts using CHIRPS dekadal rainfall data (WFP / HDX).

## Overview
- Implements preprocessing, feature engineering, classical ML (Logistic Regression, Random Forest, XGBoost) and deep learning (TensorFlow) experiments.
- Aggregates dekadal CHIRPS rainfall into seasonal features (MAM, OND) and predicts drought stress classes derived from 3-month rainfall anomaly.

## Data
- Source: WFP / HDX — "RWA Rainfall Subnational Data.csv" (CHIRPS dekadal, 1981–2025).
- Stored in: `data/RWA Rainfall Subnational Data.csv` (place the CSV file here before running).

## Notebook
- Main analysis and pipeline: `module/Rwanda Drought Stress ML Pipeline.ipynb`.

## Requirements
Recommended Python packages:
```
pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow imbalanced-learn
```
Install with pip, for example:
```
python -m pip install -r requirements.txt
# or
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow imbalanced-learn
```

## Quickstart
1. Place the dataset CSV into the `data/` folder.
2. Open the notebook `module/Rwanda Drought Stress ML Pipeline.ipynb` in Jupyter / VS Code.
3. Run cells sequentially from the Environment Setup section to reproduce preprocessing, training, and evaluation.

## Output
- Figures and tables are saved to the workspace (e.g., `fig1_distributions.png`, `experiment_results.csv`).
- Final experiment log is exported to `experiment_results.csv`.

## Notes
- The notebook uses a temporal train/val/test split (train ≤2015, val 2016–2019, test ≥2020) to avoid leakage.
- Class imbalance is addressed via class weighting; very rare classes (e.g., Severe Drought) may require resampling to improve detection.

