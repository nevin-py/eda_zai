# Predictive Maintenance for Jet Engines (NASA CMAPSS FD001)

## Project Overview

Unplanned engine downtime directly affects flight operations, maintenance costs, and safety margins.  
This project builds a predictive maintenance system that estimates Remaining Useful Life (RUL) and flags near-term failure risk from multivariate sensor streams.  
The output is an explainable Streamlit dashboard that helps maintenance teams prioritize interventions across a fleet.

## Business Impact

Proactive maintenance alerts can reduce unplanned downtime by 25% by identifying high-risk engines earlier and enabling planned interventions.

## Technical Approach

1. EDA: Loaded CMAPSS FD001 train/test sets, computed RUL labels, analyzed sensor trends, RUL distribution, and sensor-to-RUL correlations.
2. Feature Engineering: Built rolling mean/std features over 5 and 10 cycles per sensor, removed near-zero variance sensors, clipped RUL at 125.
3. Modeling: Trained baseline Linear Regression and tuned XGBoost for both regression (RUL) and classification (failure within 30 cycles).
4. Explainability: Used SHAP TreeExplainer for global feature importance, local waterfall explanations, and sensor contribution over time.

## Results

Regression (XGBoost):

- RMSE: 16.642
- MAE: 11.682
- R²: 0.636

Classification (XGBoost, failure within 30 cycles):

- Precision: 0.703
- Recall: 0.720
- F1: 0.711
- ROC-AUC: 0.989

Baseline Regression (Linear Regression):

- RMSE: 19.266
- MAE: 14.257

## How to Run

```bash
/home/ariva/.pyenv/versions/3.11.9/bin/python -m pip install -r requirements.txt
/home/ariva/.pyenv/versions/3.11.9/bin/python scripts/download_data.py --data-dir data
/home/ariva/.pyenv/versions/3.11.9/bin/python scripts/run_eda.py --data-dir data --output-dir plots --summary-path README_DATA_SUMMARY.md
/home/ariva/.pyenv/versions/3.11.9/bin/python scripts/train_models.py --data-dir data --models-dir models --plots-dir plots
/home/ariva/.pyenv/versions/3.11.9/bin/python scripts/run_explainability.py --data-dir data --models-dir models --plots-dir plots
/home/ariva/.pyenv/versions/3.11.9/bin/python scripts/generate_notebooks.py
/home/ariva/.pyenv/versions/3.11.9/bin/python -m streamlit run app.py
```

## Data Summary

- Train set has 20,631 rows across 100 engines; test set has 13,096 rows across 100 engines.
- Average observed lifecycle is 206.3 cycles, with longest engine run of 362 cycles in train data.
- Near-zero variance sensors removed: sensor_1, sensor_10, sensor_16, sensor_18, sensor_19, sensor_5, sensor_6.
- Most RUL-informative sensors by absolute correlation: sensor_11, sensor_4, sensor_12, sensor_7, sensor_15.

## Folder Structure

```text
predictive-maintenance/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── notebooks/
│   ├── eda.ipynb
│   ├── modeling.ipynb
│   └── explainability.ipynb
├── models/
│   ├── rul_model.pkl
│   ├── classifier.pkl
│   ├── metrics.json
│   ├── metadata.json
│   └── fleet_predictions.csv
├── plots/
│   ├── eda_sensor_trends.png
│   ├── eda_rul_distribution.png
│   ├── eda_sensor_rul_corr_heatmap.png
│   ├── confusion_matrix.png
│   ├── shap_summary_bar.png
│   ├── shap_waterfall_healthy.png
│   ├── shap_waterfall_degrading.png
│   ├── shap_waterfall_near_failure.png
│   └── shap_top_sensor_over_time.png
├── scripts/
├── src/predictive_maintenance/
├── app.py
├── requirements.txt
└── README.md
```
