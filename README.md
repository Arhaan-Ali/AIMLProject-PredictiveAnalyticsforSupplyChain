# Predictive Analytics for Supply Chain

End-to-end supply chain analytics pipeline for:
- demand forecasting (Prophet + XGBoost),
- inventory optimization (ROP, safety stock, EOQ),
- and dashboard generation.

The project uses relative file paths in scripts, so commands must be run from the correct folder.

## Project Flow

1. Preprocess raw files into merged datasets  
2. Engineer time-series features  
3. Fill missing feature values  
4. Train Prophet and XGBoost models  
5. Generate future forecasts  
6. Run inventory optimization  
7. Generate dashboard panels and full dashboard

## Folder Structure (important paths)

- `Data/`: raw and processed datasets
- `Production/`: feature engineering, forecasting, inventory scripts
- `Training/`: model training scripts
- `Dashboard/`: dashboard generation script and output images
- `Models/`: saved model artifacts

## Requirements

- Python 3.10+ recommended
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Run the following commands from project root (`AIMLProject-PredictiveAnalyticsforSupplyChain`).
Each command first changes into the script folder so existing relative paths (`../Data/...`) resolve correctly.

### 1) Data preprocessing

```bash
cd research && python preprocessing.py && python merger.py && cd ..
```

### 2) Feature engineering and missing value handling

```bash
cd Production && python Data_engineering.py && python Imputation.py && cd ..
```

### 3) Train models

```bash
cd Training && python ProphetTrain.py && python XGBoostTrain.py && cd ..
```

### 4) Generate forecasts

```bash
cd Production && python Forecasting.py && cd ..
```

### 5) Inventory optimization

```bash
cd Production && python Inventory_Optimisation.py && cd ..
```

### 6) Generate dashboard

```bash
cd Dashboard && python Dashboard.py && cd ..
```

## Outputs

- Models:
  - `Models/prophet_models.pkl`
  - `Models/xgboost_model.pkl`
- Forecasts:
  - `Data/Results/prophet_all_stores_30d.csv`
  - `Data/Results/prophet_all_stores_90d.csv`
  - `Data/Results/xgb_all_stores_next_day.csv`
  - `Data/Results/xgb_all_stores_7day.csv`
- Inventory reports:
  - `Data/Inventory/weekly_replenishment_report.csv`
  - `Data/Inventory/reorder_now.csv`
- Dashboard:
  - `Dashboard/full_dashboard.png`

## Notes

- If `prophet` installation fails on your machine, update `pip`, `setuptools`, and `wheel`, then retry:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install prophet
```

- Some scripts display plots during execution and can take time depending on dataset size and number of stores.
