import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("../Models",          exist_ok=True)
os.makedirs("../Data/Forecasts",  exist_ok=True)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


print("STEP 1 — Loading data...")

df = pd.read_csv("../Data/Processed Data/train_features.csv")
df['Date']  = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)


df = df[df['Open'] == 1].reset_index(drop=True)

print(f"  Dataset shape after filtering closed days: {df.shape}")


print("STEP 2 — Adding lag_30...")

df['lag_30'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(30))
df['lag_30'] = df.groupby('Store')['lag_30'].transform(lambda x: x.ffill().bfill())


print("STEP 3 — Applying log transform to Sales...")


df['Sales_log'] = np.log1p(df['Sales'])


FEATURES = [
    'Customers', 'rolling_7', 'lag_1', 'lag_14', 'days_since_promo', 'Promo', 'Store', 'lag_7','rolling_30','DayOfWeek'
]

print(f"  Features: {FEATURES}")


print("STEP 5 — Splitting data chronologically...")

df       = df.sort_values('Date').reset_index(drop=True)
split    = int(len(df) * 0.80)
train    = df.iloc[:split]
test     = df.iloc[split:]

X_train  = train[FEATURES]
y_train  = train['Sales_log']
X_test   = test[FEATURES]
y_test   = test['Sales']

print(f"  Train shape : {train.shape}")
print(f"  Test shape  : {test.shape}")
print(f"  Train dates : {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"  Test dates  : {test['Date'].min().date()}  → {test['Date'].max().date()}")


print("\nSTEP 6 — Training base XGBoost model...")

base_model = XGBRegressor(
    n_estimators  = 500,
    learning_rate = 0.05,
    max_depth     = 6,
    subsample     = 0.8,
    colsample_bytree = 0.8,
    random_state  = 42,
    n_jobs        = -1       # use all CPU cores
)

base_model.fit(X_train, y_train)

base_pred     = np.expm1(base_model.predict(X_test))
base_pred     = np.clip(base_pred, 0, None)

base_mae  = mean_absolute_error(y_test, base_pred)
base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
base_mape = mean_absolute_percentage_error(y_test, base_pred)
base_r2   = r2_score(y_test, base_pred)

print(f"\n Base Model Performance")
print(f"  MAE  : {base_mae:.2f}")
print(f"  RMSE : {base_rmse:.2f}")
print(f"  MAPE : {base_mape:.2f}%")
print(f"  R²   : {base_r2:.4f}")


print("\nSTEP 7 — Tuning hyperparameters with Optuna (20 trials)...")

def objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators',    200, 1000),
        'max_depth'       : trial.suggest_int('max_depth',       4,   10),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample'       : trial.suggest_float('subsample',     0.6,  1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma'           : trial.suggest_float('gamma',         0.0,  1.0),
        'random_state'    : 42,
        'n_jobs'          : -1
    }

    m = XGBRegressor(**params)
    m.fit(X_train, y_train)

    pred = np.expm1(m.predict(X_test))
    pred = np.clip(pred, 0, None)
    return np.sqrt(mean_squared_error(y_test, pred))  # minimize RMSE

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"\n  Best params : {study.best_params}")
print(f"  Best RMSE   : {study.best_value:.2f}")


print("\nSTEP 8 — Training final model with best params...")

final_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

final_pred  = np.expm1(final_model.predict(X_test))
final_pred  = np.clip(final_pred, 0, None)

final_mae   = mean_absolute_error(y_test, final_pred)
final_rmse  = np.sqrt(mean_squared_error(y_test, final_pred))
final_mape  = mean_absolute_percentage_error(y_test, final_pred)
final_r2    = r2_score(y_test, final_pred)

print(f"\n Final Model Performance ")
print(f"  MAE  : {final_mae:.2f}")
print(f"  RMSE : {final_rmse:.2f}")
print(f"  MAPE : {final_mape:.2f}%")
print(f"  R²   : {final_r2:.4f}")


print("\nSTEP 10 — Calculating SHAP values...")

# Use a sample for SHAP since full test set is large
shap_sample  = X_test.sample(2000, random_state=42)
explainer    = shap.Explainer(final_model)
shap_values  = explainer(shap_sample)

# Summary plot — shows which features drive predictions most
shap.summary_plot(shap_values, shap_sample, show=False)
plt.title("SHAP Feature Importance — XGBoost", fontweight='bold')
plt.tight_layout()
plt.savefig("../Data/Processed Data/Graphs/shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print(" SHAP plot saved → ../Data/Processed Data/Graphs/shap_summary.png")

# Bar plot — mean absolute SHAP values
shap.summary_plot(shap_values, shap_sample, plot_type='bar', show=False)
plt.title("SHAP Feature Importance (Bar) — XGBoost", fontweight='bold')
plt.tight_layout()
plt.savefig("../Data/Processed Data/Graphs/shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print(" SHAP bar plot saved → ../Data/Processed Data/Graphs/shap_bar.png")


results         = test[['Date', 'Store', 'Sales']].copy()
results['yhat'] = final_pred

# Plot for one sample store
sample_store    = 1
store_results   = results[results['Store'] == sample_store].sort_values('Date')

plt.figure(figsize=(15, 5))
plt.plot(store_results['Date'], store_results['Sales'], label='Actual',    alpha=0.7)
plt.plot(store_results['Date'], store_results['yhat'],  label='Predicted', alpha=0.7)
plt.title(f"XGBoost — Actual vs Predicted Sales (Store {sample_store})", fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("../Data/Processed Data/Graphs/xgboost_actual_vs_predicted.png", dpi=150)
plt.show()


joblib.dump(final_model, "../Models/xgboost_model.pkl")
print("\n Final model saved → ../Models/xgboost_model.pkl")

# Save best params
pd.DataFrame([study.best_params]).to_csv("../Models/xgboost_best_params.csv", index=False)
print(" Best params saved → ../Models/xgboost_best_params.csv")