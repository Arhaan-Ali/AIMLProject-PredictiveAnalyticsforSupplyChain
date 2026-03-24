# ============================================================
# predict_prophet.py  —  Load models and forecast future sales
# ============================================================

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

os.makedirs("../Data/Forecasts", exist_ok=True)

# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading models...")
with open("../Models/prophet_models.pkl", "rb") as f:
    store_models = pickle.load(f)
print(f"✅ Loaded {len(store_models)} store models")

# ── Load Data (to get last known regressor values per store) ──────────────────
df = pd.read_csv("../Data/Processed Data/train_features.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

REGRESSORS = [
    'Open', 'DayOfWeek', 'Customers', 'Promo',
    'days_since_promo', 'rolling_7', 'lag_7', 'lag_14', 'rolling_30'
]

# ── Helper: Build future dataframe with regressors ────────────────────────────
def build_future_df(store_df, periods):
    last_date  = store_df['Date'].max()
    last_known = store_df.iloc[-1]  # last row of this store

    future_dates = pd.date_range(
        start = last_date + pd.Timedelta(days=1),
        periods = periods,
        freq    = 'D'
    )

    future = pd.DataFrame({'ds': future_dates})

    # Fill regressors with last known values as baseline
    future['Open']             = 1
    future['DayOfWeek']        = future['ds'].dt.dayofweek + 1  # 1=Mon, 7=Sun
    future['Customers']        = store_df['Customers'].rolling(7).mean().iloc[-1]
    future['Promo']            = 0  # assume no promo by default
    future['days_since_promo'] = range(1, periods + 1)  # counting up from last promo
    future['rolling_7']        = store_df['Sales'].tail(7).mean()
    future['rolling_30']       = store_df['Sales'].tail(30).mean()
    future['lag_7']            = store_df['Sales'].tail(7).mean()
    future['lag_14']           = store_df['Sales'].tail(14).mean()

    return future

# ── Forecast for all stores ───────────────────────────────────────────────────
forecasts_30  = []
forecasts_90  = []
total = len(store_models)
done  = 0

for store_id, model in store_models.items():
    store_df = df[df['Store'] == store_id].sort_values('Date').reset_index(drop=True)

    for periods, forecast_list in [(30, forecasts_30), (90, forecasts_90)]:

        future   = build_future_df(store_df, periods)

        # ── 80% uncertainty interval ─────────────────────────────────────────
        model.interval_width = 0.80
        forecast_80 = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_80.columns = ['Date', 'yhat', 'lower_80', 'upper_80']

        # ── 95% uncertainty interval ─────────────────────────────────────────
        model.interval_width = 0.95
        forecast_95 = model.predict(future)[['yhat_lower', 'yhat_upper']].copy()
        forecast_95.columns = ['lower_95', 'upper_95']

        # ── Combine ───────────────────────────────────────────────────────────
        combined              = pd.concat([forecast_80, forecast_95], axis=1)
        combined['Store']     = store_id
        combined['yhat']      = combined['yhat'].clip(lower=0)
        combined['lower_80']  = combined['lower_80'].clip(lower=0)
        combined['upper_80']  = combined['upper_80'].clip(lower=0)
        combined['lower_95']  = combined['lower_95'].clip(lower=0)
        combined['upper_95']  = combined['upper_95'].clip(lower=0)

        combined = combined[['Store', 'Date', 'yhat', 'lower_80', 'upper_80', 'lower_95', 'upper_95']]
        forecast_list.append(combined)

    done += 1
    print(f"  ✅ Store {store_id} done ({done}/{total})")

# ── Combine all stores ────────────────────────────────────────────────────────
df_30 = pd.concat(forecasts_30).reset_index(drop=True)
df_90 = pd.concat(forecasts_90).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
df_30.to_csv("../Data/Forecasts/forecast_30_days.csv",  index=False)
df_90.to_csv("../Data/Forecasts/forecast_90_days.csv",  index=False)

print(f"\n✅ 30-day forecast saved → ../Data/Forecasts/forecast_30_days.csv")
print(f"✅ 90-day forecast saved → ../Data/Forecasts/forecast_90_days.csv")
print(f"\n30-day shape : {df_30.shape}")
print(f"90-day shape : {df_90.shape}")

# ── Sample output ─────────────────────────────────────────────────────────────
print(f"\n── Sample 30-day forecast (Store 1) ─────────────────────")
print(df_30[df_30['Store'] == 1].head(10).to_string(index=False))

# ── Plot sample store forecast ────────────────────────────────────────────────
sample_store = 1
hist  = df[df['Store'] == sample_store].tail(60)
f30   = df_30[df_30['Store'] == sample_store]
f90   = df_90[df_90['Store'] == sample_store]

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

for ax, forecast, label in zip(axes, [f30, f90], ['30-Day', '90-Day']):
    ax.plot(hist['Date'],    hist['Sales'],    color='black',      label='Historical',  linewidth=1.5)
    ax.plot(forecast['Date'], forecast['yhat'], color='steelblue', label='Forecast',    linewidth=1.5)
    ax.fill_between(forecast['Date'], forecast['lower_95'], forecast['upper_95'],
                    alpha=0.2, color='steelblue', label='95% Interval')
    ax.fill_between(forecast['Date'], forecast['lower_80'], forecast['upper_80'],
                    alpha=0.4, color='steelblue', label='80% Interval')
    ax.set_title(f"{label} Forecast — Store {sample_store}", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

plt.tight_layout()
plt.savefig("../Data/Forecasts/forecast_plot_store1.png", dpi=150)
plt.show()

print("\n✅ Forecast plot saved → ../Data/Forecasts/forecast_plot_store1.png")


print("\nPlotting forecast decomposition for Store 1...")

sample_store = 1
model        = store_models[sample_store]
store_df     = df[df['Store'] == sample_store].sort_values('Date').reset_index(drop=True)

future   = build_future_df(store_df, 90)
model.interval_width = 0.95
forecast = model.predict(future)

fig = model.plot_components(forecast)
plt.suptitle(f"Forecast Decomposition — Store {sample_store}", fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../Data/Forecasts/decomposition_store_1.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Decomposition plot saved → ../Data/Forecasts/decomposition_store_1.png")
