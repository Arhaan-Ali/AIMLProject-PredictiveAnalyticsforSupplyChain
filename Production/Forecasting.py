import os
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings

warnings.filterwarnings("ignore")

os.makedirs("../Data/Results", exist_ok=True)

print("Loading models...")

with open("../Models/prophet_models.pkl", "rb") as f:
    prophet_models = pickle.load(f)

xgb_model = joblib.load("../Models/xgboost_model.pkl")

print(f"Prophet models loaded: {len(prophet_models)}")
print("XGBoost model loaded")

train = pd.read_csv("../Data/Processed Data/train_features.csv")
train["Date"] = pd.to_datetime(train["Date"])
train = train.sort_values(["Store", "Date"])

print("\nPROPHET FORECASTING...")

all_prophet_30d = []
all_prophet_90d = []

for store_id, model in prophet_models.items():
    store_train = train[train["Store"] == store_id].sort_values("Date")

    if len(store_train) < 30:
        continue

    last_date = store_train["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=90,
        freq="D"
    )

    future = pd.DataFrame({"ds": future_dates})

    future["Open"] = 1
    future["DayOfWeek"] = future["ds"].dt.dayofweek + 1
    future["Promo"] = 0
    future["days_since_promo"] = range(1, 91)

    future["rolling_7"] = store_train["Sales"].tail(7).mean()
    future["rolling_30"] = store_train["Sales"].tail(30).mean()
    future["lag_7"] = store_train["Sales"].tail(7).mean()
    future["lag_14"] = store_train["Sales"].tail(14).mean()

    model.interval_width = 0.80
    fc_80 = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc_80.columns = ["Date", "yhat", "lower_80", "upper_80"]

    model.interval_width = 0.95
    fc_95 = model.predict(future)[["yhat_lower", "yhat_upper"]]
    fc_95.columns = ["lower_95", "upper_95"]

    forecast = pd.concat([fc_80, fc_95], axis=1)
    forecast["Store"] = store_id

    forecast[["yhat", "lower_80", "upper_80", "lower_95", "upper_95"]] = (
        forecast[["yhat", "lower_80", "upper_80", "lower_95", "upper_95"]].clip(lower=0)
    )

    all_prophet_30d.append(forecast.head(30))
    all_prophet_90d.append(forecast)

    print(f"Prophet Store {store_id} done")

# Save all Prophet forecasts into single files
pd.concat(all_prophet_30d, ignore_index=True).to_csv(
    "../Data/Results/prophet_all_stores_30d.csv", index=False
)
pd.concat(all_prophet_90d, ignore_index=True).to_csv(
    "../Data/Results/prophet_all_stores_90d.csv", index=False
)
print("Prophet forecasts saved → prophet_all_stores_30d.csv & prophet_all_stores_90d.csv")


XGB_FEATURES = [
    "Customers",
    "rolling_7",
    "lag_1",
    "lag_14",
    "days_since_promo",
    "Promo",
    "Store",
    "lag_7",
    "rolling_30",
    "DayOfWeek"
]


def recursive_xgb_forecast(store_df, model, days=7):
    history = store_df.copy().sort_values("Date")
    preds = []

    for i in range(days):
        last_row = history.iloc[-1:].copy()
        next_date = last_row["Date"] + pd.Timedelta(days=1)

        new_row = last_row.copy()
        new_row["Date"] = next_date
        new_row["DayOfWeek"] = next_date.dt.dayofweek + 1

        new_row["lag_1"] = history["Sales"].iloc[-1]

        if len(history) >= 7:
            new_row["lag_7"] = history["Sales"].iloc[-7]
        else:
            new_row["lag_7"] = history["Sales"].mean()

        if len(history) >= 14:
            new_row["lag_14"] = history["Sales"].iloc[-14]
        else:
            new_row["lag_14"] = history["Sales"].mean()

        new_row["rolling_7"] = history["Sales"].tail(7).mean()
        new_row["rolling_30"] = history["Sales"].tail(30).mean()

        pred = np.expm1(model.predict(new_row[XGB_FEATURES]))[0]
        pred = max(pred, 0)

        new_row["Sales"] = pred
        preds.append((next_date.values[0], pred, store_df["Store"].iloc[0]))

        history = pd.concat([history, new_row], ignore_index=True)

    return preds


print("\nXGBOOST FORECASTING...")

all_xgb_next_day = []
all_xgb_next_week = []

for store_id in train["Store"].unique():
    store_train = train[train["Store"] == store_id].copy()

    next_day  = recursive_xgb_forecast(store_train, xgb_model, days=1)
    next_week = recursive_xgb_forecast(store_train, xgb_model, days=7)

    all_xgb_next_day.extend(next_day)
    all_xgb_next_week.extend(next_week)

    print(f"XGB Store {store_id} done")

# Save all XGBoost forecasts into single files
pd.DataFrame(all_xgb_next_day, columns=["Date", "Predicted_Sales", "Store"]).to_csv(
    "../Data/Results/xgb_all_stores_next_day.csv", index=False
)
pd.DataFrame(all_xgb_next_week, columns=["Date", "Predicted_Sales", "Store"]).to_csv(
    "../Data/Results/xgb_all_stores_7day.csv", index=False
)
print("XGBoost forecasts saved → xgb_all_stores_next_day.csv & xgb_all_stores_7day.csv")

print("\nALL DONE — FORECASTS GENERATED FOR ALL STORES")