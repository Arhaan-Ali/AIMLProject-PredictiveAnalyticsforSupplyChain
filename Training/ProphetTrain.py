import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

os.makedirs("../Models", exist_ok=True)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

REGRESSORS = [
    'DayOfWeek', 'Customers', 'Open', 'Promo',
     'rolling_7', 'lag_7', 'lag_14', 'rolling_30', 'days_since_promo'
]


df = pd.read_csv("../Data/Processed Data/train_features.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)


years    = list(range(2013, 2015))
holidays = make_holidays_df(year_list=years, country='DE')


def prepare_prophet_df(df):
    cols = ['Date', 'Sales'] + REGRESSORS
    return df[cols].rename(columns={'Date': 'ds', 'Sales': 'y'})


store_models    = {}
store_forecasts = []
train_list      = []
test_list       = []

total  = df['Store'].nunique()
done   = 0

for store_id, store_df in df.groupby('Store'):
    store_df  = store_df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(store_df) * 0.80)

    train = store_df.iloc[:split_idx]
    test  = store_df.iloc[split_idx:]

    train_list.append(train)
    test_list.append(test)

    train_p = prepare_prophet_df(train)
    test_p  = prepare_prophet_df(test)

    m = Prophet(
        weekly_seasonality      = True,
        yearly_seasonality      = False,
        daily_seasonality       = False,
        holidays                = holidays,
        seasonality_mode        = 'multiplicative',
        changepoint_prior_scale= 0.05,
        holidays_prior_scale    = 10.0,
        seasonality_prior_scale = 15.0
    )
    for reg in REGRESSORS:
        m.add_regressor(reg)

    m.fit(train_p)

    forecast        = m.predict(test_p)
    forecast['Store'] = store_id
    forecast['y']     = test_p['y'].values
    store_forecasts.append(forecast[['Store', 'ds', 'y', 'yhat']])

    store_models[store_id] = m

    done += 1
    print(f"Store {store_id} done ({done}/{total})")


train_split   = pd.concat(train_list).reset_index(drop=True)
test_split    = pd.concat(test_list).reset_index(drop=True)
all_forecasts = pd.concat(store_forecasts).reset_index(drop=True)

print(f"\nTrain shape : {train_split.shape}")
print(f"Test shape  : {test_split.shape}")


with open("../Models/prophet_models.pkl", "wb") as f:
    pickle.dump(store_models, f)

train_split.to_csv("../Data/Processed Data/train_split.csv", index=False)
test_split.to_csv("../Data/Processed Data/test_split.csv",   index=False)
all_forecasts.to_csv("../Data/Processed Data/all_forecasts.csv", index=False)

print("Models saved  → ../Models/prophet_models.pkl")
print("Splits saved  → ../Data/Processed Data/")


all_forecasts['yhat'] = all_forecasts['yhat'].clip(lower=0)

mae  = mean_absolute_error(all_forecasts['y'], all_forecasts['yhat'])
rmse = np.sqrt(mean_squared_error(all_forecasts['y'], all_forecasts['yhat']))
mape = mean_absolute_percentage_error(all_forecasts['y'], all_forecasts['yhat'])
r2   = 1 - (np.sum((all_forecasts['y'] - all_forecasts['yhat'])**2) /
            np.sum((all_forecasts['y'] - all_forecasts['y'].mean())**2))

print(f"\n{'='*45}")
print(f"  Overall Model Performance")
print(f"{'='*45}")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAPE : {mape:.2f}%")
print(f"  R²   : {r2:.4f}")
print(f"{'='*45}")



print(f"\n── Metrics Distribution:")
print(f"  MAPE — min: {metrics_df['MAPE'].min():.2f}%  |  mean: {metrics_df['MAPE'].mean():.2f}%  |  max: {metrics_df['MAPE'].max():.2f}%")
print(f"  MAE  — min: {metrics_df['MAE'].min():.2f}   |  mean: {metrics_df['MAE'].mean():.2f}   |  max: {metrics_df['MAE'].max():.2f}")
print(f"  R²   — min: {metrics_df['R2'].min():.4f}  |  mean: {metrics_df['R2'].mean():.4f}  |  max: {metrics_df['R2'].max():.4f}")


plt.figure(figsize=(15, 5))
plt.plot(all_forecasts['ds'], all_forecasts['y'],    label='Actual',    alpha=0.7)
plt.plot(all_forecasts['ds'], all_forecasts['yhat'], label='Predicted', alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Sales — All Stores")
plt.tight_layout()
plt.savefig("../Data/Processed Data/Graphs/prophet_actual_vs_predicted.png", dpi=150)
plt.show()


plt.figure(figsize=(12, 5))
plt.hist(metrics_df['MAPE'], bins=50, color='steelblue', edgecolor='black')
plt.axvline(metrics_df['MAPE'].mean(), color='red', linestyle='--',
            label=f"Mean MAPE: {metrics_df['MAPE'].mean():.2f}%")
plt.title("MAPE Distribution Across Stores")
plt.xlabel("MAPE (%)")
plt.ylabel("Number of Stores")
plt.legend()
plt.tight_layout()
plt.savefig("../Data/Processed Data/Graphs/prophet_mape_distribution.png", dpi=150)
plt.show()