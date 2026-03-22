import pandas as pd
import numpy as np


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])


    ff_cols = [col for col in df.columns if col.startswith('rolling_') or col.startswith('lag_') or col == 'days_since_promo']

    if ff_cols:
        for col in ff_cols:
            before = df[col].isnull().sum()
            df[col] = df.groupby('Store')[col].transform(lambda x: x.ffill())
            after = df[col].isnull().sum()
            print(f"  {col}: {before} NaNs → {after} remaining after ffill")


    if 'yoy_growth' in df.columns:
        before = df['yoy_growth'].isnull().sum()

        df['_month'] = df['Date'].dt.month
        df['_day']   = df['Date'].dt.day


        seasonal_median = (
            df.groupby(['Store', '_month', '_day'])['yoy_growth']
            .transform('median')
        )
        df['yoy_growth'] = df['yoy_growth'].fillna(seasonal_median)
        after_seasonal = df['yoy_growth'].isnull().sum()
        print(f"  yoy_growth: {before} NaNs → {after_seasonal} remaining after seasonal median")


        df['yoy_growth'] = (
            df.groupby('Store')['yoy_growth']
            .transform(lambda x: x.ffill().bfill())
        )
        after_fallback = df['yoy_growth'].isnull().sum()
        print(f"  yoy_growth: {after_seasonal} NaNs → {after_fallback} remaining after fallback ffill+bfill")

        df.drop(columns=['_month', '_day'], inplace=True)


    print("\n── Final Missing Value Report ───────────────────────────────────────")
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]

    if not remaining.empty:
        print("Remaining NaNs:")
        for col, count in remaining.items():
            pct = (count / len(df)) * 100
            print(f"     {col}: {count} ({pct:.2f}%)")
    else:
        print("No missing values remaining")

    return df


files = {
    "train": "../Data/Processed Data/train_features.csv",
    "test":  "../Data/Processed Data/test_features.csv"
}

try:
    for name, path in files.items():
        print(f"\n{'='*55}")
        print(f" Processing: {name}")
        print(f"{'='*55}")

        df = pd.read_csv(path)
        df = fill_missing_features(df)
        df.to_csv(path, index=False)  # overwrites the same file
        print(f"\n  ✅ {name} saved. Shape: {df.shape}")

except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Data error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")