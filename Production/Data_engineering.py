import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame, windows: list = [7, 30]) -> pd.DataFrame():

    required_cols = ['Store', 'Date']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)


    if 'Sales' in df.columns:

        for window in windows:
            df[f'rolling_{window}'] = (
                df.groupby('Store')['Sales']
                .transform(lambda x: x.rolling(window).mean())
            )

        # Lag features
        for lag in [1, 7, 14]:
            df[f'lag_{lag}'] = (
                df.groupby('Store')['Sales']
                .transform(lambda x: x.shift(lag))
            )


        df['lag_365'] = (
            df.groupby('Store')['Sales']
            .transform(lambda x: x.shift(365))
        )
        df['yoy_growth'] = ((df['Sales'] - df['lag_365']) / df['lag_365'].replace(0, np.nan)) * 100
        df.drop(columns=['lag_365'], inplace=True)


    holiday_cols = [col for col in df.columns if col in ['StateHoliday', 'SchoolHoliday']]
    if holiday_cols:
        def is_holiday(row):
            for col in holiday_cols:
                val = str(row[col]).strip()
                if val not in ['0', '0.0', 'False', 'false', '', 'nan']:
                    return 1
            return 0
        df['is_holiday'] = df.apply(is_holiday, axis=1)
    else:
        print("Warning: No holiday columns found, skipping is_holiday")


    if 'Promo' in df.columns:
        def days_since_promo(series):
            result = []
            counter = np.nan
            for val in series:
                if val == 1:
                    counter = 0
                elif counter is not np.nan:
                    counter += 1
                result.append(counter)
            return result

        df['days_since_promo'] = (
            df.groupby('Store')['Promo']
            .transform(days_since_promo)
        )
    else:
        print("Warning: 'Promo' column not found, skipping days_since_promo")

    return df


files = {
    "train": (
        "../Data/Processed Data/train_merged.csv",
        "../Data/Processed Data/train_features.csv"
    ),
    "test": (
        "../Data/Processed Data/test_merged.csv",
        "../Data/Processed Data/test_features.csv"
    )
}

try:
    for name, (input_path, output_path) in files.items():
        print(f"\nProcessing {name}...")
        df = pd.read_csv(input_path)
        df = add_features(df, windows=[7, 30])
        df.to_csv(output_path, index=False)
        print(f"{name} done. Shape: {df.shape}")
        print(df.head(3))

except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Data error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")