import pandas as pd

datasets = {
    "train": ("Data/train.csv", "Store"),
    "test": ("Data/test.csv", "Id"),
    "store": ("Data/store.csv", "Store")
}

for name, (path, id_col) in datasets.items():
    df = pd.read_csv(path, low_memory=False)
    df = df.set_index(id_col)

    missing = df.isnull().stack()
    missing = missing[missing]
    missing.index.names = [id_col, "column"]

    missing_df = missing.reset_index()[[id_col, "column"]]
    missing_df.to_csv(f"missing_in_{name}.csv", index=False)

    print(f"{name}: {len(missing_df)} missing values across {missing_df[id_col].nunique()}  unique '{id_col}' values")