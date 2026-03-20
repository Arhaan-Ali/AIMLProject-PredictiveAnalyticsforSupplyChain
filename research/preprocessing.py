import pandas as pd

datasets = {
    "test": ("../Data/test.csv", "Id", 1),
    "store": ("../Data/store.csv", "Store" , 0)
}

for name, (path, id_col, fill_val) in datasets.items():
    df = pd.read_csv(path, low_memory=False)
    df = df.fillna(fill_val)

    df.to_csv(f"../Data/Processed Data/{name}_filled.csv", index=False)
    print(f"{name}: saved with missing values filled with {fill_val}")