import kagglehub
import shutil
import os

# Download
path = kagglehub.dataset_download("mosapabdelghany/telcom-customer-churn-dataset")
print("Path to dataset files:", path)

# Copie dans data/
for f in os.listdir(path):
    if f.endswith(".csv"):
        shutil.copy(os.path.join(path, f), "data/")
        print(f"Copied: {f} → data/")

#%%
import polars as pl

df = pl.read_csv("data/Telco_Cusomer_Churn.csv")
print(df.shape)
print(df.columns)
print(df.head(5))
# %%
