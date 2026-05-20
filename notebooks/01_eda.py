# %% Imports
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("..")

from src.data_loader import load_data
from src.config import TARGET, NUMERICAL_COLS

sns.set_theme(style="whitegrid")
df = load_data()
print(df.shape)


# %% Vue d'ensemble
print(df.dtypes)
print(df.null_count())


# %% Distribution de la cible
churn_rate = df[TARGET].mean()
print(f"Churn rate: {churn_rate:.2%}")

plt.figure(figsize=(5, 4))
sns.countplot(x=df[TARGET].to_list(), palette=["steelblue", "tomato"])
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.title("Distribution du Churn")
plt.ylabel("Nombre de clients")
plt.tight_layout()
plt.show()

# %% Distribution des variables numériques
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, NUMERICAL_COLS):
    data = df[col].drop_nulls().to_list()
    ax.hist(data, bins=40, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("")
plt.suptitle("Distribution des variables numériques", y=1.02)
plt.tight_layout()
plt.show()

# %% Distribution des variables catégorielles
key_cats = ["Contract", "InternetService", "PaymentMethod", "tenure"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, col in zip(axes, key_cats):
    if col == "tenure":
        df_pd = df.select(["tenure", TARGET]).to_pandas()
        df_pd["tenure_bucket"] = pl.Series(
            df.with_columns(
                (pl.col("tenure") // 12).alias("tenure_bucket")
            )["tenure_bucket"].to_list()
        )
        churn_by = df_pd.groupby("tenure_bucket")[TARGET].mean().sort_index()
        churn_by.plot(kind="bar", ax=ax, color="tomato", edgecolor="white")
        ax.set_title("Churn rate par ancienneté (années)")
        ax.set_xlabel("Années")
    else:
        df_pd = df.select([col, TARGET]).to_pandas()
        churn_by = df_pd.groupby(col)[TARGET].mean().sort_values(ascending=False)
        churn_by.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Churn rate par {col}")
        ax.set_xlabel("")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()


# %% Churn vs MonthlyCharges
df_pd = df.select(["MonthlyCharges", TARGET]).to_pandas()

plt.figure(figsize=(8, 4))
for label, color in [(0, "steelblue"), (1, "tomato")]:
    subset = df_pd[df_pd[TARGET] == label]["MonthlyCharges"]
    plt.hist(subset, bins=40, alpha=0.6, color=color,
             label="No Churn" if label == 0 else "Churn", edgecolor="white")
plt.title("MonthlyCharges : Churn vs No Churn")
plt.xlabel("Monthly Charges ($)")
plt.ylabel("Nombre de clients")
plt.legend()
plt.tight_layout()
plt.show()

# %%  Corrélation numérique
corr = df.select(NUMERICAL_COLS + [TARGET]).to_pandas().corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()