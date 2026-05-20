# %% Imports
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import sys
sys.path.append("..")
import os
os.makedirs("../report", exist_ok=True)
os.makedirs("../models", exist_ok=True)
print("Directories ready.")

from src.data_loader import load_data, drop_id  # noqa: E402
from src.config import MODELS_DIR, TARGET, RANDOM_STATE  # noqa: E402

sns.set_theme(style="whitegrid")

# Chargement
X_train, X_test, y_train, y_test, feature_names = joblib.load(MODELS_DIR / "train_test_data.pkl")
calibrated_model = joblib.load(MODELS_DIR / "xgb_calibrated.pkl")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

# Dataset original pour CLV
df = load_data()
df = drop_id(df)

print("Loaded.")

# %% SHAP values globales
# Extraire le XGBoost sous-jacent du modèle calibré
xgb_base = calibrated_model.calibrated_classifiers_[0].estimator

explainer = shap.TreeExplainer(xgb_base)
shap_values = explainer.shap_values(X_test)

joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")

# Beeswarm plot — top 15 features
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test,
                  feature_names=feature_names,
                  max_display=15,
                  show=False)
plt.title("SHAP — Feature Importance Globale")
plt.tight_layout()
plt.savefig("../report/shap_global.png", dpi=150)
plt.show()


# %% SHAP bar plot
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test,
                  feature_names=feature_names,
                  plot_type="bar",
                  max_display=15,
                  show=False)
plt.title("SHAP — Mean Absolute Impact")
plt.tight_layout()
plt.savefig("../report/shap_bar.png", dpi=150)
plt.show()

# %% Reconstruction du dataset test avec probabilités
df_pd = df.to_pandas()
from sklearn.model_selection import train_test_split  # noqa: E402
from src.config import TEST_SIZE  # noqa: E402

# Même cast que dans le preprocessing
df_pd["SeniorCitizen"] = df_pd["SeniorCitizen"].astype(str)

_, df_test = train_test_split(
    df_pd, test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_pd[TARGET]
)
df_test = df_test.reset_index(drop=True)

# Probabilités calibrées
X_test_raw = df_test.drop(columns=[TARGET])
y_proba_cal = calibrated_model.predict_proba(
    preprocessor.transform(X_test_raw)
)[:, 1]

df_test["ChurnProbability"] = y_proba_cal
print(df_test[["MonthlyCharges", "tenure", "ChurnProbability"]].head())

# %% CLV et RetentionPriority
# CLV simplifiée
df_test["CLV"] = df_test["MonthlyCharges"] * df_test["tenure"]

# Clients avec tenure=0 → CLV=0, on remplace par MonthlyCharges
df_test["CLV"] = df_test.apply(
    lambda r: r["MonthlyCharges"] if r["CLV"] == 0 else r["CLV"], axis=1
)

# Score de priorité
df_test["RetentionPriority"] = df_test["ChurnProbability"] * df_test["CLV"]

print(df_test[["MonthlyCharges", "tenure", "CLV", "ChurnProbability", "RetentionPriority"]].describe())

# %% Segmentation CRM
churn_threshold = 0.5
priority_high = df_test["RetentionPriority"].quantile(0.75)
priority_mid  = df_test["RetentionPriority"].quantile(0.40)

def assign_segment(row):
    if row["ChurnProbability"] >= churn_threshold and row["RetentionPriority"] >= priority_high:
        return "A — High Priority"
    elif row["ChurnProbability"] >= 0.3 or row["RetentionPriority"] >= priority_mid:
        return "B — Medium Priority"
    else:
        return "C — Low Priority"

df_test["Segment"] = df_test.apply(assign_segment, axis=1)

print(df_test["Segment"].value_counts())
print("\nChurn rate par segment:")
print(df_test.groupby("Segment")[TARGET].mean().round(3))


# %% Visualisation segmentation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution des segments
colors = {"A — High Priority": "tomato",
          "B — Medium Priority": "orange",
          "C — Low Priority": "steelblue"}

segment_counts = df_test["Segment"].value_counts()
axes[0].bar(segment_counts.index, segment_counts.values,
            color=[colors[s] for s in segment_counts.index])
axes[0].set_title("Nombre de clients par segment")
axes[0].set_ylabel("Clients")
axes[0].tick_params(axis="x", rotation=15)

# Scatter CLV vs ChurnProbability
for seg, color in colors.items():
    mask = df_test["Segment"] == seg
    axes[1].scatter(
        df_test.loc[mask, "ChurnProbability"],
        df_test.loc[mask, "CLV"],
        alpha=0.4, s=15, color=color, label=seg
    )
axes[1].set_xlabel("Churn Probability")
axes[1].set_ylabel("CLV ($)")
axes[1].set_title("Segmentation : CLV × ChurnProbability")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("../report/segmentation.png", dpi=150)
plt.show()

# %% Business Case & ROI
avg_monthly_revenue = df_test["MonthlyCharges"].mean()
segment_a = df_test[df_test["Segment"] == "A — High Priority"]

n_at_risk = len(segment_a)
revenue_at_risk_monthly = segment_a["MonthlyCharges"].sum()
revenue_at_risk_annual  = revenue_at_risk_monthly * 12

print("=== BUSINESS CASE ===")
print(f"Clients Segment A (high risk):     {n_at_risk}")
print(f"Revenu mensuel à risque:           ${revenue_at_risk_monthly:,.0f}")
print(f"Revenu annuel à risque:            ${revenue_at_risk_annual:,.0f}")

# Simulation ROI
print("\n=== ROI SIMULATION ===")
for conversion_rate in [0.10, 0.20, 0.30]:
    saved = revenue_at_risk_annual * conversion_rate
    campaign_cost = n_at_risk * 15  # $15 par client contacté
    roi = (saved - campaign_cost) / campaign_cost * 100
    print(f"Conversion {int(conversion_rate*100)}% → "
          f"Revenu sauvé: ${saved:,.0f} | "
          f"Coût campagne: ${campaign_cost:,.0f} | "
          f"ROI: {roi:.0f}%")
    
# %% Sauvegarde dataset final
import os  # noqa: E402
os.makedirs("../data", exist_ok=True)

df_test.to_csv("../data/df_scored.csv", index=False)
print("Scored dataset saved → data/df_scored.csv")
print(f"\nColonnes finales: {list(df_test.columns)}")


# %%
