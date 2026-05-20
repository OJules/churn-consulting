# %% Imports
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append("..")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

from src.config import MODELS_DIR, RANDOM_STATE

# Chargement des données preprocessées
X_train, X_test, y_train, y_test, feature_names = joblib.load(
    MODELS_DIR / "train_test_data.pkl"
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% Baseline : Logistic Regression
mlflow.set_experiment("churn-consulting")

with mlflow.start_run(run_name="logistic_regression"):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_proba_lr)

    mlflow.log_metric("auc_roc", auc_lr)
    mlflow.sklearn.log_model(lr, "logistic_regression")

    print(f"Logistic Regression AUC-ROC: {auc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["No Churn", "Churn"]))


# %% XGBoost sans balancing
with mlflow.start_run(run_name="xgboost_no_balancing"):
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0
    )
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_proba_xgb)

    mlflow.log_metric("auc_roc", auc_xgb)
    mlflow.sklearn.log_model(xgb, "xgboost")

    print(f"XGBoost AUC-ROC: {auc_xgb:.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=["No Churn", "Churn"]))

# %% XGBoost avec class weights

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

with mlflow.start_run(run_name="xgboost_class_weight"):
    xgb_w = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0
    )
    xgb_w.fit(X_train, y_train)

    y_pred_xgb_w = xgb_w.predict(X_test)
    y_proba_xgb_w = xgb_w.predict_proba(X_test)[:, 1]
    auc_xgb_w = roc_auc_score(y_test, y_proba_xgb_w)

    mlflow.log_metric("auc_roc", auc_xgb_w)
    mlflow.sklearn.log_model(xgb_w, "xgboost_weighted")

    print(f"XGBoost Weighted AUC-ROC: {auc_xgb_w:.4f}")
    print(classification_report(y_test, y_pred_xgb_w, target_names=["No Churn", "Churn"]))

# %% Comparaison des courbes ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
for model, proba, name in [
    (lr, y_proba_lr, "Logistic Regression"),
    (xgb, y_proba_xgb, "XGBoost"),
    (xgb_w, y_proba_xgb_w, "XGBoost Weighted"),
]:
    RocCurveDisplay.from_predictions(
        y_test, proba, name=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})",
        ax=axes[0]
    )
axes[0].set_title("ROC Curves")
axes[0].plot([0,1],[0,1],"k--")

# Precision-Recall
for proba, name in [
    (y_proba_lr, "Logistic Regression"),
    (y_proba_xgb, "XGBoost"),
    (y_proba_xgb_w, "XGBoost Weighted"),
]:
    PrecisionRecallDisplay.from_predictions(
        y_test, proba, name=name, ax=axes[1]
    )
axes[1].set_title("Precision-Recall Curves")

plt.tight_layout()
plt.show()

# %% sauvegarde du meilleur modèle
# On sauvegarde XGBoost weighted comme modèle principal
joblib.dump(xgb_w, MODELS_DIR / "xgb_model.pkl")
print("Best model saved.")